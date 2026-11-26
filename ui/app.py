"""Streamlit UI. It talks to the FastAPI service.
Run:  streamlit run ui/app.py
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = API_URL = "https://corn-or-maize-leaf-disease-api.onrender.com"
STATS_PATH = Path(__file__).parent / "dataset_stats.json"

st.set_page_config(page_title="Maize Leaf Disease", page_icon="🌽", layout="wide")


def call_api(method, path, **kwargs):
    """Call the API; show a friendly error and return None if it fails."""
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=kwargs.pop("timeout", 30), **kwargs)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            st.error(f"API error {r.status_code}: {detail}")
            return None
        return r.json()
    except requests.exceptions.RequestException:
        st.error(f"Cannot reach the API at {API_URL}. Is it running?")
        return None


def fmt_uptime(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


st.sidebar.title("🌽 Maize Leaf Disease")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict", "Insights", "Upload & Retrain"])
st.sidebar.caption(f"API: {API_URL}")

#  Dashboard
if page == "Dashboard":
    st.title("Model dashboard")
    health = call_api("GET", "/health")
    if health:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", "Online ✅" if health["status"] == "ok" else "Problem")
        c2.metric("Uptime", fmt_uptime(health["uptime_seconds"]))
        c3.metric("New images waiting", health["pending_new_images"])
        c4.metric("Retraining now?", "Yes" if health["retraining"] else "No")
        st.write("**Classes:**", ", ".join(health["classes"]))
        st.caption(f"Automatic retraining starts when {health['auto_retrain_threshold']} "
                   "or more new images are waiting.")
        history = call_api("GET", "/retrain/history")
        if history:
            st.subheader("Latest retraining")
            last = history[-1]
            a, b, c = st.columns(3)
            a.metric("Accuracy before", f"{last['test_accuracy_before']:.3f}")
            b.metric("Accuracy after", f"{last['test_accuracy_after']:.3f}",
                     delta=round(last["test_accuracy_after"] - last["test_accuracy_before"], 4))
            c.metric("New images used", last["new_images"])
            st.caption(f"Finished at {last['time']}")
    st.button("Refresh")

#  Predict
elif page == "Predict":
    st.title("Predict one leaf image")
    file = st.file_uploader("Upload a maize leaf photo", type=["jpg", "jpeg", "png"])
    if file:
        st.image(file, width=350)
        if st.button("Predict", type="primary"):
            with st.spinner("Running the model..."):
                res = call_api("POST", "/predict",
                               files={"file": (file.name, file.getvalue(), file.type)})
            if res:
                st.success(f"Prediction: **{res['class']}** ({res['confidence']:.1%} confidence)")
                probs = pd.Series(res["probabilities"], name="probability")
                st.bar_chart(probs)

#  Insights
elif page == "Insights":
    st.title("What the data tells us")
    if not STATS_PATH.exists():
        st.warning("ui/dataset_stats.json not found. Run: python src/make_stats.py data/raw")
        st.stop()
    stats = pd.DataFrame(json.load(open(STATS_PATH))).T

    st.subheader("1. Class balance")
    st.bar_chart(stats["count"])
    st.caption("Gray Leaf Spot has far fewer images than the other classes, so the model "
               "sees it less often. I used class weights, and I watch its recall and precision.")

    st.subheader("2. Brightness and colour")
    c1, c2 = st.columns(2)
    c1.bar_chart(stats["brightness"])
    c2.bar_chart(stats[["R", "G", "B"]])
    st.caption("Healthy leaves are the greenest and brightest. Common Rust is the darkest. "
               "Blight and Gray Leaf Spot have almost the same colour profile, which explains "
               "why the model confuses them most.")

    st.subheader("3. Image size")
    st.bar_chart(stats[["width", "height"]])
    st.caption("Healthy images are all the same size, while diseased ones vary. This hints at "
               "different sources, so real field photos may be harder than the test set.")

    st.dataframe(stats.round(1))

#  Upload & Retrain
else:
    st.title("Upload new data and retrain")
    health = call_api("GET", "/health")
    if not health:
        st.stop()

    st.subheader("1. Upload labelled images")
    label = st.selectbox("All images in this upload belong to class:", health["classes"])
    files = st.file_uploader("Choose images (you can select many)",
                             type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if files and st.button(f"Upload {len(files)} images as {label}"):
        payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        with st.spinner("Uploading..."):
            res = call_api("POST", "/upload", data={"label": label}, files=payload, timeout=300)
        if res:
            st.success(f"Saved {res['saved']} images. {res['pending_new_images']} waiting for retraining.")
            if res["auto_retrain_triggered"]:
                st.info("The automatic trigger fired: retraining has started.")

    st.divider()
    st.subheader("2. Retrain")
    st.write(f"**{health['pending_new_images']}** new images waiting "
             f"(automatic retraining at {health['auto_retrain_threshold']}).")
    if st.button("Retrain now", type="primary"):
        if call_api("POST", "/retrain"):
            st.info("Retraining started. This can take a few minutes.")
            with st.spinner("Retraining..."):
                while True:
                    time.sleep(5)
                    status = call_api("GET", "/retrain/status")
                    if not status or not status["retraining"]:
                        break
            if status and status["error"]:
                st.error(f"Retraining failed: {status['error']}")
            elif status and status["last_result"]:
                r = status["last_result"]
                st.success(f"Done. Test accuracy {r['test_accuracy_before']:.3f} → "
                           f"{r['test_accuracy_after']:.3f}")

    st.divider()
    st.subheader("3. Retraining history")
    if st.button("Refresh history"):
        pass
    history = call_api("GET", "/retrain/history")
    if history:
        df = pd.DataFrame(history)
        st.dataframe(df)
        st.line_chart(df.set_index("time")[["test_accuracy_before", "test_accuracy_after"]])
    else:
        st.caption("No retraining runs yet.")
