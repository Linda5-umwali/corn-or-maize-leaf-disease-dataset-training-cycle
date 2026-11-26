"""FastAPI app: predict, bulk upload, retrain trigger, uptime/health.
Run from the project root:  uvicorn api.main:app --reload
"""
import io
import time
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.prediction import load_model, predict_image
from src.retrain import IMG_EXTS, NEW_DIR, count_new_images, read_log, retrain

AUTO_RETRAIN_THRESHOLD = 50  # automatic trigger: retrain when this many new images are waiting

app = FastAPI(title="Maize Leaf Disease API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

START_TIME = time.time()
model, class_names = load_model()
state = {"retraining": False, "last_result": None, "error": None}


def run_retrain():
    """Runs in the background so the API stays responsive while training."""
    global model
    state["error"] = None
    try:
        state["last_result"] = retrain()
        model, _ = load_model()  # start using the new model immediately
    except Exception as e:
        state["error"] = str(e)
    finally:
        state["retraining"] = False


def start_retrain(background_tasks: BackgroundTasks):
    state["retraining"] = True  # set right away so two retrains can't start together
    background_tasks.add_task(run_retrain)


@app.get("/health")
def health():
    return {"status": "ok",
            "uptime_seconds": round(time.time() - START_TIME),
            "classes": class_names,
            "pending_new_images": count_new_images(),
            "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD,
            "retraining": state["retraining"]}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    data = file.file.read()
    try:
        return predict_image(model, class_names, io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Could not read that image.")


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks,
                 label: str = Form(...),
                 files: List[UploadFile] = File(...)):
    """Bulk upload: many images that all belong to one class (label)."""
    if label not in class_names:
        raise HTTPException(400, f"label must be one of {class_names}")
    folder = NEW_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    saved = 0
    for i, f in enumerate(files):
        ext = Path(f.filename or "").suffix.lower()
        if ext not in IMG_EXTS:
            continue
        (folder / f"{stamp}_{i}{ext}").write_bytes(await f.read())
        saved += 1
    if saved == 0:
        raise HTTPException(400, "No valid images (.jpg, .jpeg, .png) were uploaded.")

    pending = count_new_images()
    auto_triggered = False
    if pending >= AUTO_RETRAIN_THRESHOLD and not state["retraining"]:
        start_retrain(background_tasks)  # automatic trigger
        auto_triggered = True
    return {"saved": saved, "pending_new_images": pending,
            "auto_retrain_triggered": auto_triggered}


@app.post("/retrain")
def retrain_now(background_tasks: BackgroundTasks):
    """Manual trigger (the button in the UI calls this)."""
    if state["retraining"]:
        raise HTTPException(409, "Retraining is already running.")
    if count_new_images() == 0:
        raise HTTPException(400, "Upload new images first.")
    start_retrain(background_tasks)
    return {"message": "Retraining started."}


@app.get("/retrain/status")
def retrain_status():
    return state


@app.get("/retrain/history")
def retrain_history():
    return read_log()
