"""Locust load test for the maize API.

Setup (once):
    pip install locust
    put 10-20 leaf images in locust/sample_images/   (any class, .jpg/.png)

Run with the web page (http://localhost:8089):
    locust -f locust/locustfile.py --host http://localhost:8000

Run without a browser and save the numbers as CSV (good for the README):
    locust -f locust/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 -t 2m --csv locust/results/1_container
      -u = number of simulated users, -r = users started per second, -t = run time
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

HERE = Path(__file__).parent
EXTS = {".jpg", ".jpeg", ".png"}
FOLDER = HERE / "sample_images"
IMAGES = [p for p in FOLDER.glob("*") if p.suffix.lower() in EXTS] if FOLDER.exists() else []
if not IMAGES:  # fall back to the test set
    IMAGES = [p for p in (HERE.parent / "data" / "test").rglob("*") if p.suffix.lower() in EXTS][:20]
if not IMAGES:
    raise SystemExit("No images found. Put some in locust/sample_images/")

PAYLOADS = [(p.name, p.read_bytes(), "image/png" if p.suffix.lower() == ".png" else "image/jpeg")
            for p in IMAGES]


class MaizeUser(HttpUser):
    wait_time = between(0.5, 2)  # each simulated user pauses 0.5-2 s between requests

    @task(10)
    def predict(self):
        name, data, mime = random.choice(PAYLOADS)
        self.client.post("/predict", files={"file": (name, data, mime)}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
