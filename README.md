# Maize Leaf Disease Classification: End-to-End ML Pipeline

> **Video demo:** [YouTube link](https://studio.youtube.com/video/PuskKr0AMxI/edit)
> **Live API URL:** [API URL](https://corn-or-maize-leaf-disease-api.onrender.com)
> **GitHub repo:** [github repo link](https://github.com/Linda5-umwali/corn-or-maize-leaf-disease-dataset-training-cycle.git)

---

## 1. Project Description

This project builds a complete machine learning pipeline that classifies **maize (corn) leaf images** into four categories:

| Class | Meaning |
|---|---|
| `Blight` | Northern corn leaf blight |
| `Common_Rust` | Common rust |
| `Gray_Leaf_Spot` | Gray leaf spot |
| `Healthy` | No disease |

**Why this matters:** Maize is a staple crop across Africa. Early, low-cost disease detection from a phone photo can help farmers act before a disease spreads to save tons of production.

**What the project covers:** data acquisition, preprocessing, model training and testing, an API, a web UI, retraining on newly uploaded data, cloud deployment, and a load test with Locust.

### Dataset
- **Source:** Kaggle, *Corn or Maize Leaf Disease Dataset* (images, not tabular)
- **Size:** 4,188 images (Blight 1,146 · Common_Rust 1,306 · Gray_Leaf_Spot 574 · Healthy 1,162)
- **Split:** stratified 80/20 (seed 42): 3,348 train (of which 20% is used for validation) and 840 test

### Model
- **MobileNetV2** (ImageNet weights) with transfer learning
- Data augmentation (flip, rotation, zoom), global average pooling, dropout 0.3, softmax output (4 classes)
- Class weights to handle the smaller Gray Leaf Spot class
- Saved as `models/maize_model.h5`

### Results on the test set

| Metric | Value |
|---|---|
| Accuracy | 0.917 |
| Macro precision | 0.898 |
| Macro recall | 0.916 |
| Macro F1 | 0.899 |
| Test loss | 0.198 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Blight | 0.951 | 0.761 | 0.845 |
| Common_Rust | 0.962 | 0.977 | 0.970 |
| Gray_Leaf_Spot | 0.677 | 0.930 | 0.784 |
| Healthy | 1.000 | 0.996 | 0.998 |

**Main weakness:** 48 of 230 Blight leaves were predicted as Gray Leaf Spot. Their colour and brightness statistics are almost identical, so the two classes look alike to the model.

---

## 2. Features

| Requirement | Where it is |
|---|---|
| Predict one image | UI "Predict" page / `POST /predict` |
| Visualizations (3+ feature stories) | UI "Insights" page and the notebook |
| Bulk upload of new data | UI "Upload" page / `POST /upload` |
| Trigger retraining | UI retrain button / `POST /retrain` |
| Model uptime | UI dashboard / `GET /health` |

### Retraining
1. New labelled images are uploaded to `data/new/<class>/`.
2. Retraining is triggered **manually** (button / `POST /retrain`) or **automatically** when 50 or more new images are waiting.
3. The pipeline mixes the new images with a random sample of old ones, fine-tunes the saved model for a few epochs, and evaluates it on the test set before and after.
4. The previous model is backed up (`models/maize_model_prev.h5`), the new model replaces it, and the results are logged in `models/retrain_log.json`.

---

## 3. Repository Structure

```
Project_name/
├── README.md
├── requirements.txt
├── notebook/
│   └── corn-or-maize-leaf-disease-dataset.ipynb
├── api/
│   └── main.py              # FastAPI app
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── retrain.py
├── ui/
├── locust/
│   └── locustfile.py
├── data/
│   ├── train/
│   ├──test/
│   └── raw/
├── models/
│   └── maize_model.h5
├── Dockerfile
└── docker-compose.yml
```

---

## 4. Setup Instructions

### Prerequisites
- Python 3.10 or newer
- Git
- Docker (optional, for containers)

### Run locally
```bash
# 1. Clone the repository
git clone https://github.com/Linda5-umwali/corn-or-maize-leaf-disease-dataset-training-cycle.git
cd corn-or-maize-leaf-disease-dataset-training-cycle

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Get the data and model
- **Model:** `models/maize_model.h5` and `models/class_names.json` are included in the repo.
- **Data:** download the Kaggle dataset, place the class folders in `data/raw/`, then create the split:
  ```bash
  python -c "from src.preprocessing import split_dataset; split_dataset()"
  ```
  Alternatively, a small sample is included in `data/`.

### Start the API
```bash
uvicorn api.main:app --reload
```
Open `http://127.0.0.1:8000/docs` to test all endpoints.

### Start the UI
```bash
[ADD YOUR UI COMMAND HERE, e.g. streamlit run ui/app.py]
```

### Run with Docker
```bash
docker compose up --build
```

### API endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Uptime, status, pending uploads |
| POST | `/predict` | Predict one image |
| POST | `/upload` | Bulk upload labelled images (form field `label` + `files`) |
| POST | `/retrain` | Start retraining |
| GET | `/retrain/status` | Is retraining running, last result |
| GET | `/retrain/history` | Accuracy before/after for each run |

---

## 5. Deployment

- **Platform:** [ AWS EC2]
- **Live URL:** [https://REPLACE_MEeee](https://REPLACE_ME)
- **How it is deployed:** [one or two lines, e.g. Docker image built from the Dockerfile and deployed as a web service]

---

## 6. Flood Request Simulation (Locust)

**Setup:** [Locust version] sending `POST /predict` with a sample leaf image.
**Test settings:** [e.g. 100 users, spawn rate 10 per second, run time 2 minutes]
**Run command:**
```bash
locust -f locust/locustfile.py --host http://[URL]
```

### Results by number of Docker containers
> Fill in from the Locust report for each run. Use the same user count and duration each time so the comparison is fair.

| Containers | Users | Requests/sec | Median latency (ms) | 95th percentile (ms) | Failures (%) |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

**Screenshots:** add your Locust charts under `locust/results/` and link them here.
`![1 container](locust/results/1_container.png)`

**Observations:** [Two or three sentences: how latency and throughput changed as you added containers, and where the limit was.]

---

## 7. Notebook

`notebook/project_name.ipynb` contains:
- Data acquisition and preprocessing
- Visualizations and interpretations (class balance, colour/brightness by class, image size)
- Model training (frozen base and fine-tuning) with loss and accuracy curves
- Evaluation: accuracy, precision, recall, F1, confusion matrix
- Prediction function and model saving

---

## 8. Limitations and Future Work
- Blight and Gray Leaf Spot are often confused; more examples and higher-resolution images would likely help.
- All Healthy images are 256×256 while the diseased images vary in size, which suggests different sources. Real field photos may score lower than the test set.
- Fine-tuning the MobileNetV2 layers did not improve validation loss, so the final model is effectively the frozen-base result.

---

## 9. Author
[  Linda Umwali], African Leadership University
