import json
import random
import shutil
import time
from pathlib import Path

import tensorflow as tf

MODEL_PATH = Path("models/maize_model.h5")
CLASSES_PATH = Path("models/class_names.json")
LOG_PATH = Path("models/retrain_log.json")
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")
NEW_DIR = Path("data/new")          # uploaded images: data/new/<class>/
TMP_DIR = Path("data/retrain_tmp")  # temporaryy mix of new + old images
IMG_SIZE = (224, 224)
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _images(folder):
    return [f for f in Path(folder).rglob("*") if f.suffix.lower() in IMG_EXTS]


def count_new_images():
    """How many uploaded images are waiting to be used for retraining."""
    return len(_images(NEW_DIR)) if NEW_DIR.exists() else 0


def read_log():
    """History of all retraining runs (used by the UI)."""
    return json.load(open(LOG_PATH)) if LOG_PATH.exists() else []


def retrain(epochs=3, replay_per_class=100, batch=32):
    class_names = json.load(open(CLASSES_PATH))
    new_files = [(f, f.parent.name) for f in _images(NEW_DIR)]
    if not new_files:
        raise ValueError("No new images to retrain on.")

    # 1. Build a small training folder: all new images + a random sample of old
    #    ones (so the model does not forget what it already learned).
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    for c in class_names:
        (TMP_DIR / c).mkdir(parents=True)
    for f, label in new_files:
        shutil.copy(f, TMP_DIR / label / f.name)
    random.seed(42)
    for c in class_names:
        old = _images(TRAIN_DIR / c)
        for f in random.sample(old, min(replay_per_class, len(old))):
            shutil.copy(f, TMP_DIR / c / f.name)

    kw = dict(image_size=IMG_SIZE, batch_size=batch, class_names=class_names)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TMP_DIR, validation_split=0.2, subset="training", seed=42, **kw)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TMP_DIR, validation_split=0.2, subset="validation", seed=42, **kw)
    test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, shuffle=False, **kw)

    # 2. Load the current model, measure it, train it a little more, measure again.
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    _, acc_before = model.evaluate(test_ds, verbose=0)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
    _, acc_after = model.evaluate(test_ds, verbose=0)

    # 3. Keep a backup of the old model, save the new one.
    shutil.copy(MODEL_PATH, MODEL_PATH.with_name("maize_model_prev.h5"))
    model.save(MODEL_PATH)

    # 4. The new images now belong to the training set.
    for f, label in new_files:
        shutil.move(str(f), TRAIN_DIR / label / f.name)
    shutil.rmtree(TMP_DIR)

    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
             "new_images": len(new_files), "epochs": epochs,
             "test_accuracy_before": round(float(acc_before), 4),
             "test_accuracy_after": round(float(acc_after), 4)}
    log = read_log()
    log.append(entry)
    json.dump(log, open(LOG_PATH, "w"), indent=2)
    return entry

