import random, shutil
from pathlib import Path
import tensorflow as tf

def split_dataset(raw="data/raw", train="data/train", test="data/test", split=0.8, seed=42):
    random.seed(seed)
    for class_dir in Path(raw).iterdir():
        if not class_dir.is_dir():
            continue
        images = sorted(class_dir.glob("*"))
        random.shuffle(images)
        cut = int(len(images) * split)
        for subset, files in [(Path(train), images[:cut]), (Path(test), images[cut:])]:
            dest = subset / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)

def load_datasets(train_dir="data/train", test_dir="data/test", img_size=(224, 224), batch=32):
    kw = dict(image_size=img_size, batch_size=batch)
    train = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="training", seed=42, **kw)
    val = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=0.2, subset="validation", seed=42, **kw)
    test = tf.keras.utils.image_dataset_from_directory(test_dir, shuffle=False, **kw)
    return train, val, test
