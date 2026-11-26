"""One-off script: compute dataset statistics for the UI "Insights" page.
Run from the project root, on the folder that holds ALL images in class folders:
    python src/make_stats.py data/raw
The result (ui/dataset_stats.json) is small, so commit it to GitHub. That way the
Insights page works even if the repo only holds a small sample of the images.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
OUT = Path("ui/dataset_stats.json")
SAMPLE = 100  # images measured per class (counts use every image)
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif"}

random.seed(42)
stats = {}
for class_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
    all_files = [f for f in class_dir.rglob("*") if f.is_file()]
    files = [f for f in all_files if f.suffix.lower() in EXT]
    others = Counter(f.suffix.lower() for f in all_files if f.suffix.lower() not in EXT)
    if others:
        print(f"  note: {class_dir.name} has non-image files ignored: {dict(others)}")
    if not files:
        print(f"SKIPPED '{class_dir.name}': no images found (is it really a class folder?)")
        continue

    rows = []
    for f in random.sample(files, min(SAMPLE, len(files))):
        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue  # unreadable file, skip it
        a = np.asarray(img.resize((128, 128)), dtype=np.float32)
        rows.append((img.size[0], img.size[1], a.mean(),
                     a[..., 0].mean(), a[..., 1].mean(), a[..., 2].mean()))
    if not rows:
        print(f"SKIPPED '{class_dir.name}': no readable images")
        continue

    m = np.mean(rows, axis=0)
    stats[class_dir.name] = {"count": len(files), "width": float(m[0]),
                             "height": float(m[1]), "brightness": float(m[2]),
                             "R": float(m[3]), "G": float(m[4]), "B": float(m[5])}
    print(class_dir.name, {k: round(v, 1) for k, v in stats[class_dir.name].items()})

if not stats:
    sys.exit("No classes found. Check the folder you passed in.")

OUT.parent.mkdir(exist_ok=True)
json.dump(stats, open(OUT, "w"), indent=2)
print("Saved", OUT, "with classes:", list(stats))
