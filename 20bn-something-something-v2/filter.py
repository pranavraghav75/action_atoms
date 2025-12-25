import json
import shutil
from pathlib import Path

TARGET_LABELS = {
    "Moving [something] closer to [something]",
    "Moving [something] away from [something]"
}

VIDEO_DIR = Path("videos")
OUT_DIR = Path("filtered_videos")
OUT_DIR.mkdir(exist_ok=True)

label_map = {}

for split in ["train", "validation"]:
    with open(f"labels/{split}.json") as f:
        for entry in json.load(f):
            label_map[entry["id"]] = entry["template"]

for vid, label in label_map.items():
    if label in TARGET_LABELS:
        src = VIDEO_DIR / f"{vid}.webm"
        if src.exists():
            shutil.move(src, OUT_DIR / src.name)
