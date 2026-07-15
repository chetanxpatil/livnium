"""Canonical data and checkpoint paths for vision experiments."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
COCO_DIR = HERE / "data" / "coco"
COCO_IMAGES = str(COCO_DIR / "val2017")
COCO_CAPTIONS = str(COCO_DIR / "annotations" / "captions_val2017.json")


def model_path(name):
    return str(MODEL_DIR / name)
