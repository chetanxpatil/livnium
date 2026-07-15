"""Canonical local paths for the chat-brain experiment."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = HERE / "data"
MODEL_DIR = HERE / "model"
RAW_EXPORT = ROOT / "conversations.json"
NOUN_CHECKPOINT = ROOT / "models" / "noun-collapse" / "model" / "noun_collapse_pure.pt"


def data_path(name):
    return str(DATA_DIR / name)


def model_path(name):
    return str(MODEL_DIR / name)
