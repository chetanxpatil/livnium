"""Shared external-data paths for language probes."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNLI_DIR = ROOT / "benchmarks" / "nli" / "data" / "snli"
SNLI_TRAIN = str(SNLI_DIR / "snli_1.0_train.jsonl")
