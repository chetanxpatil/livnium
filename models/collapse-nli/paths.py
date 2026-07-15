"""Shared dataset and checkpoint paths for the collapse-NLI experiments."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNLI_DIR = ROOT / "benchmarks" / "nli" / "data" / "snli"
SNLI_TRAIN = str(SNLI_DIR / "snli_1.0_train.jsonl")
SNLI_DEV = str(SNLI_DIR / "snli_1.0_dev.jsonl")
SNLI_TEST = str(SNLI_DIR / "snli_1.0_test.jsonl")
NLI_CHECKPOINT = str(HERE / "model_nli_v1" / "nli_epoch20.pt")
