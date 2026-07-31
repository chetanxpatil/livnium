#!/usr/bin/env python3
"""
Held-out SNLI ablations for the cube_embed archive.

The source program trains on SNLI train but repeatedly selects on dev and then
reports dev.  This read-only control uses the archive's otherwise-unused SNLI
test split:

  * train: first 10,000 valid train examples, shuffled with seed 42
  * model selection: first 2,000 valid dev examples
  * final report: first 2,000 valid test examples
  * frozen sentence parameters: alpha=0.7, beta=0.5 (source replay winner)
  * deterministic PYTHONHASHSEED=0 required for the OOV character fallback

Feature ablations use standardized multinomial logistic regression.  C is
selected only on dev; test is never used for model selection.

Run from the archive root:
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
      python3 .codex_memory_staging/CUBE_EMBED_SNLI_ABLATION_PROBE.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from cube_embed import angle_ops
from cube_embed.cooc_ops import build_semantic_vectors
from cube_embed.counter_fit import counter_fit
from cube_embed.sentence import (
    SentenceField,
    negation_features,
    pair_features,
    tokenize,
)


DATA = ROOT / "data"
CUBE = ROOT / "cube_embed"
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
N_TRAIN, N_DEV, N_TEST = 10_000, 2_000, 2_000

ANTONYM_SEEDS = [
    ("north","south"),("east","west"),("large","small"),("big","small"),
    ("hot","cold"),("fast","slow"),("old","new"),("old","young"),
    ("black","white"),("dark","bright"),("hard","soft"),("tall","short"),
    ("rich","poor"),("strong","weak"),("good","bad"),("full","empty"),
    ("open","close"),("start","stop"),("buy","sell"),("give","take"),
    ("push","pull"),("build","destroy"),("love","hate"),("win","lose"),
    ("alive","dead"),("wet","dry"),("clean","dirty"),("safe","dangerous"),
    ("up","down"),("left","right"),("before","after"),("enter","exit"),
    ("increase","decrease"),("add","remove"),("happy","sad"),("loud","quiet"),
    ("wide","narrow"),("thick","thin"),("smooth","rough"),("early","late"),
]
SYNONYM_SEEDS = [
    ("big","large"),("small","little"),("fast","quick"),("smart","intelligent"),
    ("happy","joyful"),("sad","unhappy"),("begin","start"),("stop","cease"),
    ("talk","speak"),("buy","purchase"),("help","assist"),("show","display"),
    ("run","sprint"),("walk","stride"),("build","construct"),("change","alter"),
    ("kill","murder"),("find","discover"),("see","look"),("move","travel"),
    ("car","automobile"),("film","movie"),("physician","doctor"),
    ("child","kid"),("man","person"),("woman","lady"),
]


def load_snli(path: Path, n: int) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            if ex.get("gold_label") not in LABEL_MAP:
                continue
            rows.append(
                {
                    "premise": ex["sentence1"],
                    "hypothesis": ex["sentence2"],
                    "label": LABEL_MAP[ex["gold_label"]],
                }
            )
            if len(rows) >= n:
                break
    return rows


def mean_base(tokens: List[str]) -> np.ndarray:
    vecs = [angle_ops.word_base_vector(t) for t in tokens if t]
    if not vecs:
        return np.zeros(27, dtype=np.float32)
    vec = np.mean(vecs, axis=0)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec.astype(np.float32)


def build_feature_bank(
    examples: List[dict], sf: SentenceField, name: str
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    n = len(examples)
    cube = np.zeros((n, 376), dtype=np.float32)
    neg = np.zeros((n, 10), dtype=np.float32)
    svd_cos = np.zeros((n, 1), dtype=np.float32)
    svd_pair = np.zeros((n, 108), dtype=np.float32)
    labels = np.asarray([x["label"] for x in examples], dtype=np.int32)

    for i, ex in enumerate(examples):
        if i % 1000 == 0:
            print(f"  {name}: {i}/{n}", flush=True)
        p = tokenize(ex["premise"])
        h = tokenize(ex["hypothesis"])
        u = sf.embed(p)
        v = sf.embed(h)
        full_pair = pair_features(u, v, p, h)
        cube[i] = full_pair[:376]
        neg[i] = negation_features(p, h) * 5.0

        ps = mean_base(p)
        hs = mean_base(h)
        svd_cos[i, 0] = float(np.dot(ps, hs))
        svd_pair[i] = np.concatenate([ps, hs, ps * hs, np.abs(ps - hs)])
    print(f"  {name}: {n}/{n} done", flush=True)
    return {
        "cube376": cube,
        "neg10": neg,
        "svd_cos1": svd_cos,
        "svd_pair108": svd_pair,
    }, labels


def assemble(bank: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    c, n, s, p = (
        bank["cube376"],
        bank["neg10"],
        bank["svd_cos1"],
        bank["svd_pair108"],
    )
    return {
        "cube only (376)": c,
        "cube + neg (386)": np.concatenate([c, n], axis=1),
        "cube + SVD cosine (377)": np.concatenate([c, s], axis=1),
        "source full (387)": np.concatenate([c, n, s], axis=1),
        "neg + SVD cosine (11)": np.concatenate([n, s], axis=1),
        "SVD sentence pair (108)": p,
        "SVD sentence pair + neg (118)": np.concatenate([p, n], axis=1),
        "neg only (10)": n,
        "SVD cosine only (1)": s,
    }


def tune_and_test(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float, float]:
    candidates = (0.01, 0.1, 1.0, 10.0)
    best_c, best_dev, best_model = candidates[0], -1.0, None
    for c in candidates:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c,
                max_iter=500,
                solver="lbfgs",
                random_state=42,
            ),
        )
        model.fit(x_train, y_train)
        dev = float(model.score(x_dev, y_dev))
        if dev > best_dev:
            best_c, best_dev, best_model = c, dev, model
    assert best_model is not None
    test = float(best_model.score(x_test, y_test))
    print(
        f"{name:<31} C={best_c:<5g} dev={best_dev:.4f} test={test:.4f}",
        flush=True,
    )
    return best_c, best_dev, test


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Set PYTHONHASHSEED=0 for reproducible OOV vectors.")
    random.seed(42)
    np.random.seed(42)

    print("=" * 78)
    print("CUBE_EMBED HELD-OUT SNLI ABLATION")
    print("=" * 78)
    train_path = DATA / "snli_1.0_train.jsonl"
    dev_path = DATA / "snli_1.0_dev.jsonl"
    test_path = DATA / "snli_1.0_test.jsonl"
    for path in (train_path, dev_path, test_path):
        print(f"{path.name}: exists={path.exists()} bytes={path.stat().st_size}")

    print("\nBuilding the same WikiText-valid base space used by the source...")
    snli_freq: Dict[str, int] = {}
    with train_path.open(encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            if ex.get("gold_label") not in LABEL_MAP:
                continue
            for sentence in (ex["sentence1"], ex["sentence2"]):
                for token in tokenize(sentence):
                    snli_freq[token] = snli_freq.get(token, 0) + 1
    targets = [
        w
        for w, count in sorted(snli_freq.items(), key=lambda item: -item[1])
        if count >= 5
    ]
    _vocab, vecs = build_semantic_vectors(
        wiki_path=str(CUBE / "wikitext-103" / "wiki.valid.tokens"),
        vocab_size=8000,
        dim=27,
        target_vocab=targets,
        verbose=True,
    )
    fitted = counter_fit(
        vecs,
        SYNONYM_SEEDS,
        ANTONYM_SEEDS,
        n_iter=100,
        lr=0.05,
        gamma_syn=0.8,
        gamma_ant=-0.3,
        reg=0.05,
        verbose=False,
    )
    angle_ops.set_semantic_vectors(fitted)
    print(f"semantic_vocab={len(fitted)} target_words={len(targets)}")

    train = load_snli(train_path, N_TRAIN)
    dev = load_snli(dev_path, N_DEV)
    test = load_snli(test_path, N_TEST)
    random.shuffle(train)
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        counts = [sum(x["label"] == label for x in rows) for label in range(3)]
        print(f"{name}: n={len(rows)} labels={counts}")

    sf = SentenceField(alpha=0.7, beta=0.5)
    started = time.time()
    train_bank, y_train = build_feature_bank(train, sf, "train")
    dev_bank, y_dev = build_feature_bank(dev, sf, "dev")
    test_bank, y_test = build_feature_bank(test, sf, "test")
    print(f"feature_build_seconds={time.time()-started:.1f}")

    train_features = assemble(train_bank)
    dev_features = assemble(dev_bank)
    test_features = assemble(test_bank)

    print("\nCondition                       selected-C  dev      held-out-test")
    print("-" * 78)
    results = {}
    for name in train_features:
        results[name] = tune_and_test(
            name,
            train_features[name],
            y_train,
            dev_features[name],
            y_dev,
            test_features[name],
            y_test,
        )

    majority_label = int(np.bincount(y_train).argmax())
    majority_dev = float(np.mean(y_dev == majority_label))
    majority_test = float(np.mean(y_test == majority_label))
    print(
        f"{'train-majority':<31} C={'--':<5} "
        f"dev={majority_dev:.4f} test={majority_test:.4f}"
    )

    full_test = results["source full (387)"][2]
    no_cube_test = results["neg + SVD cosine (11)"][2]
    svd_pair_test = results["SVD sentence pair + neg (118)"][2]
    cube_only_test = results["cube only (376)"][2]
    print("\nKey deltas on held-out test")
    print(f"full minus neg+SVD-cosine = {full_test-no_cube_test:+.4f}")
    print(f"full minus SVD-pair+neg   = {full_test-svd_pair_test:+.4f}")
    print(f"cube-only minus majority  = {cube_only_test-majority_test:+.4f}")
    angle_ops.clear_semantic_vectors()


if __name__ == "__main__":
    main()
