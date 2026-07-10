"""
supervised_embed.py — labels SHAPE the wells, not just grade them.

The unsupervised runs (token_path_embed, gravity_embed) build structure from
co-occurrence and let the labels only judge it. Here the labels do the shaping:
a contrastive loss pulls ENTAILMENT pairs together and pushes CONTRADICTION pairs
apart, so token positions (and the descent codes read off them) form to serve the
task. Then the same path harness scores it.

Method
------
  - sentence vector m_S = mean of its token embeddings (bag of tokens)
  - sim(S1,S2) = cos(m1, m2)
  - loss per pair = (sim - y)^2 ,  y = 1 (entailment) / 0 (contradiction)
  - gradient descent on the token embedding matrix E, warm-started from the
    UNSUPERVISED PPMI-SVD vectors (so we measure what supervision ADDS)
  - read base-26 descent codes from trained E (recursive 26-way clustering)

Honesty guards
--------------
  - TRAIN/TEST SPLIT on pairs: E is trained on train pairs only, every AUC below
    is on HELD-OUT test pairs. Train AUC is also printed so overfitting is visible.
  - Self-contained corpus is "hard": entailment and contradiction pairs use the
    SAME topic words; the ONLY signal is a negation marker in the contradiction's
    second sentence. Co-occurrence is blind to it (the marker appears across all
    topics), so unsupervised should sit near chance and supervision has to earn
    the lift. --nli-path runs the same pipeline on real SNLI.

Scope: still bag-of-tokens (no cross-sentence interaction), so this is a floor on
what supervision buys, not a ceiling. cf. chat/SNLI_BASELINES.md.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "packages", "livnium-core", "src"))

from token_path_embed import (  # noqa: E402
    arbitrary_codes,
    auc,
    build_token_pings,
    distributional_vectors,
    flat_cosine,
    learned_codes,
    load_snli,
    soft_sim,
)

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# supervised trainer (manual gradients, numpy)
# --------------------------------------------------------------------------- #
def _cos_and_grads(m1, m2):
    n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
    if n1 == 0 or n2 == 0:
        return 0.0, np.zeros_like(m1), np.zeros_like(m2)
    cos = float(m1 @ m2 / (n1 * n2))
    g1 = m2 / (n1 * n2) - cos * m1 / (n1 * n1)
    g2 = m1 / (n1 * n2) - cos * m2 / (n2 * n2)
    return cos, g1, g2


def train_embeddings(pairs, vocab, E0, epochs=40, lr=0.5):
    E = E0.copy()
    idx = list(range(len(pairs)))
    for _ in range(epochs):
        RNG.shuffle(idx)
        for p in idx:
            a, b, y = pairs[p]
            if not a or not b:
                continue
            m1, m2 = E[a].mean(0), E[b].mean(0)
            cos, g1, g2 = _cos_and_grads(m1, m2)
            err = 2.0 * (cos - y)               # dL/dcos for L=(cos-y)^2
            E[a] -= lr * err * g1 / len(a)
            E[b] -= lr * err * g2 / len(b)
    return E


# --------------------------------------------------------------------------- #
# "hard" corpus: co-occurrence blind to the label
# --------------------------------------------------------------------------- #
def synthetic_hard():
    topics = {
        "vehicle": "car truck bus van bike road drive wheel",
        "food": "banana apple bread cheese rice soup meal plate",
        "music": "guitar piano drum song melody band stage tune",
        "weather": "rain sun cloud storm wind snow sky season",
        "sport": "ball goal team match score run game court",
    }
    tw = {t: w.split() for t, w in topics.items()}
    NEG = ["not", "never", "no", "fails", "without"]   # only in contradictions
    pairs: List[Tuple[List[str], List[str], int]] = []
    for t, words in tw.items():
        for _ in range(120):
            s1 = list(RNG.choice(words, 4, replace=True))
            s2 = list(RNG.choice(words, 4, replace=True))   # same topic, high overlap
            if RNG.random() < 0.5:
                pairs.append((s1, s2, 1))                    # entailment
            else:
                s2 = s2 + [RNG.choice(NEG)]                  # marker only signal
                pairs.append((s1, s2, 0))                    # contradiction
    RNG.shuffle(pairs)
    return pairs


def snli_pairs(path, max_pairs):
    _, raw = load_snli(path, max_pairs)
    return [(a, b, 1 if g == "entailment" else 0) for a, b, g in raw]


# --------------------------------------------------------------------------- #
def auc_flat(pairs, E):
    pos = [flat_cosine(a, b, E) for a, b, y in pairs if y == 1]
    neg = [flat_cosine(a, b, E) for a, b, y in pairs if y == 0]
    return auc(pos, neg)


def auc_ping(pairs, pings):
    cache: Dict = {}
    pos = [soft_sim(a, b, pings, cache) for a, b, y in pairs if y == 1]
    neg = [soft_sim(a, b, pings, cache) for a, b, y in pairs if y == 0]
    return auc(pos, neg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default="/Users/chetanpatil/Desktop/test/data-bank/snli_1.0_train.jsonl")
    ap.add_argument("--max-pairs", type=int, default=4000)
    ap.add_argument("--max-vocab", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.5)
    args = ap.parse_args()

    if os.path.exists(args.nli_path):
        pairs_raw = snli_pairs(args.nli_path, args.max_pairs)
        source = f"SNLI {args.nli_path}"
    else:
        pairs_raw = synthetic_hard()
        source = "SYNTHETIC-HARD (co-occurrence blind to label; real run needs --nli-path)"

    # train/test split BEFORE building anything
    cut = int(0.7 * len(pairs_raw))
    train_raw, test_raw = pairs_raw[:cut], pairs_raw[cut:]

    # vocab + co-occurrence from TRAIN sentences only (no test leakage)
    train_sents = [s for a, b, _ in train_raw for s in (a, b)]
    cnt = Counter(t for s in train_sents for t in s if t)
    keep = [w for w, _ in cnt.most_common(args.max_vocab)]
    vocab = {w: i for i, w in enumerate(keep)}
    V = len(vocab)
    depth = max(2, math.ceil(math.log(max(V, 2)) / math.log(26)))

    def to_ids(p):
        return [([vocab[w] for w in a if w in vocab],
                 [vocab[w] for w in b if w in vocab], y) for a, b, y in p]

    train, test = to_ids(train_raw), to_ids(test_raw)

    E_unsup = distributional_vectors(train_sents, vocab, args.dim)
    E_sup = train_embeddings(train, vocab, E_unsup, args.epochs, args.lr)

    codes_sup = learned_codes(E_sup, depth)
    codes_arb = arbitrary_codes(V, depth)
    pings_sup = build_token_pings(codes_sup)
    pings_arb = build_token_pings(codes_arb)

    print(f"source : {source}")
    print(f"vocab  : {V}   depth {depth}   train/test pairs : {len(train)}/{len(test)}")
    print("                                   train    test")
    print(f"flat cosine  UNSUPERVISED (PPMI) :  {auc_flat(train, E_unsup):.3f}   {auc_flat(test, E_unsup):.3f}")
    print(f"flat cosine  SUPERVISED          :  {auc_flat(train, E_sup):.3f}   {auc_flat(test, E_sup):.3f}")
    print(f"ping  codes  SUPERVISED          :  {auc_ping(train, pings_sup):.3f}   {auc_ping(test, pings_sup):.3f}")
    print(f"ping  codes  ARBITRARY (control) :  {auc_ping(train, pings_arb):.3f}   {auc_ping(test, pings_arb):.3f}")
    lift = auc_flat(test, E_sup) - auc_flat(test, E_unsup)
    print(f"supervision lift on TEST (flat)  :  {lift:+.3f}  -> "
          + ("labels shaped real, generalizing structure" if lift > 0.02
             else "no held-out gain — supervision only memorized or task needs cross-interaction"))


if __name__ == "__main__":
    main()
