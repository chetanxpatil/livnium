"""
supervised_gravity.py — the mover, moved by meaning.

gravity_embed.py flowed tokens on CO-OCCURRENCE affinity. This adds a second
force from the LABELS:

    entailment pair  -> its tokens ATTRACT  (+)
    contradiction    -> its tokens REPEL    (-)

The field now obeys a signed affinity  W = A_cooc + lambda * S_label  and flows
on the conserved unit shell exactly as before (gradient descent on
E = -sum W_ij <p_i,p_j>, energy descends, wells form). The wells -> base-26
descent codes. So the same physics, but the basins are carved by the task, not
just by what co-occurs.

Honesty guards (same as supervised_embed.py)
--------------------------------------------
  - TRAIN/TEST split on pairs; BOTH affinities (co-occurrence and supervised) are
    built from TRAIN only; every AUC is on HELD-OUT test pairs (train also shown).
  - "Hard" corpus: entail/contra share topic+words, only a negation marker in the
    contradiction differs. Co-occurrence is blind to it, so unsupervised gravity
    should sit near chance and the label force has to earn the lift.
  - --nli-path runs the same pipeline on real SNLI.

Scope: still bag-of-tokens via a token field (no cross-sentence interaction), so
a floor on what label-gravity buys, not a ceiling.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "packages", "livnium-core", "src"))

from gravity_embed import ppmi_affinity  # noqa: E402
from supervised_embed import snli_pairs, synthetic_hard  # noqa: E402
from token_path_embed import (  # noqa: E402
    _kmeans,
    arbitrary_codes,
    auc,
    build_token_pings,
    distributional_vectors,
    flat_cosine,
    soft_sim,
)

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# signed gravity flow (W may contain negatives = repulsion)
# --------------------------------------------------------------------------- #
def signed_flow(W: np.ndarray, dim: int, iters: int, lr: float):
    n = len(W)
    if n <= 1:
        return np.zeros((n, dim)), [0.0]
    G = W.copy()
    np.fill_diagonal(G, 0.0)
    P = RNG.standard_normal((n, dim))
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    deg = G.sum(1, keepdims=True)
    trace = []
    for _ in range(iters):
        delta = G @ P - deg * P
        P = P + lr * delta
        P -= P.mean(0)
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        trace.append(float(-(G * (P @ P.T)).sum() / 2.0))
    return P, trace


def gravity_codes_signed(W: np.ndarray, depth: int, dim: int, iters: int, lr: float):
    codes: List[List[int]] = [[] for _ in range(len(W))]
    edrop = {"start": None, "end": None}

    def split(idx: np.ndarray, level: int) -> None:
        if level >= depth or len(idx) <= 1:
            for i in idx:
                codes[i] = codes[i] + [0] * (depth - len(codes[i]))
            return
        sub = W[np.ix_(idx, idx)]
        P, trace = signed_flow(sub, dim, iters, lr)
        if edrop["start"] is None and len(trace) > 1:
            edrop["start"], edrop["end"] = trace[0], trace[-1]
        k = min(26, max(2, int(round(math.sqrt(len(idx))))))
        labels = _kmeans(P, k)
        for c in sorted(set(labels)):
            members = idx[labels == c]
            for i in members:
                codes[i].append(c)
            split(members, level + 1)

    split(np.arange(len(W)), 0)
    return codes, edrop


# --------------------------------------------------------------------------- #
# supervised (signed) affinity from labeled pairs
# --------------------------------------------------------------------------- #
def label_affinity(train, V: int) -> np.ndarray:
    S = np.zeros((V, V))
    for a, b, y in train:
        sign = 1.0 if y == 1 else -1.0
        for i in a:
            for j in b:
                if i != j:
                    S[i, j] += sign
                    S[j, i] += sign
    return S


def norm01(M: np.ndarray) -> np.ndarray:
    m = np.abs(M).max()
    return M / m if m > 0 else M


def auc_ping(pairs, pings):
    cache: Dict = {}
    pos = [soft_sim(a, b, pings, cache) for a, b, y in pairs if y == 1]
    neg = [soft_sim(a, b, pings, cache) for a, b, y in pairs if y == 0]
    return auc(pos, neg)


def auc_flat(pairs, E):
    pos = [flat_cosine(a, b, E) for a, b, y in pairs if y == 1]
    neg = [flat_cosine(a, b, E) for a, b, y in pairs if y == 0]
    return auc(pos, neg)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default="/Users/chetanpatil/Desktop/test/data-bank/snli_1.0_train.jsonl")
    ap.add_argument("--max-pairs", type=int, default=4000)
    ap.add_argument("--max-vocab", type=int, default=1200)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=4.0, help="weight of the label force")
    args = ap.parse_args()

    if os.path.exists(args.nli_path):
        pairs_raw = snli_pairs(args.nli_path, args.max_pairs)
        source = f"SNLI {args.nli_path}"
    else:
        pairs_raw = synthetic_hard()
        source = "SYNTHETIC-HARD (co-occurrence blind to label; real run needs --nli-path)"

    cut = int(0.7 * len(pairs_raw))
    train_raw, test_raw = pairs_raw[:cut], pairs_raw[cut:]
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

    A = norm01(ppmi_affinity(train_sents, vocab))      # co-occurrence (>=0)
    S = norm01(label_affinity(train, V))               # supervised (signed)
    W_unsup = A
    W_sup = A + args.lam * S
    E_flat = distributional_vectors(train_sents, vocab, args.dim)

    codes_U, _ = gravity_codes_signed(W_unsup, depth, args.dim, args.iters, args.lr)
    codes_G, edrop = gravity_codes_signed(W_sup, depth, args.dim, args.iters, args.lr)
    codes_R = arbitrary_codes(V, depth)
    pings_U = build_token_pings(codes_U)
    pings_G = build_token_pings(codes_G)
    pings_R = build_token_pings(codes_R)

    print(f"source : {source}")
    print(f"vocab  : {V}   depth {depth}   train/test : {len(train)}/{len(test)}   lambda {args.lam}")
    if edrop["start"] is not None:
        print(f"energy : label-gravity flow {edrop['start']:.2f} -> {edrop['end']:.2f}  "
              f"({'descended' if edrop['end'] <= edrop['start'] else 'rose (!)'})")
    print("                                      train    test")
    print(f"flat cosine UNSUPERVISED (PPMI)    :  {auc_flat(train, E_flat):.3f}   {auc_flat(test, E_flat):.3f}")
    print(f"ping gravity UNSUPERVISED (co-occ) :  {auc_ping(train, pings_U):.3f}   {auc_ping(test, pings_U):.3f}")
    print(f"ping gravity SUPERVISED  (+labels) :  {auc_ping(train, pings_G):.3f}   {auc_ping(test, pings_G):.3f}")
    print(f"ping gravity ARBITRARY  (control)  :  {auc_ping(train, pings_R):.3f}   {auc_ping(test, pings_R):.3f}")
    lift = auc_ping(test, pings_G) - auc_ping(test, pings_U)
    print(f"label-force lift on TEST           :  {lift:+.3f}  -> "
          + ("the mover learned the task into its wells" if lift > 0.02
             else "no held-out gain from the label force"))


if __name__ == "__main__":
    main()
