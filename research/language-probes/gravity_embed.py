"""
gravity_embed.py — the mover: structure-energy where movement becomes meaning.

The previous step (token_path_embed.py) CUT the structure with k-means. This one
GROWS it. Tokens are placed in a field and obey one rule:

    correlated tokens attract, uncorrelated tokens repel,
    and the whole field is held on a conserved shell so it cannot collapse.

The flow is a gradient descent on an energy

    E(P) = - sum_ij (A_ij - b) <p_i, p_j>          (A = PPMI affinity, b = its mean)

run on the unit sphere |p_i| = 1 (the conserved shell; the Om/core is the
singularity the points fall toward but never reach, so they spread instead of
imploding). Attraction pulls correlated tokens into the same basin; the repulsive
(below-mean) part pushes the rest away; the shell stops the black-hole collapse.
At equilibrium the tokens have settled into WELLS — and the wells, taken
recursively, ARE the descent code:

    densest well -> first doorway, sub-well -> next doorway, ... (base-26 path)

So the code is not imposed; it falls out of where movement settles. car and
vehicle drift into one basin (shared early path); banana into another.

This mirrors the repo's collapse engine: a monotone energy descent to a stable
attractor (we print the energy trace to show it descends, Lyapunov-style).

Honest scope
------------
This is a force-directed / spectral embedding wearing the cube's shell. The claim
to test is the same measurable one: does the gravity-GROWN path beat an ARBITRARY
path (and roughly match k-means), with flat cosine as the signal ceiling, on
AUC(entailment vs contradiction). Self-contained synthetic proof; --nli-path for
real SNLI.
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
sys.path.insert(0, _HERE)                                   # for token_path_embed
sys.path.insert(0, os.path.join(_HERE, "..", "..", "packages", "livnium-core", "src"))                  # for livnium_core

from token_path_embed import (  # noqa: E402  (reuse the harness)
    _kmeans,
    arbitrary_codes,
    auc,
    build_token_pings,
    distributional_vectors,
    flat_cosine,
    learned_codes,
    load_snli,
    soft_sim,
    synthetic,
)
from paths import SNLI_TRAIN

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# affinity + the gravity flow
# --------------------------------------------------------------------------- #
def ppmi_affinity(sents: List[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    V = len(vocab)
    co = np.zeros((V, V))
    for toks in sents:
        ids = [vocab[t] for t in toks if t in vocab]
        for a in ids:
            for b in ids:
                if a != b:
                    co[a, b] += 1.0
    total = co.sum()
    if total == 0:
        return co
    row, col = co.sum(1, keepdims=True), co.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co * total) / (row @ col + 1e-12) + 1e-12)
    A = np.maximum(pmi, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def gravity_flow(A: np.ndarray, dim: int = 8, iters: int = 200, lr: float = 0.05):
    """Gradient flow of E = -sum (A_ij-b)<p_i,p_j> on the unit shell.
    Returns settled positions and the per-step energy trace (should descend)."""
    n = len(A)
    if n <= 1:
        return np.zeros((n, dim)), [0.0]
    b = A[A > 0].mean() if (A > 0).any() else 0.0
    G = A - b
    np.fill_diagonal(G, 0.0)
    P = RNG.standard_normal((n, dim))
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    deg = G.sum(1, keepdims=True)
    trace = []
    for _ in range(iters):
        # delta_i = sum_j G_ij (p_j - p_i) = G@P - deg*P   (descent direction of E)
        delta = G @ P - deg * P
        P = P + lr * delta
        P -= P.mean(0)                                   # kill global drift
        P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12   # conserved shell
        trace.append(float(-(G * (P @ P.T)).sum() / 2.0))
    return P, trace


# --------------------------------------------------------------------------- #
# recursive wells -> emergent descent code
# --------------------------------------------------------------------------- #
def gravity_codes(A: np.ndarray, depth: int, dim: int, iters: int, lr: float):
    """Recursively: let the field settle, split into wells (<=26), assign a
    doorway digit, recurse inside each well. The code grows from the motion."""
    codes: List[List[int]] = [[] for _ in range(len(A))]
    energy_drop = {"start": None, "end": None}

    def well_split(idx: np.ndarray, level: int) -> None:
        if level >= depth or len(idx) <= 1:
            for i in idx:
                codes[i] = codes[i] + [0] * (depth - len(codes[i]))
            return
        sub = A[np.ix_(idx, idx)]
        P, trace = gravity_flow(sub, dim=dim, iters=iters, lr=lr)
        if energy_drop["start"] is None and len(trace) > 1:
            energy_drop["start"], energy_drop["end"] = trace[0], trace[-1]
        k = min(26, max(2, int(round(math.sqrt(len(idx))))))
        labels = _kmeans(P, k)                            # wells the flow carved out
        for c in sorted(set(labels)):
            members = idx[labels == c]
            for i in members:
                codes[i].append(c)
            well_split(members, level + 1)

    well_split(np.arange(len(A)), 0)
    return codes, energy_drop


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default=SNLI_TRAIN)
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--max-vocab", type=int, default=1200)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    if os.path.exists(args.nli_path):
        sents, pairs = load_snli(args.nli_path, args.max_pairs)
        source = f"SNLI {args.nli_path}"
    else:
        sents, pairs = synthetic()
        source = "SYNTHETIC clustered corpus (mechanism proof; real run needs --nli-path)"

    cnt = Counter(t for s in sents for t in s if t)
    keep = [w for w, _ in cnt.most_common(args.max_vocab)]
    vocab = {w: i for i, w in enumerate(keep)}
    V = len(vocab)
    depth = max(2, math.ceil(math.log(max(V, 2)) / math.log(26)))

    A = ppmi_affinity(sents, vocab)
    vecs = distributional_vectors(sents, vocab, args.dim)        # for flat + kmeans

    code_G, edrop = gravity_codes(A, depth, args.dim, args.iters, args.lr)
    code_K = learned_codes(vecs, depth)
    code_R = arbitrary_codes(V, depth)
    pings = {"G": build_token_pings(code_G), "K": build_token_pings(code_K),
             "R": build_token_pings(code_R)}
    caches = {"G": {}, "K": {}, "R": {}}

    pos = {k: [] for k in ("flat", "G", "K", "R")}
    neg = {k: [] for k in ("flat", "G", "K", "R")}
    for s1, s2, g in pairs:
        a = [vocab[w] for w in s1 if w in vocab]
        b = [vocab[w] for w in s2 if w in vocab]
        bucket = pos if g == "entailment" else neg
        bucket["flat"].append(flat_cosine(a, b, vecs))
        for k in ("G", "K", "R"):
            bucket[k].append(soft_sim(a, b, pings[k], caches[k]))

    print(f"source : {source}")
    print(f"vocab  : {V} tokens   depth {depth}  (capacity 26^{depth} = {26**depth:,})")
    print(f"pairs  : {len(pos['flat'])} entailment / {len(neg['flat'])} contradiction")
    if edrop["start"] is not None:
        print(f"energy : top-level flow {edrop['start']:.2f} -> {edrop['end']:.2f}  "
              f"({'descended (Lyapunov-style)' if edrop['end'] <= edrop['start'] else 'rose (!)'} )")
    print(f"AUC flat cosine (reference) : {auc(pos['flat'], neg['flat']):.3f}")
    print(f"AUC ping GRAVITY  (grown)   : {auc(pos['G'], neg['G']):.3f}")
    print(f"AUC ping KMEANS   (cut)     : {auc(pos['K'], neg['K']):.3f}")
    print(f"AUC ping ARBITRARY (id)     : {auc(pos['R'], neg['R']):.3f}")
    dG = auc(pos["G"], neg["G"]) - auc(pos["R"], neg["R"])
    print(f"gravity - arbitrary         : {dG:+.3f}  -> "
          + ("movement grew a meaningful path" if dG > 0.02
             else "no gain — the flow did not form usable structure"))


if __name__ == "__main__":
    main()
