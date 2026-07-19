"""
token_path_embed.py — token as an inward path, and the one thing that matters:
is the path LEARNED or ARBITRARY?

A token becomes a descent through the 26 cube doorways (livnium_core.ping):

    token -> code (base-26 over the 26 doorways) -> doorway_1 -> doorway_2 -> ...

The base27 alphabet has '0' = the Om/core, which is the origin, not a door — so a
clean code is base-26 over CELLS_26 (depth D addresses 26^D tokens: 26, 676,
17576, 456976 ...). Two tokens are similar to the degree their descents agree
early (shared prefix) — exactly "car and vehicle share an early path, banana does
not".

The experiment (your 4 steps)
-----------------------------
1. build a token vocab from the corpus
2. give each token a Ping path two ways:
     LEARNED   : distributional vectors (PPMI + SVD) -> recursive 26-way k-means
                 -> the code IS the token's position in a semantic tree, so similar
                 tokens share early digits by construction
     ARBITRARY : a random permutation -> base-26 code (an address, not an embedding)
3. sentence similarity = soft token match via meaning_match over token descents
4. compare AUC(entailment vs contradiction) for:
     flat cosine (the same distributional vectors)   -- how much signal exists
     ping LEARNED                                     -- does the path recover it
     ping ARBITRARY                                   -- control: an address alone

Honest scope
------------
The win to look for is LEARNED > ARBITRARY (an arranged path carries meaning, a
raw id does not), with flat cosine as the reference ceiling. The discrete path
buys interpretability + prefix search; it is a *lossy* code, so matching flat
cosine exactly is not expected. Self-contained synthetic mode proves the
mechanism; --nli-path runs it on real SNLI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from paths import SNLI_TRAIN

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "packages", "livnium-core", "src"))

from livnium_core.ping import CELLS_26, Ping, Step, meaning_match  # noqa: E402

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# codes -> pings
# --------------------------------------------------------------------------- #
def digits_to_ping(digits: List[int]) -> Ping:
    """A base-26 code (each digit 0..25) becomes a descent with identity frames,
    so similarity is carried purely by which doorways the path takes."""
    return Ping(steps=tuple(Step(cell=CELLS_26[d]) for d in digits))


def rank_to_digits(rank: int, depth: int) -> List[int]:
    out = [0] * depth
    for i in range(depth - 1, -1, -1):
        out[i] = rank % 26
        rank //= 26
    return out


# --------------------------------------------------------------------------- #
# distributional vectors: PPMI + SVD
# --------------------------------------------------------------------------- #
def distributional_vectors(sents: List[List[str]], vocab: Dict[str, int], dim: int) -> np.ndarray:
    V = len(vocab)
    co = np.zeros((V, V), dtype=np.float64)
    for toks in sents:
        ids = [vocab[t] for t in toks if t in vocab]
        for a in ids:
            for b in ids:
                if a != b:
                    co[a, b] += 1.0
    total = co.sum()
    if total == 0:
        return np.zeros((V, dim))
    row = co.sum(1, keepdims=True)
    col = co.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co * total) / (row @ col + 1e-12) + 1e-12)
    ppmi = np.maximum(pmi, 0.0)
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    k = min(dim, u.shape[1])
    return u[:, :k] * s[:k]


# --------------------------------------------------------------------------- #
# recursive 26-way k-means -> hierarchical code (the LEARNED path)
# --------------------------------------------------------------------------- #
def _kmeans(X: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
    n = len(X)
    k = max(1, min(k, n))
    centers = X[RNG.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(0)
    return labels


def learned_codes(vecs: np.ndarray, depth: int) -> List[List[int]]:
    """Recursive 26-way clustering: digit at each level = which cluster, so similar
    vectors share leading digits. Branch factor capped at 26 (the doorways)."""
    codes: List[List[int]] = [[] for _ in range(len(vecs))]

    def split(idx: np.ndarray, level: int) -> None:
        if level >= depth or len(idx) <= 1:
            for i in idx:
                codes[i] = codes[i] + [0] * (depth - len(codes[i]))
            return
        k = min(26, max(2, int(round(math.sqrt(len(idx))))))
        labels = _kmeans(vecs[idx], k)
        for c in sorted(set(labels)):
            sub = idx[labels == c]
            for i in sub:
                codes[i].append(c)
            split(sub, level + 1)

    split(np.arange(len(vecs)), 0)
    return codes


def arbitrary_codes(n: int, depth: int) -> List[List[int]]:
    perm = RNG.permutation(n)
    rank = np.empty(n, dtype=int)
    rank[perm] = np.arange(n)
    return [rank_to_digits(int(rank[i]), depth) for i in range(n)]


# --------------------------------------------------------------------------- #
# sentence similarity
# --------------------------------------------------------------------------- #
def build_token_pings(codes: List[List[int]]) -> List[Ping]:
    return [digits_to_ping(c) for c in codes]


def soft_sim(a_ids: List[int], b_ids: List[int], pings: List[Ping], cache: Dict) -> float:
    """Mean over a-tokens of the best meaning_match to any b-token (and back)."""
    if not a_ids or not b_ids:
        return 0.0

    def one_way(src, dst):
        tot = 0.0
        for i in src:
            best = 0.0
            for j in dst:
                key = (i, j) if i < j else (j, i)
                v = cache.get(key)
                if v is None:
                    v = meaning_match(pings[i], pings[j]).score
                    cache[key] = v
                if v > best:
                    best = v
            tot += best
        return tot / len(src)

    return 0.5 * (one_way(a_ids, b_ids) + one_way(b_ids, a_ids))


def flat_cosine(a_ids: List[int], b_ids: List[int], vecs: np.ndarray) -> float:
    if not a_ids or not b_ids:
        return 0.0
    va, vb = vecs[a_ids].mean(0), vecs[b_ids].mean(0)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return 0.0 if na == 0 or nb == 0 else float(va @ vb / (na * nb))


def auc(pos: List[float], neg: List[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def synthetic() -> Tuple[List[List[str]], List[Tuple[List[str], List[str], str]]]:
    topics = {
        "vehicle": "car truck bus van bike motorcycle drive road",
        "food": "banana apple bread cheese rice soup meal eat",
        "music": "guitar piano drum song melody band play tune",
        "weather": "rain sun cloud storm wind snow sky cold",
        "sport": "ball goal team match score run game field",
        "ocean": "wave fish boat shore tide coral swim deep",
    }
    topic_words = {t: w.split() for t, w in topics.items()}
    sents: List[List[str]] = []
    by_topic: Dict[str, List[List[str]]] = {t: [] for t in topics}
    for t, words in topic_words.items():
        for _ in range(60):
            s = list(RNG.choice(words, size=5, replace=True))
            sents.append(s)
            by_topic[t].append(s)
    pairs: List[Tuple[List[str], List[str], str]] = []
    tnames = list(topics)
    for t in tnames:
        for _ in range(40):
            a, b = RNG.choice(len(by_topic[t]), 2, replace=False)
            pairs.append((by_topic[t][a], by_topic[t][b], "entailment"))
        for _ in range(40):
            t2 = tnames[(tnames.index(t) + 1) % len(tnames)]
            a = RNG.integers(len(by_topic[t])); b = RNG.integers(len(by_topic[t2]))
            pairs.append((by_topic[t][a], by_topic[t2][b], "contradiction"))
    return sents, pairs


def load_snli(path: str, max_pairs: int):
    sents, pairs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_pairs and len(pairs) >= max_pairs:
                break
            d = json.loads(line.strip() or "{}")
            g = d.get("gold_label", "")
            s1 = (d.get("sentence1") or "").lower().split()
            s2 = (d.get("sentence2") or "").lower().split()
            s1 = [w.strip(".,!?;:\"'") for w in s1]
            s2 = [w.strip(".,!?;:\"'") for w in s2]
            sents.append(s1); sents.append(s2)
            if g in ("entailment", "contradiction"):
                pairs.append((s1, s2, g))
    return sents, pairs


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default=SNLI_TRAIN)
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--max-vocab", type=int, default=2000)
    ap.add_argument("--svd-dim", type=int, default=24)
    args = ap.parse_args()

    if os.path.exists(args.nli_path):
        sents, pairs = load_snli(args.nli_path, args.max_pairs)
        source = f"SNLI {args.nli_path}"
    else:
        sents, pairs = synthetic()
        source = "SYNTHETIC clustered corpus (mechanism proof; real run needs --nli-path)"

    # vocab: most frequent tokens
    from collections import Counter
    cnt = Counter(t for s in sents for t in s if t)
    keep = [w for w, _ in cnt.most_common(args.max_vocab)]
    vocab = {w: i for i, w in enumerate(keep)}
    V = len(vocab)
    depth = max(2, math.ceil(math.log(max(V, 2)) / math.log(26)))

    vecs = distributional_vectors(sents, vocab, args.svd_dim)
    code_L = learned_codes(vecs, depth)
    code_A = arbitrary_codes(V, depth)
    pings_L = build_token_pings(code_L)
    pings_A = build_token_pings(code_A)
    cache_L: Dict = {}
    cache_A: Dict = {}

    pos = {"flat": [], "L": [], "A": []}
    neg = {"flat": [], "L": [], "A": []}
    for s1, s2, g in pairs:
        a = [vocab[w] for w in s1 if w in vocab]
        b = [vocab[w] for w in s2 if w in vocab]
        bucket = pos if g == "entailment" else neg
        bucket["flat"].append(flat_cosine(a, b, vecs))
        bucket["L"].append(soft_sim(a, b, pings_L, cache_L))
        bucket["A"].append(soft_sim(a, b, pings_A, cache_A))

    print(f"source     : {source}")
    print(f"vocab      : {V} tokens   code depth : {depth}  (capacity 26^{depth} = {26**depth:,})")
    print(f"pairs      : {len(pos['flat'])} entailment / {len(neg['flat'])} contradiction")
    print(f"AUC flat cosine (reference)   : {auc(pos['flat'], neg['flat']):.3f}")
    print(f"AUC ping  LEARNED  path       : {auc(pos['L'], neg['L']):.3f}")
    print(f"AUC ping  ARBITRARY path      : {auc(pos['A'], neg['A']):.3f}")
    dL = auc(pos["L"], neg["L"]) - auc(pos["A"], neg["A"])
    print(f"learned - arbitrary           : {dL:+.3f}  -> "
          + ("a learned path carries meaning an id does not"
             if dL > 0.02 else "no gain — learning did not arrange the path"))


if __name__ == "__main__":
    main()
