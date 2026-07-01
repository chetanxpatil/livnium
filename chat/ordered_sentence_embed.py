"""
ordered_sentence_embed.py — a sentence is ONE ordered descent; position is meaning.

The token experiments quietly became bag-of-words: soft_sim matched each word to
its best partner, ignoring order. That throws away "where the word sits". This
file takes the sentence as the unit (bounded by a full stop), splits words on
spaces, and walks them LEFT TO RIGHT as a single descent:

    "a dog chased the cat"  ->  word_0 -> word_1 -> word_2 -> word_3 -> word_4

Each word has a fixed-width base-26 code, so word k always lands at the same
depth band of the descent. Two sentences are then compared by POSITION-ALIGNED
meaning_match (depth i of A vs depth i of B) — so "dog chased cat" and
"cat chased dog" are no longer the same string of doorways.

The honest subtlety
-------------------
We keep identity frames. Cube rotations preserve angles, so a position-frame
R^k would NOT change the cosine between same-position doorways — the positional
signal here is the index-aligned comparison itself ("word k of A vs word k of B"),
which is exactly "where the word sits". Frames matter for the *path's shape*
(world_path), not for this same-depth similarity; claiming otherwise would be
hand-waving.

The test
--------
Synthetic where ORDER is the only signal: contradiction = the same sentence with
subject and object swapped (identical word multiset). The bag sees identical
words and cannot separate the classes (it can even invert); the ordered descent
should. --nli-path runs the same scorers on real SNLI.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from livnium_core.ping import CELLS_26, Ping, Step, meaning_match  # noqa: E402
from token_path_embed import auc  # noqa: E402

RNG = np.random.default_rng(0)
MAX_DEPTH = 48


# --------------------------------------------------------------------------- #
# words -> fixed-width codes -> one ordered sentence descent
# --------------------------------------------------------------------------- #
def word_codes(vocab: Dict[str, int], width: int) -> Dict[str, List[int]]:
    """Each word -> a fixed-width base-26 code (consistent across all sentences,
    so position k always occupies the same depth band)."""
    codes = {}
    for w, idx in vocab.items():
        d, r = [0] * width, idx
        for i in range(width - 1, -1, -1):
            d[i] = r % 26
            r //= 26
        codes[w] = d
    return codes


def sentence_to_ping(words: List[str], codes: Dict[str, List[int]]) -> Ping:
    steps: List[Step] = []
    for w in words:
        c = codes.get(w)
        if c is None:
            continue
        for dig in c:
            steps.append(Step(cell=CELLS_26[dig]))   # identity frame
        if len(steps) >= MAX_DEPTH:
            break
    if not steps:
        steps = [Step(cell=CELLS_26[0])]
    return Ping(steps=tuple(steps))


def ordered_sim(a: List[str], b: List[str], codes) -> float:
    return meaning_match(sentence_to_ping(a, codes), sentence_to_ping(b, codes)).score


def bag_cosine(a: List[str], b: List[str], vocab: Dict[str, int]) -> float:
    va, vb = np.zeros(len(vocab)), np.zeros(len(vocab))
    for w in a:
        if w in vocab:
            va[vocab[w]] += 1
    for w in b:
        if w in vocab:
            vb[vocab[w]] += 1
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return 0.0 if na == 0 or nb == 0 else float(va @ vb / (na * nb))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def synthetic_order() -> List[Tuple[List[str], List[str], int]]:
    nouns = "dog cat man woman child bird fish horse mouse lion".split()
    verbs = {"chased": "pursued", "bit": "nipped", "saw": "spotted",
             "fed": "nourished", "watched": "observed", "followed": "trailed"}
    vlist = list(verbs)
    pairs: List[Tuple[List[str], List[str], int]] = []
    for _ in range(600):
        a, b = RNG.choice(nouns, 2, replace=False)
        v = str(RNG.choice(vlist))
        s1 = [a, v, b]
        if RNG.random() < 0.5:
            s2 = [a, verbs[v], b]          # same roles/order, synonym verb -> entailment
            pairs.append((s1, s2, 1))
        else:
            s2 = [b, v, a]                  # SAME words, subject/object swapped -> contradiction
            pairs.append((s1, s2, 0))
    return pairs


def load_snli_ordered(path: str, max_pairs: int):
    import json
    pairs, sents = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_pairs and len(pairs) >= max_pairs:
                break
            d = json.loads(line.strip() or "{}")
            g = d.get("gold_label", "")
            s1 = [w.strip(".,!?;:\"'").lower() for w in (d.get("sentence1") or "").split()]
            s2 = [w.strip(".,!?;:\"'").lower() for w in (d.get("sentence2") or "").split()]
            sents += [s1, s2]
            if g in ("entailment", "contradiction"):
                pairs.append((s1, s2, 1 if g == "entailment" else 0))
    return pairs, sents


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default="/Users/chetanpatil/Desktop/test/data-bank/snli_1.0_train.jsonl")
    ap.add_argument("--max-pairs", type=int, default=4000)
    args = ap.parse_args()

    if os.path.exists(args.nli_path):
        pairs, sents = load_snli_ordered(args.nli_path, args.max_pairs)
        source = f"SNLI {args.nli_path}"
    else:
        pairs = synthetic_order()
        sents = [s for a, b, _ in pairs for s in (a, b)]
        source = "SYNTHETIC-ORDER (contradiction = subject/object swap; real run needs --nli-path)"

    vocab = {}
    for s in sents:
        for w in s:
            if w and w not in vocab:
                vocab[w] = len(vocab)
    V = len(vocab)
    width = max(1, math.ceil(math.log(max(V, 2)) / math.log(26)))
    codes = word_codes(vocab, width)

    ob_pos, ob_neg, bg_pos, bg_neg = [], [], [], []
    for a, b, y in pairs:
        o, bg = ordered_sim(a, b, codes), bag_cosine(a, b, vocab)
        if y == 1:
            ob_pos.append(o); bg_pos.append(bg)
        else:
            ob_neg.append(o); bg_neg.append(bg)

    print(f"source : {source}")
    print(f"vocab  : {V}   word-code width {width}   pairs : {len(ob_pos)} ent / {len(ob_neg)} contra")
    print(f"AUC  bag cosine  (order-free)     : {auc(bg_pos, bg_neg):.3f}")
    print(f"AUC  ordered descent (position)   : {auc(ob_pos, ob_neg):.3f}")
    d = auc(ob_pos, ob_neg) - auc(bg_pos, bg_neg)
    print(f"order - bag                       : {d:+.3f}  -> "
          + ("position carries signal the bag throws away" if d > 0.02
             else "no gain from order on this data"))


if __name__ == "__main__":
    main()
