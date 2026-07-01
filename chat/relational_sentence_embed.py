"""
relational_sentence_embed.py — a word means by its relation to EVERY other word.

Three views of the same sentence, in increasing structure:

  bag            : multiset of words            (no order at all)
  ordered-index  : word k of A vs word k of B   (absolute slot; brittle to inserts)
  relational     : every word vs every other word, DIRECTED (i before j)

The third is "where the word sits compared to every word in the sentence". A word
is characterised not by its slot but by its directed relations to all sentence-
mates: "dog -> cat" (dog before cat) is a different relation than "cat -> dog".
A sentence becomes its set of directed word-pairs; two sentences are similar to
the degree those directed relations overlap. This is robust to insertions (adding
"the" only adds pairs, it does not shift everyone) yet still order-aware — the
cross/skip relation that chat/SNLI_BASELINES.md names as the missing mechanism.

(Livnium reading: each word is an origin, every other word a viewed cell, the
directed pair is the oriented relation cos-theta(om -> lo). Here we score the
discrete overlap of those oriented relations; weighting each pair by the descent
cos-theta is the natural next refinement.)

Test: contradiction = the same words with subject/object swapped (identical
multiset). bag is blind (and inverts); ordered-index helps; relational should
separate cleanly because dog->cat and cat->dog are different relations.
--nli-path runs all three on real SNLI.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from ordered_sentence_embed import (  # noqa: E402
    bag_cosine,
    load_snli_ordered,
    ordered_sim,
    synthetic_order,
    word_codes,
)
from token_path_embed import auc  # noqa: E402


# --------------------------------------------------------------------------- #
# directed all-pairs relations
# --------------------------------------------------------------------------- #
def directed_pairs(words: List[str]) -> Counter:
    """Every ordered pair (earlier word -> later word) in the sentence."""
    c: Counter = Counter()
    n = len(words)
    for i in range(n):
        for j in range(i + 1, n):
            if words[i] and words[j]:
                c[(words[i], words[j])] += 1
    return c


def relational_sim(a: List[str], b: List[str]) -> float:
    ca, cb = directed_pairs(a), directed_pairs(b)
    if not ca or not cb:
        return 0.0
    dot = sum(v * cb[k] for k, v in ca.items() if k in cb)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


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

    vocab: Dict[str, int] = {}
    for s in sents:
        for w in s:
            if w and w not in vocab:
                vocab[w] = len(vocab)
    V = len(vocab)
    width = max(1, math.ceil(math.log(max(V, 2)) / math.log(26)))
    codes = word_codes(vocab, width)

    rows = {"bag": ([], []), "ord": ([], []), "rel": ([], [])}
    for a, b, y in pairs:
        scores = {
            "bag": bag_cosine(a, b, vocab),
            "ord": ordered_sim(a, b, codes),
            "rel": relational_sim(a, b),
        }
        for k, (pos, neg) in rows.items():
            (pos if y == 1 else neg).append(scores[k])

    print(f"source : {source}")
    print(f"vocab  : {V}   pairs : {len(rows['bag'][0])} ent / {len(rows['bag'][1])} contra")
    print(f"AUC  bag            (no order)        : {auc(*rows['bag']):.3f}")
    print(f"AUC  ordered-index  (absolute slot)   : {auc(*rows['ord']):.3f}")
    print(f"AUC  relational     (every word pair) : {auc(*rows['rel']):.3f}")


if __name__ == "__main__":
    main()
