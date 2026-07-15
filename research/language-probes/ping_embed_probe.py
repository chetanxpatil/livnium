"""
ping_embed_probe.py — does a Livnium *descent* embedding carry signal that a flat
char-cosine embedding does not? A measured probe, no learning.

Idea
----
base27's alphabet is '0' + a..z: the Om/core plus exactly 26 letters — one per
doorway (livnium_core.ping.CELLS_26). So a word maps to a descent with no
arbitrary choices: letter i -> doorway CELLS_26[i], with a child OM-frame taken
deterministically from the 24-element rotation group. A sentence is the
concatenation of its words' descents (capped at MAX_DEPTH letters).

Two sentences are then compared two ways:
  - ping  : meaning_match(p, q).score  — mean cosine over the shared descent depth
            (the hierarchical, path-based similarity)
  - flat  : cosine of 26-dim letter-count vectors (plain bag-of-letters)

We score SNLI pairs and ask a single, falsifiable question: which similarity
better separates ENTAILMENT pairs (should be similar) from CONTRADICTION pairs?
Metric: AUC of (entailment=positive vs contradiction=negative). Neutral ignored.

Honest scope
------------
This is an UNLEARNED, lexical-level map. The doorways are letters, so both
scorers mostly see lexical overlap; the question is only whether the descent's
structure adds anything on top of flat overlap. A large win is not expected
without a learned input->doorway map (see ping.py "meaning enters only as data
choosing the doorways"). Run on real data with --nli-path; the inline SMOKE
sample only proves the harness computes both scores and the AUC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

# make livnium_core importable when run as a script from anywhere
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "packages", "livnium-core", "src"))

from livnium_core.ping import CELLS_26, Ping, Step, cosine, meaning_match  # noqa: E402
from livnium_core.rotations import rotation_group  # noqa: E402
from paths import SNLI_TRAIN

_GROUP = rotation_group()                       # 24 OM-frames
_LETTER_IDX = {chr(ord("a") + i): i for i in range(26)}
MAX_DEPTH = 24                                  # cap descent length per sentence


# --------------------------------------------------------------------------- #
# the hand-coded map: text -> descent
# --------------------------------------------------------------------------- #
def sentence_to_ping(sent: str) -> Ping:
    """Letter i -> doorway CELLS_26[i]; frame = group[i % 24]. Non-letters skipped."""
    steps: List[Step] = []
    for ch in sent.lower():
        i = _LETTER_IDX.get(ch)
        if i is None:
            continue
        steps.append(Step(cell=CELLS_26[i], frame=_GROUP[i % 24]))
        if len(steps) >= MAX_DEPTH:
            break
    if not steps:  # degenerate (no letters): a single neutral doorway
        steps = [Step(cell=CELLS_26[0])]
    return Ping(steps=tuple(steps))


def letter_counts(sent: str) -> List[int]:
    """26-dim bag-of-letters vector (the flat baseline's representation)."""
    v = [0] * 26
    for ch in sent.lower():
        i = _LETTER_IDX.get(ch)
        if i is not None:
            v[i] += 1
    return v


def flat_cosine(a: str, b: str) -> float:
    va, vb = letter_counts(a), letter_counts(b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(x * x for x in vb) ** 0.5
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def ping_sim(a: str, b: str) -> float:
    return meaning_match(sentence_to_ping(a), sentence_to_ping(b)).score


# --------------------------------------------------------------------------- #
# AUC: rank-based, entailment(+) vs contradiction(-)
# --------------------------------------------------------------------------- #
def auc(scores_pos: List[float], scores_neg: List[float]) -> float:
    if not scores_pos or not scores_neg:
        return float("nan")
    wins = 0.0
    for p in scores_pos:
        for n in scores_neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(scores_pos) * len(scores_neg))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_snli(path: str, max_pairs: int) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_pairs and len(out) >= max_pairs:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            g = d.get("gold_label", "")
            if g in ("entailment", "contradiction"):
                out.append((d.get("sentence1", ""), d.get("sentence2", ""), g))
    return out


# A tiny labeled SMOKE sample — NOT real SNLI, only to prove the harness runs.
SMOKE: List[Tuple[str, str, str]] = [
    ("a man is playing a guitar on stage", "a man is performing music", "entailment"),
    ("a dog runs across the green field", "an animal is moving outdoors", "entailment"),
    ("two children build a sandcastle", "kids are playing at the beach", "entailment"),
    ("a woman reads a book by the window", "someone is reading indoors", "entailment"),
    ("a chef chops vegetables in a kitchen", "a person prepares food", "entailment"),
    ("the cyclist rides up a steep hill", "a bike is being ridden uphill", "entailment"),
    ("a man is playing a guitar on stage", "the man is sleeping in his bed", "contradiction"),
    ("a dog runs across the green field", "the cat sits still on a couch", "contradiction"),
    ("two children build a sandcastle", "the adults are demolishing a wall", "contradiction"),
    ("a woman reads a book by the window", "a woman is swimming in the ocean", "contradiction"),
    ("a chef chops vegetables in a kitchen", "a chef is empty handed outside", "contradiction"),
    ("the cyclist rides up a steep hill", "the cyclist is asleep on the grass", "contradiction"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default=SNLI_TRAIN)
    ap.add_argument("--max-pairs", type=int, default=4000)
    args = ap.parse_args()

    if os.path.exists(args.nli_path):
        pairs = load_snli(args.nli_path, args.max_pairs)
        source = f"SNLI {args.nli_path} ({len(pairs)} ent/contra pairs)"
    else:
        pairs = SMOKE
        source = f"SMOKE sample ({len(pairs)} pairs) — real numbers need --nli-path"

    ping_pos, ping_neg, flat_pos, flat_neg = [], [], [], []
    for s1, s2, g in pairs:
        ps, fs = ping_sim(s1, s2), flat_cosine(s1, s2)
        if g == "entailment":
            ping_pos.append(ps); flat_pos.append(fs)
        else:
            ping_neg.append(ps); flat_neg.append(fs)

    auc_ping = auc(ping_pos, ping_neg)
    auc_flat = auc(flat_pos, flat_neg)

    print(f"source : {source}")
    print(f"pairs  : {len(ping_pos)} entailment  /  {len(ping_neg)} contradiction")
    print(f"  AUC  ping  (meaning_match) : {auc_ping:.3f}")
    print(f"  AUC  flat  (char cosine)   : {auc_flat:.3f}")
    delta = auc_ping - auc_flat
    verdict = ("ping ADDS signal over flat overlap" if delta > 0.02 else
               "flat overlap BEATS ping" if delta < -0.02 else
               "tie — descent adds nothing over flat overlap (expected, unlearned map)")
    print(f"  delta                      : {delta:+.3f}  -> {verdict}")
    print("note : 0.5 == no separation; both see lexical overlap, so a tie means "
          "the geometry alone (no learning) just re-encodes letter overlap.")


if __name__ == "__main__":
    main()
