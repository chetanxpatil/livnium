"""build_vocab.py — stage 2: ONE shared vocabulary + noun-target set.

Identical rules to noun_collapse_pure.build_vocab (top-N by frequency,
min-count gates, WordNet noun-eligible targets), computed once from the
frozen corpus and used verbatim by all three model families.

Usage:
    python3 build_vocab.py            # defaults mirror the published run
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "research", "embeddings", "noun-collapse"))
from noun_embed import noun_set                       # noqa: E402

from common import WORK, corpus_lines, load_json, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--min-noun-count", type=int, default=50)
    ap.add_argument("--max-nouns", type=int, default=50000)
    args = ap.parse_args()

    corpus = os.path.join(WORK, "corpus.txt")
    man = load_json(os.path.join(WORK, "corpus_manifest.json"))

    print("counting word frequencies ...", flush=True)
    freq = Counter()
    for toks in corpus_lines(corpus):
        freq.update(toks)

    keep = [w for w, c in freq.most_common(args.vocab) if c >= args.min_count]
    stoi = {w: i + 1 for i, w in enumerate(keep)}                 # 0 = PAD
    nouns = noun_set()
    noun_ids = sorted(stoi[w] for w in keep
                      if w in nouns and freq[w] >= args.min_noun_count)
    noun_ids = noun_ids[: args.max_nouns]

    body = json.dumps({"stoi": stoi, "noun_ids": noun_ids}, sort_keys=True)
    vocab_sha = hashlib.sha256(body.encode()).hexdigest()
    save_json(os.path.join(WORK, "vocab.json"), {
        "stoi": stoi, "noun_ids": noun_ids,
        "params": vars(args), "corpus_sha256": man["sha256"],
        "vocab_sha256": vocab_sha,
    })
    print(f"vocab {len(stoi):,}  nouns {len(noun_ids):,}\nvocab sha256: {vocab_sha}")


if __name__ == "__main__":
    main()
