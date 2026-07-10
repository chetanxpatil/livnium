"""train_sgns.py — word2vec skip-gram negative sampling on the frozen corpus.

Matched to collapse: same frozen corpus (tokens outside the shared vocab are
DROPPED, so the lexicon is identical), dim 256, window ±5, one pass by default
(--epochs to change; the token budget = corpus tokens x epochs is recorded).
Seed controls init + negative sampling. workers=1 keeps the run reproducible;
raise it if you accept thread nondeterminism (recorded in the meta).

Output: work/models/sgns_seed{K}.npz  {words, vectors} + lineage.
"""

import argparse
import os

from common import WORK, caffeinate, corpus_lines, load_json, load_vocab, save_json, stamp


class FilteredCorpus:
    """Iterable of token lists restricted to the shared vocabulary."""

    def __init__(self, path, keep):
        self.path, self.keep = path, keep

    def __iter__(self):
        for toks in corpus_lines(self.path):
            t = [w for w in toks if w in self.keep]
            if t:
                yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--negative", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1,
                    help="1 = reproducible; >1 = faster but thread-nondeterministic")
    args = ap.parse_args()
    caffeinate()

    from gensim.models import Word2Vec

    voc = load_vocab()
    keep = set(voc["stoi"])
    corpus = os.path.join(WORK, "corpus.txt")
    out_npz = os.path.join(WORK, "models", f"sgns_seed{args.seed}.npz")
    os.makedirs(os.path.join(WORK, "models"), exist_ok=True)
    if os.path.exists(out_npz):
        print(f"{out_npz} exists — done (delete to retrain)")
        return

    sents = FilteredCorpus(corpus, keep)
    print(f"SGNS: dim {args.dim}, window {args.window}, neg {args.negative}, "
          f"epochs {args.epochs}, seed {args.seed}, workers {args.workers}", flush=True)
    model = Word2Vec(sentences=sents, vector_size=args.dim, window=args.window,
                     sg=1, negative=args.negative, min_count=1, epochs=args.epochs,
                     seed=args.seed, workers=args.workers)

    import numpy as np
    words = list(model.wv.index_to_key)
    vecs = model.wv.vectors.astype(np.float32)
    np.savez_compressed(out_npz, words=np.array(words), vectors=vecs)

    man = load_json(os.path.join(WORK, "corpus_manifest.json"))
    save_json(out_npz.replace(".npz", ".meta.json"), {
        **stamp(), "model": "sgns", "seed": args.seed, "dim": args.dim,
        "window": args.window, "negative": args.negative, "epochs": args.epochs,
        "workers": args.workers, "reproducible": args.workers == 1,
        "token_budget": man["tokens"] * args.epochs,
    })
    print(f"saved -> {out_npz}  ({len(words):,} words)")


if __name__ == "__main__":
    main()
