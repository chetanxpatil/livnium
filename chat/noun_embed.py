"""
noun_embed.py — occurrence-based NOUN embeddings from raw text (PPMI + SVD).

No neural training. The classical count pipeline, done honestly:

    1. COUNT   slide a ±window over the corpus; rows = nouns, columns = ALL
               context words (a noun's meaning comes from its verbs and
               adjectives too, not just other nouns).
    2. PPMI    reweight counts by surprise:  max(0, log p(n,c)/(p(n)·p(c)^α)).
               Kills "the"-domination; α=0.75 is word2vec's negative smoothing.
    3. SVD     truncate to --dim, scale by sqrt(Σ)  ->  the embeddings.
    4. PROBE   nearest neighbors for a few nouns, so the run ends with evidence.

Levy & Goldberg (2014): skip-gram with negative sampling implicitly factorizes
this same PMI matrix — so this IS the occurrence-based way, minus the epochs.

Noun detection: WordNet's noun lexicon as an O(1) lookup set (no tagging pass).
Ambiguous words ("run", "play") are included when they exist as nouns — a
lookup can't disambiguate; a POS tagger could, at ~100x the runtime.

Usage:
    pip3 install numpy scipy nltk
    python3 -c "import nltk; nltk.download('wordnet')"

    python3 noun_embed.py --data ~/Desktop/test/wiki.txt
    python3 noun_embed.py --data ~/Desktop/test/       # every .txt inside
    python3 noun_embed.py --probe cat physics india    # after training

Output: model/noun_embed.npz  { vectors, nouns, counts }
"""

import argparse
import glob
import os
import sys
from collections import Counter

import numpy as np

from prep_chat_context import clean  # the one tokenizer everything shares

OUT = "model/noun_embed.npz"


def noun_set():
    """WordNet's noun lexicon as a lowercase lookup set."""
    try:
        from nltk.corpus import wordnet as wn
        nouns = {l.name().lower().replace("_", " ")
                 for s in wn.all_synsets(pos="n") for l in s.lemmas()}
        return {n for n in nouns if " " not in n and n.isalpha()}
    except LookupError:
        sys.exit("wordnet missing:  python3 -c \"import nltk; nltk.download('wordnet')\"")
    except ImportError:
        sys.exit("needs nltk:  pip3 install nltk")


def iter_lines(path, max_lines=0):
    files = sorted(glob.glob(os.path.join(path, "**/*.txt"), recursive=True)) \
        if os.path.isdir(path) else [path]
    if not files:
        sys.exit(f"no .txt files under {path}")
    n = 0
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
                n += 1
                if max_lines and n >= max_lines:
                    return


def count_cooc(args, nouns):
    """One streaming pass: frequencies, then noun-row / all-word-column counts."""
    print("pass 1/2: word frequencies ...", flush=True)
    freq = Counter()
    for line in iter_lines(args.data, args.max_lines):
        freq.update(clean(line).split())
    keep_n = [w for w, c in freq.most_common()
              if w in nouns and c >= args.min_count][:args.max_nouns]
    keep_c = [w for w, c in freq.most_common(args.ctx_vocab) if c >= args.min_count]
    n_id = {w: i for i, w in enumerate(keep_n)}
    c_id = {w: i for i, w in enumerate(keep_c)}
    print(f"  nouns kept {len(keep_n):,}   context vocab {len(keep_c):,}"
          f"   (min-count {args.min_count})", flush=True)

    print("pass 2/2: co-occurrence counts ...", flush=True)
    pair = Counter()          # (noun_row, ctx_col) -> weighted count
    W = args.window
    for k, line in enumerate(iter_lines(args.data, args.max_lines)):
        toks = clean(line).split()
        ids = [(n_id.get(t), c_id.get(t)) for t in toks]
        for i, (ni, _) in enumerate(ids):
            if ni is None:
                continue
            for j in range(max(0, i - W), min(len(ids), i + W + 1)):
                if j == i:
                    continue
                cj = ids[j][1]
                if cj is not None:
                    pair[(ni, cj)] += 1.0 / abs(i - j)   # closer words count more
        if k % 200000 == 0 and k:
            print(f"  {k:,} lines   {len(pair):,} pairs", flush=True)
    return keep_n, keep_c, pair, freq


def ppmi_svd(keep_n, keep_c, pair, dim, alpha):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    rows, cols, vals = zip(*((r, c, v) for (r, c), v in pair.items()))
    M = csr_matrix((vals, (rows, cols)), shape=(len(keep_n), len(keep_c)))
    total = M.sum()
    pn = np.asarray(M.sum(1)).ravel() / total            # p(noun)
    pc = np.asarray(M.sum(0)).ravel() / total            # p(ctx)
    pc = pc ** alpha; pc /= pc.sum()                     # context smoothing
    M = M.tocoo()
    pmi = np.log((M.data / total) / (pn[M.row] * pc[M.col]))
    keep = pmi > 0                                       # the "positive" in PPMI
    M = csr_matrix((pmi[keep], (M.row[keep], M.col[keep])),
                   shape=(len(keep_n), len(keep_c)))
    print(f"PPMI matrix: {M.shape[0]:,} x {M.shape[1]:,}   nnz {M.nnz:,}", flush=True)
    U, S, _ = svds(M, k=dim)
    order = np.argsort(-S)
    X = U[:, order] * np.sqrt(S[order])                  # sqrt(Σ) weighting
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def probe(words, k=8):
    z = np.load(OUT, allow_pickle=True)
    X, nouns = z["vectors"], list(z["nouns"])
    idx = {w: i for i, w in enumerate(nouns)}
    for w in words:
        if w not in idx:
            print(f"  {w:14s} (not in noun vocab)")
            continue
        sims = X @ X[idx[w]]
        sims[idx[w]] = -1e9
        top = np.argsort(-sims)[:k]
        print(f"  {w:14s} -> " + "  ".join(f"{nouns[i]}({sims[i]:.2f})" for i in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="txt file, or a folder of .txt files")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dim", type=int, default=256, help="matches the well dim")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.75, help="context smoothing")
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--max-nouns", type=int, default=50000)
    ap.add_argument("--ctx-vocab", type=int, default=100000)
    ap.add_argument("--max-lines", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--probe", nargs="*", default=None,
                    help="skip training; show neighbors from a saved model")
    args = ap.parse_args()

    if args.probe is not None:
        probe(args.probe)
        return
    if not args.data:
        sys.exit("need --data (or --probe to inspect a saved model)")

    nouns = noun_set()
    print(f"WordNet noun lexicon: {len(nouns):,} single words", flush=True)
    keep_n, keep_c, pair, freq = count_cooc(args, nouns)
    X = ppmi_svd(keep_n, keep_c, pair, args.dim, args.alpha)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, vectors=X.astype(np.float32),
                        nouns=np.array(keep_n),
                        counts=np.array([freq[w] for w in keep_n]))
    print(f"saved -> {args.out}   ({len(keep_n):,} nouns x {args.dim}d)\n")
    print("--- sanity probe ---")
    demo = [w for w in ("man", "water", "war", "music", "city",
                        "physics", "dog") if w in set(keep_n)][:5]
    probe(demo)


if __name__ == "__main__":
    main()
