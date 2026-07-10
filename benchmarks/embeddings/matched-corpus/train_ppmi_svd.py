"""train_ppmi_svd.py — PPMI + truncated SVD on the frozen corpus.

Matched to collapse: same frozen corpus restricted to the shared vocab,
±5 window, 256 dimensions. Counting is deterministic; the seed only affects
the randomized SVD. Embedding = U * S^0.5 (Levy & Goldberg 2014).

Memory note: co-occurrence over a 100k vocab is the heavy step. Counts are
accumulated in a dict of Counters and flushed to sparse COO blocks; if RAM is
tight, cap the CONTEXT vocab with --context-cap (rows stay full-vocab).

Output: work/models/ppmi_svd_seed{K}.npz  {words, vectors} + lineage.
"""

import argparse
import os
from collections import Counter, defaultdict

from common import WORK, caffeinate, corpus_lines, load_vocab, save_json, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--context-cap", type=int, default=0,
                    help="use only the N most frequent words as CONTEXTS (0 = full vocab)")
    ap.add_argument("--flush-every", type=int, default=2000000,
                    help="flush the count dict to sparse blocks every N lines")
    args = ap.parse_args()
    caffeinate()

    import numpy as np
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD

    voc = load_vocab()
    stoi = voc["stoi"]
    out_npz = os.path.join(WORK, "models", f"ppmi_svd_seed{args.seed}.npz")
    os.makedirs(os.path.join(WORK, "models"), exist_ok=True)
    if os.path.exists(out_npz):
        print(f"{out_npz} exists — done (delete to retrain)")
        return

    # ids are 1..V in vocab.json; reindex to 0..V-1 rows here
    idx = {w: i - 1 for w, i in stoi.items()}
    V = len(idx)
    ctx_ok = None
    if args.context_cap:
        by_id = sorted(idx.values())[: args.context_cap]   # vocab is frequency-ordered
        ctx_ok = np.zeros(V, dtype=bool)
        ctx_ok[by_id] = True

    W = args.window
    blocks, counts, lines = [], defaultdict(Counter), 0

    def flush():
        if not counts:
            return
        rows, cols, vals = [], [], []
        for r, c in counts.items():
            for cc, v in c.items():
                rows.append(r); cols.append(cc); vals.append(v)
        blocks.append(sp.coo_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)), shape=(V, V)).tocsr())
        counts.clear()

    print("counting co-occurrences ...", flush=True)
    corpus = os.path.join(WORK, "corpus.txt")
    for toks in corpus_lines(corpus):
        ids = [idx[t] for t in toks if t in idx]
        for i, t in enumerate(ids):
            lo = max(0, i - W)
            for j in range(lo, min(len(ids), i + W + 1)):
                if j == i:
                    continue
                c = ids[j]
                if ctx_ok is None or ctx_ok[c]:
                    counts[t][c] += 1.0
        lines += 1
        if lines % args.flush_every == 0:
            flush()
            print(f"  {lines:,} lines ({len(blocks)} blocks)", flush=True)
    flush()
    M = blocks[0]
    for b in blocks[1:]:
        M = M + b
    print(f"co-occurrence: {M.nnz:,} nonzeros", flush=True)

    # ---- PPMI: max(0, log( p(w,c) / (p(w) p(c)) ))
    total = M.sum()
    row = np.asarray(M.sum(axis=1)).ravel()
    col = np.asarray(M.sum(axis=0)).ravel()
    M = M.tocoo()
    pmi = np.log(np.maximum(M.data * total / (row[M.row] * col[M.col]), 1e-12))
    keep = pmi > 0
    P = sp.csr_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=M.shape)
    print(f"PPMI: {P.nnz:,} nonzeros", flush=True)

    svd = TruncatedSVD(n_components=args.dim, random_state=args.seed, algorithm="randomized")
    U = svd.fit_transform(P)                                  # = U * S
    S = svd.singular_values_
    vecs = (U / np.sqrt(np.maximum(S, 1e-12))).astype(np.float32)   # U * S^0.5

    itos = sorted(idx, key=idx.get)
    np.savez_compressed(out_npz, words=np.array(itos), vectors=vecs)
    save_json(out_npz.replace(".npz", ".meta.json"), {
        **stamp(), "model": "ppmi_svd", "seed": args.seed, "dim": args.dim,
        "window": args.window, "context_cap": args.context_cap,
        "embedding": "U*S^0.5 (Levy & Goldberg 2014)",
        "nnz_cooc": int(M.nnz), "nnz_ppmi": int(P.nnz),
    })
    print(f"saved -> {out_npz}")


if __name__ == "__main__":
    main()
