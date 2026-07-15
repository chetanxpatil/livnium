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

    # plain text
    python3 noun_embed.py --data ~/Desktop/test/wiki.txt
    # a raw Wikipedia dump, streamed straight from the .bz2 (no decompression
    # to disk). START WITH A SLICE — the full dump is 2 passes x hours:
    python3 noun_embed.py \
        --data ~/Desktop/test/data-bank/enwiki-latest-pages-articles-multistream.xml.bz2 \
        --max-lines 20000000
    python3 noun_embed.py --probe cat physics india    # after training

Memory: pair counts are pruned whenever the table passes --max-pairs
(count-1 pairs dropped first) — bounded RAM at a small bias against the
rarest pairs, which PPMI would mostly zero out anyway.

Output: model/noun_embed.npz  { vectors, nouns, counts }
"""

import argparse
import bz2
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

from text import clean

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "model", "noun_embed.npz")


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


# --- wiki markup stripping (rough on purpose: co-occurrence counting only
# needs readable prose, not a perfect render) -------------------------------
_RE_DROP = re.compile(
    r"\{\{[^{}]*\}\}"            # templates {{...}} (innermost; loop below)
    r"|\{\|.*?\|\}"              # tables
    r"|<ref[^>]*/>|<ref.*?</ref>"  # references
    r"|<[^>]+>"                  # any remaining tags
    r"|\[\[(?:File|Image|Category):[^\[\]]*\]\]", re.DOTALL)
_RE_LINK = re.compile(r"\[\[(?:[^|\[\]]*\|)?([^|\[\]]*)\]\]")   # [[a|b]] -> b
_RE_URL = re.compile(r"https?://\S+|'{2,}")


def strip_wiki(text):
    for _ in range(3):                       # nested templates, a few levels
        new = _RE_DROP.sub(" ", text)
        if new == text:
            break
        text = new
    text = _RE_LINK.sub(r"\1", text)
    return _RE_URL.sub(" ", text)


def _parse_pages(line_iter):
    """Article prose out of dump XML lines: skips non-article namespaces and
    redirects, strips markup, yields cleaned text lines."""
    in_text, ns_ok, buf = False, True, []
    if True:
        for raw in line_iter:
            s = raw.strip()
            if "<ns>" in s:
                ns_ok = "<ns>0</ns>" in s        # articles only
            if "<redirect" in s:
                ns_ok = False
            if not in_text and "<text" in s:
                if not ns_ok:
                    continue
                in_text = True
                s = s[s.index("<text"):]         # drop anything before the tag
                s = s.split(">", 1)[1] if ">" in s else ""
                if s.lstrip().lower().startswith("#redirect"):
                    in_text, buf = False, []
                    continue
            if in_text:
                if "</text>" in s:
                    buf.append(s.split("</text>")[0])
                    for line in strip_wiki("\n".join(buf)).split("\n"):
                        if len(line) > 40:       # skip headings/leftover markup
                            yield line
                    in_text, buf = False, []
                else:
                    buf.append(s)


def wiki_lines(path):
    """Stream the whole dump front-to-back (article-ID order)."""
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        def _safe(src):
            try:
                yield from src
            except (EOFError, OSError):        # truncated/partial dump: use
                print("  [dump ends early (partial file) — stopping cleanly]",
                      flush=True)              # what we got, stop cleanly
        yield from _parse_pages(_safe(f))


_BZ_MAGIC = b"BZh9\x31\x41\x59\x26\x53\x59"    # start of an independent stream


def wiki_lines_sampled(path, parts=2000, seed=0):
    """MULTISTREAM random access: the dump is thousands of independent bz2
    blocks (~100 articles each). Probe `parts` evenly spaced offsets in
    RANDOM order, decompress one block at each — uniform domain coverage
    instead of article-ID order. Reproducible for a given seed, so pass 1
    (vocab) and pass 2 (training) see the same sample."""
    import random as _rnd
    size = os.path.getsize(path)
    order = list(range(parts))
    _rnd.Random(seed).shuffle(order)
    with open(path, "rb") as f:
        for k in order:
            f.seek(k * size // parts)
            # scan forward for the next stream boundary (up to 4MB)
            buf, base = b"", f.tell()
            hit = -1
            for _ in range(4):
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                hit = buf.find(_BZ_MAGIC)
                if hit >= 0:
                    break
            if hit < 0:
                continue
            f.seek(base + hit)
            dec = bz2.BZ2Decompressor()
            out = b""
            try:
                while not dec.eof:
                    raw = f.read(1 << 20)
                    if not raw:
                        break
                    out += dec.decompress(raw)
            except OSError:
                continue                        # false-positive magic: skip probe
            yield from _parse_pages(iter(out.decode("utf-8", "ignore").splitlines()))


def iter_lines(path, max_lines=0, sample_parts=0, seed=0):
    if path.endswith(".bz2"):
        src = (wiki_lines_sampled(path, sample_parts, seed) if sample_parts
               else wiki_lines(path))
    else:
        files = sorted(glob.glob(os.path.join(path, "**/*.txt"), recursive=True)) \
            if os.path.isdir(path) else [path]
        if not files:
            sys.exit(f"no .txt files under {path}")

        def _txt():
            for fp in files:
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    yield from f
        src = _txt()
    for n, line in enumerate(src, 1):
        yield line
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
    pair = Counter()          # int key: noun_row * n_ctx + ctx_col (RAM-lean)
    n_ctx = len(keep_c)
    W = args.window
    pruned = 0
    for k, line in enumerate(iter_lines(args.data, args.max_lines)):
        toks = clean(line).split()
        ids = [(n_id.get(t), c_id.get(t)) for t in toks]
        for i, (ni, _) in enumerate(ids):
            if ni is None:
                continue
            base = ni * n_ctx
            for j in range(max(0, i - W), min(len(ids), i + W + 1)):
                if j == i:
                    continue
                cj = ids[j][1]
                if cj is not None:
                    pair[base + cj] += 1.0 / abs(i - j)  # closer words count more
        if len(pair) > args.max_pairs:                   # bounded memory:
            cut = 1.0 + pruned * 0.5                     # raise the bar each time
            pair = Counter({p: v for p, v in pair.items() if v > cut})
            pruned += 1
            print(f"  [pruned pairs <= {cut:.1f}  ->  {len(pair):,} kept]", flush=True)
        if k % 200000 == 0 and k:
            print(f"  {k:,} lines   {len(pair):,} pairs", flush=True)
    return keep_n, keep_c, pair, freq


def ppmi_svd(keep_n, keep_c, pair, dim, alpha):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds
    n_ctx = len(keep_c)
    rows, cols, vals = zip(*((p // n_ctx, p % n_ctx, v) for p, v in pair.items()))
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
    k = min(dim, min(M.shape) - 1)           # svds needs k < min(shape)
    if k < dim:
        print(f"  [corpus too small for dim {dim} -> using {k}]", flush=True)
    U, S, _ = svds(M, k=k)
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
    ap.add_argument("--max-pairs", type=int, default=30_000_000,
                    help="prune the pair table past this size (~3GB RAM); "
                         "count-1 pairs go first")
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
