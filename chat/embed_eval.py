"""
embed_eval.py — ONE yardstick for both noun-embedding models.

SimLex-999 (Hill et al. 2015): 999 human-rated word pairs, rated for
SIMILARITY (not association — coffee/cup scores LOW). We take the noun
pairs, keep those where both words are in the model's noun vocab, and
report Spearman correlation between model cosine and human rating.

Same script, both models — the comparison is only honest if the ruler is:
    python3 embed_eval.py --model model/noun_embed.npz
    python3 embed_eval.py --model model/noun_collapse_pure.pt

Reference points (full Wikipedia, all-word models): PPMI+SVD ~0.38,
word2vec ~0.37-0.44. Partial-dump noun-only runs will land lower; what
matters here is the GAP between the two models on identical data.
"""

import argparse
import os
import urllib.request
import zipfile

import numpy as np

URL = "https://fh295.github.io/SimLex-999.zip"
CACHE = "data/simlex"


def simlex_noun_pairs():
    os.makedirs(CACHE, exist_ok=True)
    txt = os.path.join(CACHE, "SimLex-999", "SimLex-999.txt")
    if not os.path.exists(txt):
        z = os.path.join(CACHE, "simlex.zip")
        print(f"downloading SimLex-999 ...", flush=True)
        urllib.request.urlretrieve(URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(CACHE)
    pairs = []
    with open(txt, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        iw1, iw2, ipos, isim = (head.index(c) for c in ("word1", "word2", "POS", "SimLex999"))
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[ipos] == "N":
                pairs.append((p[iw1].lower(), p[iw2].lower(), float(p[isim])))
    return pairs


def load_vectors(path):
    """Returns (unit vectors, word -> row index) for the model's NOUNS."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        X = z["vectors"]
        idx = {w: i for i, w in enumerate(z["nouns"])}
    else:
        import torch
        ck = torch.load(path, map_location="cpu")
        W = ck["wells"]
        X = (W / W.norm(dim=-1, keepdim=True).clamp(min=1e-8)).numpy()
        itos = {i: w for w, i in ck["stoi"].items()}
        idx = {itos[i]: i for i in ck["noun_ids"]}
    return X, idx


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".npz (PPMI+SVD) or .pt (collapse)")
    args = ap.parse_args()

    X, idx = load_vectors(args.model)
    pairs = simlex_noun_pairs()
    sims, gold, skipped = [], [], 0
    for w1, w2, s in pairs:
        if w1 in idx and w2 in idx:
            sims.append(float(X[idx[w1]] @ X[idx[w2]]))
            gold.append(s)
        else:
            skipped += 1
    if len(gold) < 50:
        print(f"only {len(gold)} pairs covered — too few to trust. train on more data.")
        return
    rho = spearman(np.array(sims), np.array(gold))
    print(f"model     : {args.model}")
    print(f"coverage  : {len(gold)}/{len(pairs)} SimLex noun pairs ({skipped} skipped)")
    print(f"spearman  : {rho:.4f}")


if __name__ == "__main__":
    main()
