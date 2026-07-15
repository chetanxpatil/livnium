"""
intrinsic_dim.py — how many dimensions does the noun manifold ACTUALLY use?

The wells are stored in D=256 coordinates, but they don't fill that space —
they sit on a lower, curved manifold. This measures the *effective* /
*intrinsic* dimension of that manifold three honest ways. All three can be
fractional; that's the point.

  1. PARTICIPATION RATIO   PR = (Σλ)² / Σλ²  over PCA eigenvalues.
                           A single "effective number of axes" — how many
                           directions the variance really spreads across.
  2. PCA 90/95/99%         how many principal components hold that much variance
                           (integer counts, the coarse view).
  3. TwoNN (Facco 2017)    a geometry-native intrinsic-dimension estimator:
                           from the ratio of each point's 2nd- to 1st-nearest-
                           neighbor distance. This one is genuinely fractional
                           and is the closest thing to a "fractal dimension" of
                           the manifold. Subsampled for the O(N²) neighbor step.

Measured on the UNIT-normalized wells (cosine geometry is how the model uses
them). Nouns only by default — that's the trained, meaningful sub-table.

Usage:
    python3 intrinsic_dim.py
    python3 intrinsic_dim.py --ckpt model/noun_collapse_pure.pt --sample 4000
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F


def participation_ratio(evals):
    s1 = evals.sum()
    s2 = (evals ** 2).sum()
    return float(s1 * s1 / s2)


def pca_eigs(X):
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    evals = np.linalg.eigvalsh(cov)[::-1]          # descending
    return np.clip(evals, 0, None)


def pca_ncomp(evals, frac):
    c = np.cumsum(evals) / evals.sum()
    return int(np.searchsorted(c, frac) + 1)


def twonn(X, seed=0):
    """Facco et al. 2017 two-nearest-neighbor intrinsic dimension.
    For each point: mu = dist(2nd NN)/dist(1st NN). Then d is the slope of
    -log(1 - F(mu)) vs log(mu), fit through the origin on the linear bulk."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    # pairwise on unit vectors via cosine -> euclidean; chunk to bound memory
    d1 = np.full(N, np.inf); d2 = np.full(N, np.inf)
    for i in range(0, N, 512):
        blk = X[i:i + 512]
        dist = np.sqrt(np.clip(2 - 2 * (blk @ X.T), 0, None))  # unit-sphere L2
        for r in range(blk.shape[0]):
            dist[r, i + r] = np.inf                 # exclude self
        part = np.partition(dist, 1, axis=1)[:, :2]
        d1[i:i + blk.shape[0]] = part[:, 0]
        d2[i:i + blk.shape[0]] = part[:, 1]
    mu = d2 / np.clip(d1, 1e-12, None)
    mu = mu[np.isfinite(mu) & (mu > 1)]
    mu.sort()
    F_emp = (np.arange(1, len(mu) + 1)) / (len(mu) + 1)
    x = np.log(mu)
    y = -np.log(1 - F_emp)
    keep = int(len(mu) * 0.9)                       # drop the noisy tail
    d = float(np.sum(x[:keep] * y[:keep]) / np.sum(x[:keep] * x[:keep]))
    return d


def main():
    ap = argparse.ArgumentParser()
    default_ckpt = os.path.join(os.path.dirname(__file__), "model", "noun_collapse_pure.pt")
    ap.add_argument("--ckpt", default=default_ckpt)
    ap.add_argument("--sample", type=int, default=4000,
                    help="subsample size for the TwoNN neighbor step")
    ap.add_argument("--all-words", action="store_true",
                    help="use the full vocab, not just noun targets")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    A = F.normalize(ck["wells"], dim=-1)
    if not args.all_words:
        A = A[torch.tensor(ck["noun_ids"])]
    X = A.numpy().astype(np.float64)
    D = X.shape[1]
    print(f"wells: {X.shape[0]:,} vectors x {D} stored dimensions "
          f"({'nouns' if not args.all_words else 'all words'})\n")

    evals = pca_eigs(X)
    pr = participation_ratio(evals)
    print("--- linear (PCA) ---")
    print(f"  participation ratio (effective axes): {pr:.1f}  of {D}")
    for frac in (0.90, 0.95, 0.99):
        print(f"  components for {int(frac*100)}% variance: "
              f"{pca_ncomp(evals, frac)}")

    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], min(args.sample, X.shape[0]), replace=False)
    d_twonn = twonn(X[idx])
    print("\n--- manifold (TwoNN, can be fractional) ---")
    print(f"  intrinsic dimension: {d_twonn:.2f}  "
          f"(on {len(idx):,} sampled nouns)")

    # local (TwoNN) vs global (PR) is the honest story: their GAP is curvature.
    # a flat k-d subspace has local == global; a curved k-d manifold folded
    # through the box has local << global (Swiss-roll: 2-d sheet, needs 3 axes).
    print(f"\nverdict: the noun manifold is intrinsically ~{d_twonn:.0f}-d "
          f"(local, TwoNN) but nonlinearly embedded — curved through ~{pr:.0f} "
          f"of the {D} stored axes (global, participation ratio).")
    print(f"         the gap ({d_twonn:.0f} vs {pr:.0f}) IS the curvature the "
          f"collapse dynamics carved; a linear method (PPMI+SVD) would have "
          f"local == global.")


if __name__ == "__main__":
    main()
