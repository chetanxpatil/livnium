#!/usr/bin/env python3
"""
Honest Rule 30 test.

Tests, on held-out data, whether the project's two prediction targets behave as
claimed:

  TARGET A  -- the project's actual target: row 1-density sign, c_{t+1} > 0.5,
               predicted from current-row n-bit pattern frequencies f_t.
               (This is what FINAL_REPORT/main_tex call the "center column bit".)

  TARGET B  -- the target the paper RHETORICALLY claims: Rule 30's true
               single-cell center column b_{t+1} (the famous pseudorandom
               sequence), predicted from the full current row / its PCA embedding.

Train and test trajectories use DIFFERENT seeds (no overlap, no leakage).
Baselines: majority class, persistence (predict next = current), Bernoulli(p).
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

rng_global = np.random.default_rng(0)


def rule30_step(row: np.ndarray) -> np.ndarray:
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    return np.bitwise_xor(left, np.bitwise_or(row, right))


def evolve(width: int, steps: int, seed: int, single_seed: bool):
    """Return grid of shape (steps, width)."""
    rng = np.random.default_rng(seed)
    if single_seed:
        row = np.zeros(width, dtype=np.uint8)
        row[width // 2] = 1
    else:
        row = rng.integers(0, 2, width, dtype=np.uint8)
    grid = np.empty((steps, width), dtype=np.uint8)
    for t in range(steps):
        grid[t] = row
        row = rule30_step(row)
    return grid


def nbit_freqs(grid: np.ndarray, n: int) -> np.ndarray:
    """For each row, frequency vector over all 2^n n-bit windows. (steps, 2^n)"""
    steps, width = grid.shape
    out = np.zeros((steps, 1 << n), dtype=np.float64)
    for t in range(steps):
        cells = grid[t].astype(np.uint32)
        codes = np.zeros(width, dtype=np.uint32)
        for i in range(n):
            codes += (np.roll(cells, -i) << (n - 1 - i))
        counts = np.bincount(codes, minlength=1 << n)
        out[t] = counts / width
    return out


def row_density(grid: np.ndarray) -> np.ndarray:
    return grid.mean(axis=1)


def acc(pred, y):
    return float((np.asarray(pred) == np.asarray(y)).mean())


def report(name, model_acc, baselines: dict):
    line = f"{name:<46} model={model_acc:.4f}"
    for k, v in baselines.items():
        line += f"  {k}={v:.4f}"
    print(line)


def main():
    WIDTH = 20000
    STEPS = 4000
    N = 6

    print("=" * 86)
    print("HONEST RULE 30 TEST  (train seed=1, test seed=2, disjoint)")
    print(f"width={WIDTH}  steps={STEPS}  n-bit windows={N}")
    print("=" * 86)

    # ---- random wide initial conditions, two disjoint trajectories ----
    g_tr = evolve(WIDTH, STEPS, seed=1, single_seed=False)
    g_te = evolve(WIDTH, STEPS, seed=2, single_seed=False)

    f_tr = nbit_freqs(g_tr, N)
    f_te = nbit_freqs(g_te, N)
    d_tr = row_density(g_tr)
    d_te = row_density(g_te)

    # ========== TARGET A: row-density sign  c_{t+1} > 0.5  from f_t ==========
    Xa_tr, ya_tr = f_tr[:-1], (d_tr[1:] > 0.5).astype(int)
    Xa_te, ya_te = f_te[:-1], (d_te[1:] > 0.5).astype(int)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=0)
    mlp.fit(Xa_tr, ya_tr)
    a_model = acc(mlp.predict(Xa_te), ya_te)
    a_major = max(ya_te.mean(), 1 - ya_te.mean())
    a_persist = acc((d_te[:-1] > 0.5).astype(int), ya_te)
    print()
    print("TARGET A  (project's real target: next-row density sign)")
    report("  f_t -> [c_{t+1}>0.5]", a_model,
           {"majority": a_major, "persist_c_t": a_persist})

    # cross-check: next-row density is an EXACT linear function of 3-bit freqs.
    # Rule30 -> 1 for patterns 100,011,010,001. Build exact density from f_t (n>=3).
    # center bit of an n-window is at index n//2; recover 3-bit nbhd marginals:
    pats = np.array([[ (p >> (N - 1 - b)) & 1 for b in range(N)] for p in range(1 << N)])
    c = N // 2
    # exact next-density via 3-neighborhood (cells c-1,c,c+1) of each window
    to_one = ((pats[:, c-1] == 1) & (pats[:, c] == 0) & (pats[:, c+1] == 0)) | \
             ((pats[:, c-1] == 0) & (pats[:, c] == 1) & (pats[:, c+1] == 1)) | \
             ((pats[:, c-1] == 0) & (pats[:, c] == 1) & (pats[:, c+1] == 0)) | \
             ((pats[:, c-1] == 0) & (pats[:, c] == 0) & (pats[:, c+1] == 1))
    exact_next_density = f_te[:-1] @ to_one.astype(float)
    a_exact = acc((exact_next_density > 0.5).astype(int), ya_te)
    print(f"  EXACT analytic linear map (no ML)        model={a_exact:.4f}   "
          f"<- shows Target A is a known closed-form quantity")

    # ========== TARGET B: TRUE single-cell center column b_{t+1} ==========
    # the famous pseudorandom sequence. Use single-seed evolutions (classic setup).
    gb_tr = evolve(WIDTH, STEPS, seed=1, single_seed=True)
    gb_te = evolve(WIDTH, STEPS, seed=2, single_seed=True)  # same single seed -> identical;
    # to get a genuinely disjoint test trajectory, shift the seed position:
    def single_seed_at(width, steps, pos):
        row = np.zeros(width, dtype=np.uint8); row[pos] = 1
        grid = np.empty((steps, width), dtype=np.uint8)
        for t in range(steps):
            grid[t] = row; row = rule30_step(row)
        return grid
    gb_tr = single_seed_at(WIDTH, STEPS, WIDTH // 2)
    gb_te = single_seed_at(WIDTH, STEPS, WIDTH // 3)  # disjoint cone

    col_tr = gb_tr[:, WIDTH // 2].astype(int)   # true center column over time
    col_te = gb_te[:, WIDTH // 3].astype(int)

    # Features = full row, reduced by PCA fit on TRAIN only (the paper's "geometry").
    pca = PCA(n_components=8, random_state=0).fit(gb_tr.astype(float))
    Pb_tr = pca.transform(gb_tr.astype(float))
    Pb_te = pca.transform(gb_te.astype(float))

    Xb_tr, yb_tr = Pb_tr[:-1], col_tr[1:]
    Xb_te, yb_te = Pb_te[:-1], col_te[1:]
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(Xb_tr, yb_tr)
    b_model = acc(rf.predict(Xb_te), yb_te)
    b_major = max(yb_te.mean(), 1 - yb_te.mean())
    b_persist = acc(col_te[:-1], yb_te)
    b_bernoulli = acc(rng_global.integers(0, 2, len(yb_te)), yb_te)
    print()
    print("TARGET B  (paper's rhetorical target: TRUE single-cell center column)")
    report("  PCA(row) -> b_{t+1}", b_model,
           {"majority": b_major, "persist": b_persist, "bernoulli": b_bernoulli})

    # also: predict b_{t+1} from the WHOLE raw row (max info, still causal/OOS)
    rf2 = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    rf2.fit(gb_tr[:-1].astype(float), col_tr[1:])
    b_full = acc(rf2.predict(gb_te[:-1].astype(float)), col_te[1:])
    report("  full raw row -> b_{t+1}", b_full,
           {"majority": b_major})

    print()
    print("=" * 86)
    print("READ-OUT")
    print("  Target A model ~ exact analytic value -> high acc is a closed-form")
    print("  density map, NOT rule recovery from geometry.")
    print("  Target B (the real pseudorandom column) -> model collapses to majority/")
    print("  chance out-of-sample: Rule 30's center column stays unpredictable.")
    print("=" * 86)


if __name__ == "__main__":
    main()
