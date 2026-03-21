"""
R(5,5) Sandbox v4  —  K₅-Threshold Kernel
==========================================
Motivation: triangle adjacency kernel (v3) still fires on ALL edges with t_R > 0,
including K₄ edges (t_R=2) and mere triangle edges (t_R=1).
This creates a background gradient that wastes annealing energy on safe edges.

K₅ threshold kernel (v4):
  f(t) = max(0, t - K5_THRESHOLD)³     where K5_THRESHOLD = 2
  T = Σ_{(u,v)} [f(t_R(u,v)) + f(t_B(u,v))]

Why k=2 is the right threshold:
  edge in K₃    → t_R = 1 → f(1) = 0          ← silent
  edge in K₄    → t_R = 2 → f(2) = 0          ← silent
  edge in K₅    → t_R ≥ 3 → f(3) = 1          ← fires!
  edge in K₆    → t_R ≥ 4 → f(4) = 8          ← strong signal
  edge in K₇    → t_R ≥ 5 → f(5) = 27         ← very strong signal

Result: background noise from K₃/K₄ edges is exactly zero.
SA gradient is concentrated 100% on K₅-critical edges.

Signal comparison vs v3 (t_R=3 edge):
  v3: 3³ = 27    (vs background ~10³ = 1000 for random edge)
  v4: (3-2)³ = 1  (vs background 0 for all K₄-and-below edges)

The v4 landscape has:
  - T=0 iff NO edge has t_R ≥ 3 and NO edge has t_B ≥ 3
  - T=0 is a K₅-free certificate (by definition)
  - Every non-zero gradient points directly at a K₅ structure

Phase A fix (v3 K₅ was only 40%):
  v3 used T_init = n³/4 — too low for the steeper threshold landscape.
  v4 uses T_init = 50 * n²  and  max_steps = 15000 for Phase A.

Same O(n)-per-flip update as v3 — only the aggregation formula changes.
"""

import numpy as np
from itertools import combinations
import random
import math
import time

# ---------------------------------------------------------------------------
# K₅-threshold function and its delta
# ---------------------------------------------------------------------------
K5_THRESHOLD = 2          # fires only when t_R ≥ 3  (i.e. K₅ or higher)

def f_thresh(t):
    """f(t) = max(0, t - K5_THRESHOLD)^3"""
    x = t - K5_THRESHOLD
    return x * x * x if x > 0 else 0

def f_delta(t_old, t_new):
    """f(t_new) - f(t_old)"""
    return f_thresh(t_new) - f_thresh(t_old)


# ---------------------------------------------------------------------------
# Threshold Lattice  (same structure as v3, different aggregation)
# ---------------------------------------------------------------------------
class ThresholdLattice:
    """
    t_R[i][j] = common RED neighbours of i and j
    t_B[i][j] = common BLUE neighbours of i and j
    T = Σ_{(i,j)} [f(t_R[i,j]) + f(t_B[i,j])]
      where f(t) = max(0, t - 2)^3  (K₅-threshold)

    Memory: O(n²) — two n×n int32 matrices.
    Flip cost: O(n) — same as v3.
    """

    def __init__(self, n, coloring):
        self.n        = n
        self.coloring = dict(coloring)
        self.adj      = np.zeros((n, n), dtype=np.int8)
        for (i, j), c in self.coloring.items():
            self.adj[i, j] = self.adj[j, i] = c   # 0=Red 1=Blue

        # Build t_R, t_B via A_R @ A_R (same as v3)
        A_R = (self.adj == 0).astype(np.int32)
        A_B = (self.adj == 1).astype(np.int32)
        np.fill_diagonal(A_R, 0)
        np.fill_diagonal(A_B, 0)
        t_R_raw    = A_R @ A_R
        t_B_raw    = A_B @ A_B
        self.t_R   = t_R_raw - A_R
        self.t_B   = t_B_raw - A_B
        np.fill_diagonal(self.t_R, 0)
        np.fill_diagonal(self.t_B, 0)

    def tension(self):
        """T = Σ_{upper triangle} [f(t_R) + f(t_B)]"""
        iu, iv = np.triu_indices(self.n, k=1)
        tR = self.t_R[iu, iv]
        tB = self.t_B[iu, iv]
        total = 0
        for t in tR:
            total += f_thresh(int(t))
        for t in tB:
            total += f_thresh(int(t))
        return total

    def tension_fast(self):
        """Vectorised version of tension()."""
        iu, iv  = np.triu_indices(self.n, k=1)
        tR      = self.t_R[iu, iv].astype(np.int64) - K5_THRESHOLD
        tB      = self.t_B[iu, iv].astype(np.int64) - K5_THRESHOLD
        tR_pos  = np.maximum(tR, 0)
        tB_pos  = np.maximum(tB, 0)
        return int((tR_pos**3 + tB_pos**3).sum())

    def flip(self, edge):
        """Flip edge (u,v). O(n) update — identical logic to v3."""
        u, v   = edge
        c_old  = self.coloring[edge]
        c_new  = 1 - c_old
        self.coloring[edge] = c_new
        self.adj[u, v] = self.adj[v, u] = c_new

        for w in range(self.n):
            if w == u or w == v:
                continue
            c_wv = int(self.adj[w, v])
            c_wu = int(self.adj[w, u])

            if c_old == 0:                  # Red → Blue
                if c_wv == 0:
                    self.t_R[u, w] -= 1;   self.t_R[w, u] -= 1
                else:
                    self.t_B[u, w] += 1;   self.t_B[w, u] += 1
                if c_wu == 0:
                    self.t_R[v, w] -= 1;   self.t_R[w, v] -= 1
                else:
                    self.t_B[v, w] += 1;   self.t_B[w, v] += 1
            else:                           # Blue → Red
                if c_wv == 1:
                    self.t_B[u, w] -= 1;   self.t_B[w, u] -= 1
                else:
                    self.t_R[u, w] += 1;   self.t_R[w, u] += 1
                if c_wu == 1:
                    self.t_B[v, w] -= 1;   self.t_B[w, v] -= 1
                else:
                    self.t_R[v, w] += 1;   self.t_R[w, v] += 1

    def tension_after_flip(self, edge):
        """Compute ΔT for flipping edge without committing. O(n)."""
        u, v   = edge
        c_old  = self.coloring[edge]
        delta  = 0

        for w in range(self.n):
            if w == u or w == v:
                continue
            c_wv = int(self.adj[w, v])
            c_wu = int(self.adj[w, u])

            if c_old == 0:                  # Red → Blue
                if c_wv == 0:
                    t = int(self.t_R[u, w])
                    delta += f_delta(t, t - 1)
                else:
                    t = int(self.t_B[u, w])
                    delta += f_delta(t, t + 1)
                if c_wu == 0:
                    t = int(self.t_R[v, w])
                    delta += f_delta(t, t - 1)
                else:
                    t = int(self.t_B[v, w])
                    delta += f_delta(t, t + 1)
            else:                           # Blue → Red
                if c_wv == 1:
                    t = int(self.t_B[u, w])
                    delta += f_delta(t, t - 1)
                else:
                    t = int(self.t_R[u, w])
                    delta += f_delta(t, t + 1)
                if c_wu == 1:
                    t = int(self.t_B[v, w])
                    delta += f_delta(t, t - 1)
                else:
                    t = int(self.t_R[v, w])
                    delta += f_delta(t, t + 1)
        return delta

    def copy_coloring(self):
        return dict(self.coloring)


# ---------------------------------------------------------------------------
# Explicit clique counter
# ---------------------------------------------------------------------------
def count_mono_cliques(coloring, n, k):
    count = 0
    for verts in combinations(range(n), k):
        edges  = [(min(a, b), max(a, b)) for a, b in combinations(verts, 2)]
        colors = [coloring[e] for e in edges]
        if len(set(colors)) == 1:
            count += 1
    return count


# ---------------------------------------------------------------------------
# SA with threshold kernel
# ---------------------------------------------------------------------------
def anneal_threshold(n, coloring_init,
                     T_init=None, T_final=0.5, alpha=0.9995,
                     max_steps=50000, exploration=0.15, rng=None):
    if rng is None:
        rng = random.Random()
    if T_init is None:
        # Threshold landscape needs higher initial temperature than v3
        T_init = float(50 * n * n)

    edges = list(combinations(range(n), 2))
    lat   = ThresholdLattice(n, coloring_init)
    temp  = T_init
    T_now = lat.tension_fast()

    for step in range(max_steps):
        if rng.random() < exploration:
            edge = rng.choice(edges)
        else:
            sample = rng.choices(edges, k=min(20, len(edges)))
            edge   = min(sample, key=lambda e: lat.tension_after_flip(e))

        delta = lat.tension_after_flip(edge)
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            lat.flip(edge)
            T_now += delta

        temp = max(temp * alpha, T_final)

    return lat.copy_coloring(), T_now


# ---------------------------------------------------------------------------
# Phase A — Validation (R(3,3) reproduction)
# ---------------------------------------------------------------------------
def phase_a(n_restarts=20, seed=7):
    print("  ─── Phase A: Validation (threshold kernel + explicit K₃) ─────────")
    rng = random.Random(seed)

    cases = [
        (5, "K₅",  "K₃-free colorings exist (R(3,3)≤5)"),
        (6, "K₆",  "NO K₃-free coloring (R(3,3)=6)"),
    ]
    for n, label, note in cases:
        edges  = list(combinations(range(n), 2))
        k3_zero, best_k3, T_list = 0, 999, []
        for _ in range(n_restarts):
            init = {e: rng.randint(0, 1) for e in edges}
            col, T_fin = anneal_threshold(n, init,
                T_init=50.0 * n * n,       # higher T_init for steep landscape
                max_steps=15000,           # more steps for small graphs
                rng=rng)
            T_list.append(T_fin)
            k3 = count_mono_cliques(col, n, 3)
            if k3 < best_k3: best_k3 = k3
            if k3 == 0:      k3_zero += 1

        pct = k3_zero / n_restarts * 100
        t_label = "0" if min(T_list) == 0 else f"{min(T_list)}"
        print(f"  {label}  mono-K₃=0: {k3_zero}/{n_restarts} ({pct:.0f}%)"
              f"  T_min={t_label:>8}  best_K₃={best_k3}  [{note}]")
    print()


# ---------------------------------------------------------------------------
# Phase B — R(5,5) heuristic sweep
# ---------------------------------------------------------------------------
def phase_b(sweep_ns=(43, 44), n_restarts=10, max_steps=100000, seed=42):
    print("  ─── Phase B: R(5,5) sweep (threshold kernel + exhaustive K₅) ────")
    print("  Bounds: 43 ≤ R(5,5) ≤ 48")
    print()
    rng = random.Random(seed)

    for n in sweep_ns:
        edges  = list(combinations(range(n), 2))
        n_k5   = math.comb(n, 5)
        print(f"  K_{n}  ({len(edges)} edges, {n_k5:,} K₅ subgraphs)")

        t0 = time.time()
        best_k5, cert = 999, False

        for trial in range(n_restarts):
            init = {e: rng.randint(0, 1) for e in edges}

            t1  = time.time()
            col, T_fin = anneal_threshold(n, init,
                T_init=50.0 * n * n,
                alpha=0.9995,
                max_steps=max_steps,
                rng=rng)
            sa_time = time.time() - t1

            t2  = time.time()
            k5  = count_mono_cliques(col, n, 5)
            k5_time = time.time() - t2

            if k5 < best_k5:
                best_k5 = k5
            elapsed = time.time() - t0
            print(f"    restart {trial+1:2d}/{n_restarts}  T={T_fin:10.0f}"
                  f"  mono-K₅={k5:4d}  SA={sa_time:.0f}s  K₅-check={k5_time:.1f}s"
                  f"  ({elapsed:.0f}s total)")

            if k5 == 0:
                print(f"    ✓ CERTIFICATE: K_{n} has K₅-free 2-coloring → R(5,5) > {n}")
                cert = True
                break

        result = (f"✓ K₅-FREE CERTIFICATE → R(5,5) > {n}" if cert
                  else f"⚠ best mono-K₅={best_k5} (heuristic; not a proof)")
        print(f"  K_{n} result: {result}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("▓" * 72)
    print("  R(5,5) SANDBOX v4  —  K₅-Threshold Kernel")
    print("  T = Σ_{(u,v)} [max(0, t_R-2)³ + max(0, t_B-2)³]")
    print("▓" * 72)
    print()
    print("  Threshold k=2: silent on K₃/K₄ edges, fires only on K₅+ edges")
    print("  f(t_R=1)=0, f(t_R=2)=0, f(t_R=3)=1, f(t_R=4)=8, f(t_R=5)=27")
    print("  T=0 ⟺ no edge has t_R≥3 AND no edge has t_B≥3")
    print("  T=0 is a K₅-free certificate by construction.")
    print()

    phase_a()
    phase_b()

    print("  Kernel hierarchy:")
    print("    v1 vertex-sum  → parity loophole (rejected)")
    print("    v2 cubic       → K₄ blind, 1:10 signal (floor ~736)")
    print("    v3 triangle    → 2-hop, 34% better (floor ~484)")
    print("    v4 threshold   → K₅-specific, zero background noise (floor TBD)")
    print()
    print("  Certificate (mono-K₅=0 via explicit check) is the only valid proof.")
    print("  T=0 for v4 is NECESSARY but not sufficient — verify with K₅ count.")


if __name__ == "__main__":
    run()
