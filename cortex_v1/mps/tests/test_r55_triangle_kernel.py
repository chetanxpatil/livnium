"""
R(5,5) Sandbox v3  —  Triangle Adjacency Kernel
================================================
Upgrades from per-vertex degree (v2) to per-edge neighborhood density.

v2 post-mortem — "per-vertex kernel is K5-blind":
  T = Σ_v [σ_R³ + σ_B³] is minimized by balanced colorings.
  For K₄₃ (degree 42), a K₅ edge adds σ_R ≥ 4 to 2 vertices out of 42 —
  signal-to-background ~1:10. SA finds balanced coloring with ~800 K₅s.
  The mixed term σ_R²σ_B does not help: it is mathematically equivalent to
  T_v = 2a³(1+λ) + 2ax²(3-λ), same minima, same landscape, same blindness.

v3 fix — Triangle Adjacency Kernel (2-hop, edge-level):
  t_R[u][v] = |{w ≠ u,v : e(u,w)=R  and  e(v,w)=R}|  (common red neighbours)
  t_B[u][v] = |{w ≠ u,v : e(u,w)=B  and  e(v,w)=B}|  (common blue neighbours)
  T = Σ_{(u,v)} [ t_R(u,v)³ + t_B(u,v)³ ]

  Why this is K₅-sensitive:
    An edge in a mono-red K₅ has t_R ≥ 3 (3 other K₅ vertices as common red
    neighbours). Cubic spike: 3³ = 27 vs expected ~10³ ≈ 1000 background for
    a typical K₄₃ edge. Signal fraction ≈ 27% above baseline per K₅ edge.
    A K₅ contains 10 such edges → structural pressure ≈ +2700 per K₅.

  Why v2 couldn't see this:
    Vertex degree sees ONE hop (which vertices touch v).
    Triangle adjacency sees TWO hops (which vertices share an edge-neighbourhood).
    K₅ is a 2-hop structure — you need both hops to feel it.

O(n)-per-flip update (not O(1) — worth it for sensitivity):
  Flipping edge (u,v) Red→Blue:
    For each w ≠ u,v:
      if e(w,v)=R: t_R[u][w] -= 1            (v no longer common red nbr of u,w)
      if e(w,v)=B: t_B[u][w] += 1            (v now common blue nbr of u,w)
      if e(w,u)=R: t_R[v][w] -= 1
      if e(w,u)=B: t_B[v][w] += 1
  Blue→Red: symmetric with R↔B labels.
  Cost: O(n) per flip — for K₄₃ (n=43): 84 array ops per step.

Two-layer certificate protocol (same as v2):
  Layer 1: SA guided by triangle kernel → low-density coloring
  Layer 2: Explicit K₅ exhaustive check → certificate if count = 0
"""

import numpy as np
from itertools import combinations
import random
import math
import time


# ---------------------------------------------------------------------------
# Triangle Adjacency Lattice
# ---------------------------------------------------------------------------
class TriangleLattice:
    """
    Maintains t_R[i][j] and t_B[i][j] tables.
    T = Σ_{(i,j)} [t_R³ + t_B³].

    Memory: O(n²) — two n×n int16 matrices.
    Flip cost: O(n) — update all rows involving u and v.
    """

    def __init__(self, n, coloring):
        self.n        = n
        self.coloring = dict(coloring)
        # Edge adjacency matrix for fast lookup
        self.adj      = np.zeros((n, n), dtype=np.int8)
        for (i, j), c in self.coloring.items():
            self.adj[i, j] = self.adj[j, i] = c  # 0=Red, 1=Blue

        # Triangle tables
        self.t_R = np.zeros((n, n), dtype=np.int32)
        self.t_B = np.zeros((n, n), dtype=np.int32)

        # Build t_R, t_B from adjacency: t_R[u][v] = (A_R @ A_R)[u,v] - A_R[u,v]
        # where A_R[i,j] = (adj[i,j]==0 and i≠j)
        A_R = (self.adj == 0).astype(np.int32)
        A_B = (self.adj == 1).astype(np.int32)
        np.fill_diagonal(A_R, 0)
        np.fill_diagonal(A_B, 0)

        t_R_raw = A_R @ A_R   # t_R_raw[u,v] = # of w with A_R[u,w] and A_R[w,v]
        t_B_raw = A_B @ A_B

        # Remove self-path contribution (w=u or w=v already excluded by diagonal=0)
        self.t_R = t_R_raw - A_R  # subtract the direct edge contribution
        self.t_B = t_B_raw - A_B
        np.fill_diagonal(self.t_R, 0)
        np.fill_diagonal(self.t_B, 0)

    def tension(self):
        # Sum over upper triangle only (each undirected edge once)
        T = 0
        iu, iv = np.triu_indices(self.n, k=1)
        tR = self.t_R[iu, iv]
        tB = self.t_B[iu, iv]
        return int((tR**3 + tB**3).sum())

    def flip(self, edge):
        """Flip edge (u,v). O(n) update to t_R, t_B tables."""
        u, v = edge
        c_old = self.coloring[edge]
        c_new = 1 - c_old
        self.coloring[edge] = c_new
        self.adj[u, v] = self.adj[v, u] = c_new

        # Update all t_R[u][w] and t_B[u][w] for w ≠ u,v
        # and t_R[v][w], t_B[v][w] for w ≠ u,v
        for w in range(self.n):
            if w == u or w == v:
                continue
            c_wv = int(self.adj[w, v])
            c_wu = int(self.adj[w, u])

            if c_old == 0:          # Red → Blue
                # Flip (u,v): v was red neighbour of u
                if c_wv == 0:       # e(w,v)=R: v was common R-nbr of u,w
                    self.t_R[u, w] -= 1
                    self.t_R[w, u] -= 1
                else:               # e(w,v)=B: v is now common B-nbr of u,w
                    self.t_B[u, w] += 1
                    self.t_B[w, u] += 1
                # Flip (v,u): u was red neighbour of v
                if c_wu == 0:
                    self.t_R[v, w] -= 1
                    self.t_R[w, v] -= 1
                else:
                    self.t_B[v, w] += 1
                    self.t_B[w, v] += 1
            else:                   # Blue → Red
                if c_wv == 1:       # e(w,v)=B: v was common B-nbr of u,w
                    self.t_B[u, w] -= 1
                    self.t_B[w, u] -= 1
                else:
                    self.t_R[u, w] += 1
                    self.t_R[w, u] += 1
                if c_wu == 1:
                    self.t_B[v, w] -= 1
                    self.t_B[w, v] -= 1
                else:
                    self.t_R[v, w] += 1
                    self.t_R[w, v] += 1

    def tension_after_flip(self, edge):
        """Compute ΔT for flipping edge without committing. O(n)."""
        u, v = edge
        c_old = self.coloring[edge]
        delta = 0

        for w in range(self.n):
            if w == u or w == v:
                continue
            c_wv = int(self.adj[w, v])
            c_wu = int(self.adj[w, u])

            if c_old == 0:          # Red → Blue
                if c_wv == 0:
                    tRuw_new = self.t_R[u, w] - 1
                    delta += tRuw_new**3 - self.t_R[u, w]**3
                else:
                    tBuw_new = self.t_B[u, w] + 1
                    delta += tBuw_new**3 - self.t_B[u, w]**3
                if c_wu == 0:
                    tRvw_new = self.t_R[v, w] - 1
                    delta += tRvw_new**3 - self.t_R[v, w]**3
                else:
                    tBvw_new = self.t_B[v, w] + 1
                    delta += tBvw_new**3 - self.t_B[v, w]**3
            else:                   # Blue → Red
                if c_wv == 1:
                    tBuw_new = self.t_B[u, w] - 1
                    delta += tBuw_new**3 - self.t_B[u, w]**3
                else:
                    tRuw_new = self.t_R[u, w] + 1
                    delta += tRuw_new**3 - self.t_R[u, w]**3
                if c_wu == 1:
                    tBvw_new = self.t_B[v, w] - 1
                    delta += tBvw_new**3 - self.t_B[v, w]**3
                else:
                    tRvw_new = self.t_R[v, w] + 1
                    delta += tRvw_new**3 - self.t_R[v, w]**3
        return delta

    def copy_coloring(self):
        return dict(self.coloring)


# ---------------------------------------------------------------------------
# Explicit clique counter
# ---------------------------------------------------------------------------
def count_mono_cliques(coloring, n, k, max_checks=None):
    count = checked = 0
    for verts in combinations(range(n), k):
        edges = [(min(a,b), max(a,b)) for a,b in combinations(verts,2)]
        colors = [coloring[e] for e in edges]
        if len(set(colors)) == 1:
            count += 1
        checked += 1
        if max_checks and checked >= max_checks:
            break
    return count, checked


# ---------------------------------------------------------------------------
# SA with triangle kernel
# ---------------------------------------------------------------------------
def anneal_triangle(n, coloring_init,
                    T_init=None, T_final=0.5, alpha=0.9995,
                    max_steps=50000, exploration=0.15, rng=None):
    if rng is None:
        rng = random.Random()
    if T_init is None:
        T_init = float(n ** 3 // 4)

    edges = list(combinations(range(n), 2))
    lat   = TriangleLattice(n, coloring_init)
    temp  = T_init
    T_now = lat.tension()
    hist  = [T_now]

    for step in range(max_steps):
        # candidate
        if rng.random() < exploration:
            edge = rng.choice(edges)
        else:
            sample = rng.choices(edges, k=min(20, len(edges)))
            edge   = min(sample, key=lambda e: lat.tension_after_flip(e))

        delta = lat.tension_after_flip(edge)
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            lat.flip(edge)
            T_now += delta

        if step % 1000 == 0:
            hist.append(T_now)

        temp = max(temp * alpha, T_final)

    return lat.copy_coloring(), T_now, hist


# ---------------------------------------------------------------------------
# Phase A — Validation ladder (reproduces R(3,3))
# ---------------------------------------------------------------------------
def phase_a(n_restarts=20, seed=7):
    print("  ─── Phase A: Validation ladder (triangle kernel + explicit K₃) ──")
    rng = random.Random(seed)

    cases = [
        (5, "K₅", "K₃-free colorings exist (R(3,3)≤5)"),
        (6, "K₆", "NO K₃-free coloring (R(3,3)=6)"),
    ]
    for n, label, note in cases:
        edges = list(combinations(range(n), 2))
        k3_zero, best_k3, T_list = 0, 999, []
        for _ in range(n_restarts):
            init = {e: rng.randint(0,1) for e in edges}
            col, T_fin, _ = anneal_triangle(n, init,
                T_init=float(n**3), max_steps=5000, rng=rng)
            T_list.append(T_fin)
            k3, _ = count_mono_cliques(col, n, 3)
            if k3 < best_k3: best_k3 = k3
            if k3 == 0:      k3_zero += 1
        pct = k3_zero / n_restarts * 100
        print(f"  {label}  mono-K₃=0: {k3_zero}/{n_restarts} ({pct:.0f}%)"
              f"  T_min={min(T_list):8.0f}  best_K₃={best_k3}  [{note}]")
    print()


# ---------------------------------------------------------------------------
# Phase B — R(5,5) search
# ---------------------------------------------------------------------------
def phase_b(sweep_ns=(43, 44), n_restarts=10, max_steps=100000, seed=42):
    print("  ─── Phase B: R(5,5) sweep (triangle kernel + exhaustive K₅) ─────")
    print("  Bounds: 43 ≤ R(5,5) ≤ 48")
    print()
    rng = random.Random(seed)

    for n in sweep_ns:
        edges  = list(combinations(range(n), 2))
        n_k5   = math.comb(n, 5)
        print(f"  K_{n}  ({len(edges)} edges, {n_k5:,} K₅ subgraphs)")

        t0 = time.time()
        best_T, best_k5, cert = float('inf'), 999, False

        for trial in range(n_restarts):
            init = {e: rng.randint(0,1) for e in edges}

            # Triangle kernel SA
            t1 = time.time()
            col, T_fin, _ = anneal_triangle(n, init,
                T_init=float(n**3), alpha=0.9995,
                max_steps=max_steps, rng=rng)
            sa_time = time.time() - t1

            # Exhaustive K₅ check
            t2 = time.time()
            k5, _ = count_mono_cliques(col, n, 5)
            k5_time = time.time() - t2

            if k5 < best_k5: best_k5, best_T = k5, T_fin
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
    print("  R(5,5) SANDBOX v3  —  Triangle Adjacency Kernel")
    print("  T = Σ_{(u,v)}[t_R(u,v)³ + t_B(u,v)³]  |  2-hop K₅ sensitivity")
    print("▓" * 72)
    print()
    print("  t_R(u,v) = common RED neighbours of u and v")
    print("  t_B(u,v) = common BLUE neighbours of u and v")
    print("  K₅ edge → t_R ≥ 3  →  cubic spike ≥ 27 (vs ~10 for typical edge)")
    print("  Update cost: O(n) per flip (vs O(1) for v2, worth it for signal)")
    print()
    phase_a()
    phase_b()
    print("  Triangle kernel: heuristic guide. Clique count: ground truth.")
    print("  Certificate (mono-K₅=0) is proof. T alone is not.")


if __name__ == "__main__":
    run()
