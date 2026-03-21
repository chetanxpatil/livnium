"""
R(5,5) Sandbox v2  —  Cubic Monochromatic Degree Kernel
========================================================
Corrects the v1 vertex-sum kernel which measured BALANCE, not K₅ content.

v1 post-mortem — "parity loophole":
  T = Σσ(v)²  reaches T=0 whenever every vertex has even degree because
  a perfectly balanced coloring (equal red/blue per vertex) is trivially
  achievable via Eulerian circuit decomposition.  K₄₅ (degree 44, even)
  found T=0 in 0.0s — not a K₅-free certificate, just arithmetic.
  R(3,3) validation appeared correct only because R(3,3)=6 happens to
  align with the even→odd degree boundary (parity coincidence).

v2 fix — Cubic Monochromatic Degree Kernel:
  σ_R(v) = number of RED incident edges
  σ_B(v) = number of BLUE incident edges
  T = Σ_v [ σ_R(v)³ + σ_B(v)³ ]

  Why this is better:
  - x³ is strictly convex: heavily penalises concentration, not just imbalance
  - T is minimised when same-colour degree is as uniform as possible
  - Eliminates trivial T=0 states (since σ_R+σ_B = deg(v) > 0 always)
  - A vertex in a monochromatic K₅ has σ_R ≥ 4  →  contributes ≥ 64

  Why this is STILL a heuristic, not a proof:
  - σ_R(v) ≥ 4 is NECESSARY but not SUFFICIENT for K₅ membership
  - T_min > 0 does not prove the coloring is K₅-free
  - Only an explicit exhaustive check provides a certificate

Two-layer protocol:
  Layer 1: SA guided by cubic kernel  →  find low-concentration coloring
  Layer 2: Explicit clique checker     →  verify actual K₃ / K₅ count
  Certificate: explicit count = 0     →  valid K-free coloring found

O(1)-per-flip SA update:
  Flipping e(u,v) from Red to Blue:
    δu = (σ_R(u)-1)³ + (σ_B(u)+1)³ - σ_R(u)³ - σ_B(u)³
    δv = (σ_R(v)-1)³ + (σ_B(v)+1)³ - σ_R(v)³ - σ_B(v)³
    ΔT = δu + δv
"""

import numpy as np
from itertools import combinations
import random
import math
import time

# ---------------------------------------------------------------------------
# Cubic Monochromatic Degree Lattice
# ---------------------------------------------------------------------------
class CubicLattice:
    """
    Maintains σ_R(v), σ_B(v) incrementally.
    T = Σ_v [σ_R(v)³ + σ_B(v)³].  Single flip touches exactly 2 vertices.
    """

    def __init__(self, n, coloring):
        self.n         = n
        self.coloring  = dict(coloring)
        self.sigma_R   = np.zeros(n, dtype=np.int64)
        self.sigma_B   = np.zeros(n, dtype=np.int64)

        for (u, v), c in self.coloring.items():
            if c == 0:                       # Red
                self.sigma_R[u] += 1
                self.sigma_R[v] += 1
            else:                            # Blue
                self.sigma_B[u] += 1
                self.sigma_B[v] += 1

    def tension(self):
        return int((self.sigma_R ** 3 + self.sigma_B ** 3).sum())

    def delta_tension(self, edge):
        """ΔT if edge (u,v) is flipped. O(1)."""
        u, v = edge
        c    = self.coloring[edge]
        if c == 0:                           # Red → Blue
            du = ((self.sigma_R[u]-1)**3 + (self.sigma_B[u]+1)**3
                  - self.sigma_R[u]**3   - self.sigma_B[u]**3)
            dv = ((self.sigma_R[v]-1)**3 + (self.sigma_B[v]+1)**3
                  - self.sigma_R[v]**3   - self.sigma_B[v]**3)
        else:                                # Blue → Red
            du = ((self.sigma_R[u]+1)**3 + (self.sigma_B[u]-1)**3
                  - self.sigma_R[u]**3   - self.sigma_B[u]**3)
            dv = ((self.sigma_R[v]+1)**3 + (self.sigma_B[v]-1)**3
                  - self.sigma_R[v]**3   - self.sigma_B[v]**3)
        return int(du + dv)

    def flip(self, edge):
        u, v = edge
        c    = self.coloring[edge]
        self.coloring[edge] ^= 1
        if c == 0:                           # Red → Blue
            self.sigma_R[u] -= 1;  self.sigma_B[u] += 1
            self.sigma_R[v] -= 1;  self.sigma_B[v] += 1
        else:                                # Blue → Red
            self.sigma_R[u] += 1;  self.sigma_B[u] -= 1
            self.sigma_R[v] += 1;  self.sigma_B[v] -= 1

    def copy_coloring(self):
        return dict(self.coloring)


# ---------------------------------------------------------------------------
# Explicit clique counters  (ground truth — slow for large n)
# ---------------------------------------------------------------------------
def count_mono_cliques(coloring, n, clique_size, max_checks=None):
    """
    Exhaustive count of monochromatic clique_size subgraphs.
    Returns (count, total_checked).  Use max_checks to limit for large n.
    """
    count   = 0
    checked = 0
    for verts in combinations(range(n), clique_size):
        clique_edges = [(min(a,b), max(a,b))
                        for a,b in combinations(verts, 2)]
        colors = [coloring[e] for e in clique_edges]
        if len(set(colors)) == 1:
            count += 1
        checked += 1
        if max_checks and checked >= max_checks:
            break
    return count, checked


# ---------------------------------------------------------------------------
# Simulated Annealing on CubicLattice
# ---------------------------------------------------------------------------
def anneal_cubic(n, coloring_init,
                 T_init=None, T_final=0.5, alpha=0.998,
                 max_steps=50000, exploration=0.15, rng=None):
    """
    SA using cubic monochromatic degree tension.  O(1) per step.
    Returns (final_coloring, T_final_tension, history_sampled).
    """
    if rng is None:
        rng = random.Random()
    if T_init is None:
        T_init = float(n ** 2)

    edges = list(combinations(range(n), 2))
    lat   = CubicLattice(n, coloring_init)
    temp  = T_init
    T_now = lat.tension()
    hist  = [T_now]

    for step in range(max_steps):
        # candidate edge
        if rng.random() < exploration:
            edge = rng.choice(edges)
        else:
            # greedy among a random sample
            sample  = rng.choices(edges, k=min(30, len(edges)))
            best_e  = min(sample, key=lambda e: lat.delta_tension(e))
            edge    = best_e

        delta = lat.delta_tension(edge)
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            lat.flip(edge)
            T_now += delta

        if step % 1000 == 0:
            hist.append(T_now)

        temp = max(temp * alpha, T_final)

    return lat.copy_coloring(), T_now, hist


# ---------------------------------------------------------------------------
# Phase A  —  Validation ladder: reproduces R(3,3) with explicit check
# ---------------------------------------------------------------------------
def phase_a_validation(n_restarts=30, seed=7):
    print("  ─── Phase A: Validation ladder (cubic kernel + explicit K₃ check) ─")
    print("  T = Σ_v [σ_R³ + σ_B³]  — no trivial T=0 states")
    print()
    rng = random.Random(seed)

    cases = [
        (3, "K₃",  "trivially K₃-free possible"),
        (4, "K₄",  "K₃-free colorings exist"),
        (5, "K₅",  "K₃-free colorings exist (R(3,3)≤5)"),
        (6, "K₆",  "NO K₃-free coloring (R(3,3)=6)"),
    ]

    for n, label, note in cases:
        edges = list(combinations(range(n), 2))
        k3_zeros, best_k3, T_list = 0, 999, []

        for _ in range(n_restarts):
            init = {e: rng.randint(0,1) for e in edges}
            final_col, T_fin, _ = anneal_cubic(n, init,
                T_init=float(n**2), max_steps=8000, rng=rng)
            T_list.append(T_fin)

            k3, _ = count_mono_cliques(final_col, n, 3)
            if k3 < best_k3: best_k3 = k3
            if k3 == 0:      k3_zeros += 1

        pct = k3_zeros / n_restarts * 100
        print(f"  {label:<5}  mono-K₃=0: {k3_zeros:2}/{n_restarts}"
              f" ({pct:5.1f}%)  T_min={min(T_list):6.0f}"
              f"  best_K₃={best_k3}  [{note}]")
    print()


# ---------------------------------------------------------------------------
# Phase B  —  R(5,5) sweep with cubic kernel + exhaustive K₅ verification
# ---------------------------------------------------------------------------
def phase_b_r55(sweep_ns=(43, 44, 45), n_restarts=20,
                max_steps=200000, seed=42):
    print("  ─── Phase B: R(5,5) sweep (cubic kernel + exhaustive K₅ check) ───")
    print("  Bounds: 43 ≤ R(5,5) ≤ 48")
    print("  Certificate: explicit mono-K₅ count = 0 after SA")
    print()

    rng = random.Random(seed)

    for n in sweep_ns:
        n_edges = n * (n-1) // 2
        n_k5    = math.comb(n, 5)
        print(f"  K_{n}  ({n_edges} edges, {n_k5:,} possible K₅ subgraphs)")

        t0       = time.time()
        best_T   = float('inf')
        best_k5  = 999
        best_col = None
        cert_found = False

        for trial in range(n_restarts):
            edges = list(combinations(range(n), 2))
            init  = {e: rng.randint(0,1) for e in edges}
            final_col, T_fin, _ = anneal_cubic(
                n, init,
                T_init=float(n**2), alpha=0.9995,
                max_steps=max_steps, exploration=0.15, rng=rng
            )

            # Exhaustive K₅ check on the final coloring
            t_check = time.time()
            k5, total_checked = count_mono_cliques(final_col, n, 5)
            check_time = time.time() - t_check

            if k5 < best_k5:
                best_k5  = k5
                best_T   = T_fin
                best_col = dict(final_col)

            elapsed = time.time() - t0
            print(f"    restart {trial+1:2d}/{n_restarts}  "
                  f"T={T_fin:8.0f}  mono-K₅={k5:4d}"
                  f"  K₅-check={check_time:.1f}s  ({elapsed:.0f}s total)")

            if k5 == 0:
                cert_found = True
                print(f"    ✓ CERTIFICATE: K_{n} has K₅-free 2-coloring")
                print(f"      R(5,5) > {n}  (consistent with known bounds)")
                break

        result = f"✓ K₅-FREE CERTIFICATE  → R(5,5) > {n}" if cert_found \
                 else f"⚠ best mono-K₅={best_k5}  (heuristic only, not a proof)"
        print(f"  K_{n} result: {result}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("▓" * 72)
    print("  R(5,5) SANDBOX v2  —  Cubic Monochromatic Degree Kernel")
    print("  T = Σ_v[σ_R(v)³ + σ_B(v)³]  |  Two-layer: SA guide + explicit check")
    print("▓" * 72)
    print()
    print("  Kernel properties:")
    print("  - x³ strictly convex: penalises same-colour concentration")
    print("  - No trivial T=0 (degree always > 0)")
    print("  - Vertex in mono K₅: σ_R ≥ 4  →  contributes ≥ 64 (vs ~24 balanced)")
    print("  - T_min is a heuristic; explicit K₅ count is the certificate")
    print()

    phase_a_validation()
    phase_b_r55()

    print("  ═" * 36)
    print("  Kernel: heuristic guide.  Clique count: ground truth.")
    print("  Certificate (mono-K₅=0) is proof.  T_min alone is not.")


if __name__ == "__main__":
    run()
