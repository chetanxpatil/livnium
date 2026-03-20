"""
livnium_ramsey_v2_stochastic.py  (v2 — FAST)
=============================================
Trap-Resistant Ramsey(5,5) Hunter
----------------------------------
Architecture:
  - Binary edge flip + incremental violation counting  (90× faster than full scan)
  - Simulated Annealing  (Metropolis accept/reject)
  - Lévy Flight          (multi-edge kick to escape basins)
  - Similarity Tax       (metadynamics repulsion from visited states)

Key speedup:
  Flipping edge (i,j) only affects subsets containing BOTH i and j.
  Count = C(41,3) = 10,660  vs  962,598 full scan.

Goal: 43-vertex graph with no K_5 clique and no independent set of size 5.
R(5,5) is known to lie in [43, 48].

Usage:
    python3 livnium_ramsey_v2_stochastic.py
"""

import numpy as np
import time
import json
import os
from itertools import combinations

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

N          = 43          # vertices
K          = 5           # forbidden clique / indset size

N_STEPS    = 5_000_000   # total flip attempts (fast now — millions is fine)
LOG_EVERY  = 50_000

# Annealing  (temperature in "violation units")
TEMP_START = 3.0
TEMP_END   = 0.01
ANNEAL_END = 4_000_000   # steps over which temp drops

# Lévy Flight
LEVY_INTERVAL  = 2_000   # attempt multi-edge kick every N steps
LEVY_MIN_FLIPS = 2
LEVY_MAX_FLIPS = 30      # heavy-tail: occasionally flip many edges at once
LEVY_ALPHA     = 1.5     # Lévy exponent

# Similarity Tax
SIM_INTERVAL   = 20_000  # snapshot every N steps
SIM_MEMORY     = 30      # graphs to remember
SIM_STRENGTH   = 0.8     # extra violation penalty per unit similarity to past

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_graph.json")

# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE: for each edge (i,j), which K-subsets contain it?
# ─────────────────────────────────────────────────────────────────────────────

def build_edge_subsets(n, k):
    """
    For each edge (i,j), precompute the list of all k-subsets containing both i and j.
    These are formed by choosing (k-2) more vertices from the remaining (n-2).
    Returns dict: (i,j) -> list of tuples (sorted k-subset)
    """
    all_verts = list(range(n))
    edge_to_subsets = {}
    print(f"Building edge→subset index (C({n},2) edges, each with C({n-2},{k-2}) subsets)...",
          end=" ", flush=True)
    for i in range(n):
        for j in range(i+1, n):
            others = [v for v in all_verts if v != i and v != j]
            subs = [tuple(sorted([i, j] + list(combo)))
                    for combo in combinations(others, k-2)]
            edge_to_subsets[(i, j)] = subs
    print(f"done. Each edge touches {len(subs):,} subsets.")
    return edge_to_subsets

# ─────────────────────────────────────────────────────────────────────────────
# VIOLATION COUNTING
# ─────────────────────────────────────────────────────────────────────────────

def count_violations_full(adj, n=N, k=K):
    """Full recount — used only at init and logging."""
    cliques = indsets = 0
    for sub in combinations(range(n), k):
        e = sum(adj[sub[a], sub[b]] for a in range(k) for b in range(a+1, k))
        total = k*(k-1)//2
        if e == total: cliques += 1
        if e == 0:     indsets += 1
    return cliques, indsets

def delta_violations(adj, i, j, edge_subsets):
    """
    Compute change in (cliques + indsets) if edge (i,j) is flipped.
    Only inspects subsets containing both i and j.
    Returns delta (int), can be negative = improvement.
    """
    delta = 0
    currently_edge = adj[i, j]
    total = K*(K-1)//2

    for sub in edge_subsets[(i, j)]:
        # current edge count in this subset
        e = sum(adj[sub[a], sub[b]] for a in range(K) for b in range(a+1, K))

        # violation BEFORE flip
        before = int(e == total) + int(e == 0)

        # violation AFTER flip (one edge toggled)
        e_after = e + (1 if currently_edge == 0 else -1)
        after = int(e_after == total) + int(e_after == 0)

        delta += after - before

    return delta

# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY TAX
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityTax:
    """
    Keeps a rolling memory of past adjacency matrices (as flat bit-vectors).
    Returns an extra penalty term when current graph looks like a visited one.
    """
    def __init__(self, memory=SIM_MEMORY, strength=SIM_STRENGTH):
        self.memory   = memory
        self.strength = strength
        self.history  = []   # list of flat uint8 arrays

    def snapshot(self, adj):
        flat = adj[np.triu_indices(N, k=1)].copy()
        self.history.append(flat)
        if len(self.history) > self.memory:
            self.history.pop(0)

    def similarity_penalty(self, adj):
        """
        Returns a scalar penalty = strength * max_similarity_to_any_past_graph.
        Similarity = fraction of edges in common / total edges.
        """
        if not self.history:
            return 0.0
        flat = adj[np.triu_indices(N, k=1)]
        n_edges = len(flat)
        max_sim = max(
            np.sum(flat == h) / n_edges for h in self.history
        )
        return self.strength * max_sim

# ─────────────────────────────────────────────────────────────────────────────
# LÉVY FLIGHT  — number of edges to flip drawn from heavy-tail distribution
# ─────────────────────────────────────────────────────────────────────────────

def levy_flip_count(alpha=LEVY_ALPHA,
                    lo=LEVY_MIN_FLIPS, hi=LEVY_MAX_FLIPS):
    """
    Draw number of edges to flip from discrete Lévy-like distribution.
    P(k) ∝ k^{-alpha}  for k in [lo, hi].
    """
    weights = np.array([k**(-alpha) for k in range(lo, hi+1)])
    weights /= weights.sum()
    return np.random.choice(range(lo, hi+1), p=weights)

# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURE SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def temperature(step, t_start=TEMP_START, t_end=TEMP_END, anneal_end=ANNEAL_END):
    t = min(step / anneal_end, 1.0)
    cos = (1 - np.cos(np.pi * t)) / 2.0
    return t_start * (1 - cos) + t_end * cos   # hot → cold

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run():
    rng = np.random.default_rng(42)

    # Random start graph
    adj = rng.integers(0, 2, size=(N, N), dtype=np.int8)
    adj = np.triu(adj, 1)
    adj = adj + adj.T   # symmetric, no self-loops

    cliques, indsets = count_violations_full(adj)
    violations = cliques + indsets

    best_adj    = adj.copy()
    best_v      = violations
    best_c, best_i = cliques, indsets

    sim_tax = SimilarityTax()

    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    n_edges   = len(all_edges)

    print("=" * 60)
    print(f"  Livnium Ramsey v2 FAST  —  N={N}, K={K}")
    print(f"  {N_STEPS:,} flip attempts  |  {n_edges} edges")
    print(f"  Temp: {TEMP_START} → {TEMP_END}")
    print("=" * 60)
    print(f"Initial violations: {violations} (cliques={cliques}, indsets={indsets})\n")

    t0 = time.time()
    accepts = 0

    for step in range(1, N_STEPS + 1):

        temp = temperature(step)

        # ── Lévy Flight: flip multiple edges at once ───────────────────────
        if step % LEVY_INTERVAL == 0:
            n_flips = levy_flip_count()
            chosen  = rng.choice(n_edges, size=n_flips, replace=False)
            flip_edges = [all_edges[k] for k in chosen]
        else:
            # Normal: flip one random edge
            k = rng.integers(0, n_edges)
            flip_edges = [all_edges[k]]

        # ── Compute total delta for all flipped edges ──────────────────────
        # Apply flips one by one, accumulate delta, then decide accept/reject
        total_delta = 0
        for (fi, fj) in flip_edges:
            d = delta_violations(adj, fi, fj, EDGE_SUBSETS)
            total_delta += d
            # Temporarily apply so subsequent deltas in same Lévy kick are consistent
            adj[fi, fj] ^= 1
            adj[fj, fi]  = adj[fi, fj]

        # ── Similarity Tax ─────────────────────────────────────────────────
        sim_pen = sim_tax.similarity_penalty(adj)
        effective_delta = total_delta + sim_pen

        # ── Metropolis accept / reject ─────────────────────────────────────
        if effective_delta <= 0 or rng.random() < np.exp(-effective_delta / temp):
            # Accept: violations already updated in adj
            violations += total_delta
            accepts += 1
        else:
            # Reject: undo all flips
            for (fi, fj) in flip_edges:
                adj[fi, fj] ^= 1
                adj[fj, fi]  = adj[fi, fj]

        # ── Track best ─────────────────────────────────────────────────────
        if violations < best_v:
            best_v   = violations
            best_adj = adj.copy()
            # Recount accurately (violations can drift with Lévy multi-flips)
            best_c, best_i = count_violations_full(best_adj)
            best_v = best_c + best_i

        # ── Similarity snapshot ────────────────────────────────────────────
        if step % SIM_INTERVAL == 0:
            sim_tax.snapshot(adj)

        # ── Logging ───────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            rate    = step / elapsed
            c, i = count_violations_full(adj)
            violations = c + i   # accurate resync
            marker = " ★" if violations <= best_v else ""
            print(f"Step {step:>8,} | T={temp:.4f} | "
                  f"violations={violations:>5} (c={c} i={i}) | "
                  f"best={best_v} | "
                  f"{rate/1000:.0f}k steps/s{marker}")

            if best_v == 0:
                print("\n🎉  ZERO VIOLATIONS — Ramsey(5,5) witness found!")
                break

    # ── Final ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s  |  {accepts/N_STEPS*100:.1f}% accept rate")
    print(f"  BEST: violations={best_v}  (cliques={best_c}, indsets={best_i})")
    print(f"{'='*60}")

    result = {
        "n_vertices": N,
        "clique_size": K,
        "indset_size": K,
        "violations": best_v,
        "n_cliques": int(best_c),
        "n_indsets": int(best_i),
        "adjacency": best_adj.tolist(),
    }
    with open(SAVE_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Best graph saved → {SAVE_PATH}")
    return best_adj, best_v


# ─────────────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────────────

print(f"Building edge→subset index for N={N}, K={K} ...", end=" ", flush=True)
EDGE_SUBSETS = build_edge_subsets(N, K)
print()

if __name__ == "__main__":
    run()
