"""
validate_combined.py — SMALL test combining both mechanisms:

  (A) the nested cube hierarchy  -> conserved, inward-growing ADDRESS capacity
  (B) the cube-driven MPS        -> a quantum register over a SLICE of those cells

The hierarchy decides how many addressable cells exist (cheap, conserved). We then
try to realize a slice of them as an entangled MPS register, driven by the cube's
SU(2) lift. The point: capacity grows freely & stays conserved, but the FAITHFUL
entangled slice stays tiny. Both truths, in one run.

Run from repo root:  python cortex_v2/validate_combined.py
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mps import MPS
from lattice import Lattice, SU2
from livnium_core.hierarchy import capacity, global_ledger
from livnium_core.lattice import symbolic_weight_total as sw

MAX_SIM = 16  # never try to entangle more than this many sites (keeps it small)

def cube_driven_register(n_sites, max_chi=64, depth=None):
    """Build an MPS over n_sites cells; each site's gate = SU(2) lift of the
    cube orientation reached by walking the rotation group cell-by-cell.
    `depth` entangling layers drive real (volume-law) entanglement."""
    depth = depth if depth is not None else n_sites
    m = MPS(n_sites, max_chi=max_chi)
    cube = Lattice()
    k = 0
    for layer in range(depth):
        for s in range(n_sites):
            rot = (k % 23) + 1        # cell index -> non-identity rotation 1..23
            cube.apply(rot)          # advance the (macro) cube; ledger stays fixed
            m.apply_1q(s, SU2[rot])  # cube gate onto this site
            k += 1
        for s in range(0, n_sites - 1, 2):
            m.cnot(s, s + 1)
        for s in range(1, n_sites - 1, 2):
            m.cnot(s, s + 1)
    return m, cube

print("=" * 84)
print("COMBINED: nested-cube address capacity (conserved)  vs  faithful MPS register")
print("=" * 84)
for N, M in [(3, 3), (3, 5), (5, 3), (5, 5)]:
    cap = capacity(N, M)                       # (A) inward-grown capacity
    ledger = global_ledger(N, M)
    additive = ledger == (N**3) * sw(M) + sw(N)
    n_sim = min(cap, MAX_SIM)                   # (B) slice we actually entangle
    m, cube = cube_driven_register(n_sim)
    faithful = abs(m.trunc_error) < 1e-9
    print(f"N={N} M={M}: capacity={cap:>7} cells  ledger={ledger:>8} additive={additive} "
          f"cube_SW={cube.total_sw()} bij={cube.is_bijection()}")
    print(f"          -> simulated slice n={n_sim:2d}: max_bond={m.max_chi_used:3d}/64 "
          f"trunc_err={m.trunc_error:.2e} faithful={faithful}")

print()
print("growing the simulated slice while capacity is 'infinite & conserved':")
for n_sim in [8, 12, 14, 16]:
    m, cube = cube_driven_register(n_sim)
    print(f"  n={n_sim:2d}  max_bond={m.max_chi_used:3d}/64  trunc_err={m.trunc_error:.2e}  "
          f"faithful={abs(m.trunc_error)<1e-9}  (cube ledger still {cube.total_sw()}, conserved)")
print()
print("READING: capacity & ledger scale freely and stay conserved (geometry);")
print("the faithful entangled slice stops at ~12-15 sites (entanglement, not geometry).")
