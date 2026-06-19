"""
validate_sites_via_nesting.py — "how many SITES are possible via nesting?"

Two different meanings of 'site', measured together so the answer is unambiguous:

  ADDRESS sites  = addressable cells in the nested cube = M^(3*depth).
                   Conserved geometry, closed form, astronomically large.
  QUANTUM sites  = faithful entangled qubits in an MPS register.
                   Limited by ENTANGLEMENT, not address space — nesting does
                   NOT raise this ceiling. We re-measure it empirically here.

Run from repo root:  python cortex_v2/validate_sites_via_nesting.py
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mps import MPS
from lattice import Lattice, SU2

M = 3
print("=" * 80)
print(f"ADDRESS sites via nesting (M={M}): cells = {M}^(3*depth)")
print("=" * 80)
for d in [1, 2, 3, 5, 10, 20, 50, 100]:
    cells = M ** (3 * d)
    print(f"  depth {d:4d}:  {cells:.3e} address sites   ({len(str(cells))} digits)")

print()
print("=" * 80)
print("QUANTUM sites: faithful entangled qubits in a cube-driven MPS (max_chi=64)")
print("=" * 80)

def faithful(n, max_chi=64):
    m = MPS(n, max_chi=max_chi); cube = Lattice(); k = 0
    for layer in range(n):
        for s in range(n):
            rot = (k % 23) + 1; cube.apply(rot); m.apply_1q(s, SU2[rot]); k += 1
        for s in range(0, n - 1, 2): m.cnot(s, s + 1)
        for s in range(1, n - 1, 2): m.cnot(s, s + 1)
    return abs(m.trunc_error) < 1e-9, m.max_chi_used, m.trunc_error

last_ok = 0
for n in range(6, 21):
    ok, bond, err = faithful(n)
    tag = "FAITHFUL" if ok else "broken  "
    if ok: last_ok = n
    print(f"  n={n:2d} sites:  {tag}  max_bond={bond:3d}/64  trunc_err={err:.2e}")
print(f"\n  => deepest FAITHFUL quantum register at chi=64:  {last_ok} sites")

print()
print("ANSWER:")
print(f"  * As ADDRESS sites (nesting): effectively unlimited — e.g. depth 10 = "
      f"{M**30:.2e} sites, conserved, cheap.")
print(f"  * As QUANTUM sites (entangled): only ~{last_ok} faithful at chi=64; "
      f"nesting does NOT raise this.")
print("  The big number is address space; the small number is entangled compute.")
