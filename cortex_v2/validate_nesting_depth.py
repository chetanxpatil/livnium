"""
validate_nesting_depth.py — how DEEP can the nested cube go (how many layers)?

Split-&-decrease model (your design): each layer partitions a parent cell's value
among its M^3 children. Three independent limits:

  (1) conservation   — exact at ANY depth (Fraction arithmetic): total = SW(top)
  (2) capacity count — closed form M^(3*depth): astronomically large but free
  (3) per-cell value — float64 underflows when the smallest share^depth < ~1e-308

So "how deep" depends on which limit you mean. We find all three.
Run from repo root:  python cortex_v2/validate_nesting_depth.py
"""
import sys, os
from fractions import Fraction
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from livnium_core.lattice import SW, _iter_cells, symbolic_weight_total as sw

M = 3
SWM = sw(M)                       # 486
shares = [Fraction(SW(c, M), SWM) for c in _iter_cells(M)]   # exact partition shares
nonzero = [s for s in shares if s > 0]
max_share = max(nonzero)          # 27/486 = 1/18  (slowest decay)
min_share = min(nonzero)          # 9/486  = 1/54  (fastest decay)
TOP = Fraction(sw(3))             # top cube budget = 486

print("=" * 78)
print(f"NESTING DEPTH (M={M}):  top budget={TOP}  shares: max={max_share} min={min_share}")
print("=" * 78)

print("\n(1) CONSERVATION across depth — exact Fraction arithmetic:")
for d in [1, 5, 20, 100, 1000, 10000]:
    # total after splitting: sum over all leaves = TOP * (sum of shares)^d = TOP * 1
    total = TOP * (sum(shares)) ** d
    print(f"   depth {d:6d}:  total = {total}   conserved={total == TOP}")

print("\n(2) CAPACITY (cells = M^(3*depth)) — closed form, big-int, no limit:")
for d in [1, 5, 10, 50, 100]:
    cells = M ** (3 * d)
    print(f"   depth {d:4d}:  {cells:.3e} cells  ({len(str(cells))} digits)")

print("\n(3) PER-CELL VALUE underflow in float64 (~smallest normal 2.2e-308):")
import math
TINY = 2.225e-308
# slowest-shrinking path uses max_share; it underflows LAST -> the true max depth
slow = 27.0; fast = 27.0
d_slow = d_fast = 0
sf, ff = float(max_share), float(min_share)
while slow > TINY:
    slow *= sf; d_slow += 1
while fast > TINY:
    fast *= ff; d_fast += 1
print(f"   fastest-decaying path (x{float(min_share):.5f}/level): underflows at depth {d_fast}")
print(f"   slowest-decaying path (x{float(max_share):.5f}/level): underflows at depth {d_slow}")
print(f"   -> in float64, deepest meaningful layer is ~{d_slow} before values vanish")

print()
print("READING:")
print(f"  * conservation: UNLIMITED depth — total stays {TOP} exactly, forever.")
print("  * capacity: unlimited in closed form (you never enumerate cells).")
print(f"  * numerics: float64 per-cell values vanish by ~depth {d_slow};")
print("    use exact rationals (Fraction) or log-domain to go arbitrarily deep.")
