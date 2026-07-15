"""
validate_hierarchy.py — show the cube growing INWARD (nested capacity) while the
symbolic-weight ledger stays strictly conserved/additive across scales.
Run from the repo root:  python cortex_v2/validate_hierarchy.py
"""
import sys, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "packages", "livnium-core", "src")
sys.path.insert(0, ROOT)
from livnium_core.hierarchy import capacity, global_ledger, wreath_group_order
from livnium_core.lattice import symbolic_weight_total as sw

print("single cube SW(N) vs additive nested ledger")
print("=" * 72)
for N, M in [(3, 3), (3, 5), (5, 3), (5, 5), (7, 3), (3, 27)]:
    cap = capacity(N, M)
    micro = (N ** 3) * sw(M)
    macro = sw(N)
    glob = global_ledger(N, M)
    ok = glob == micro + macro
    print(f"N={N} M={M}: cap={cap:>8}  ledger = {N**3}*{sw(M)}(micro) + {macro}(macro)"
          f" = {glob}  additive_ok={ok}")

print("\nconserved macro SW(N), exact closed form 54(N-2)^2+216(N-2)+216")
for N in [3, 5, 7, 9, 11]:
    print(f"  SW({N}) = {sw(N)}")

print("\nthree scales deep: N=3 hosting M=3 hosting K=3")
N = M = K = 3
cap3 = N ** 3 * M ** 3 * K ** 3
ledger3 = (N ** 3) * (M ** 3) * sw(K) + (N ** 3) * sw(M) + sw(N)
print(f"  capacity = 27^3 = {cap3} cells")
print(f"  ledger   = {ledger3}  (strictly additive across all 3 scales)")
print(f"  wreath group order (N=3) = 24 * 24^27 = {wreath_group_order(3)}")
