"""
validate_nesting_values.py — how do weights flow cube -> cube when nested?

A cell's weight is fixed by its POSITION only:  SW = 9 * exposure, exposure in {0,1,2,3}
  corner(3 faces)=27, edge(2)=18, face-center(1)=9, deep-core(0)=0

Nesting (hierarchy.py):  each of the N^3 macro cells hosts a FULL micro cube.
  global ledger = N^3 * SW(M)  +  SW(N)     <- strictly ADDITIVE, not divided.

This script prints the breakdown so you can see exactly how the books add up.
Run from repo root:  python cortex_v2/validate_nesting_values.py
"""
import sys, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "packages", "livnium-core", "src")
sys.path.insert(0, ROOT)
from livnium_core.lattice import class_counts, symbolic_weight_total as sw, SW, _iter_cells
from livnium_core.hierarchy import global_ledger

def breakdown(N):
    cc = class_counts(N)
    rows = [("corner", cc["corner"], 27), ("edge", cc["edge"], 18),
            ("center", cc["center"], 9), ("core", cc["core"], 0)]
    return rows

N, M = 3, 3
print("=" * 70)
print(f"ONE cube, N={N}: weight comes only from a cell's boundary exposure")
print("=" * 70)
tot = 0
for name, count, w in breakdown(N):
    print(f"  {name:7} x {count:2}  @ SW={w:2}  -> {count*w:4}")
    tot += count * w
print(f"  {'TOTAL':7}              -> {tot}   (= SW({N}) = {sw(N)})")

print()
print("=" * 70)
print(f"NESTED N={N} hosting M={M}: how the values add across the two scales")
print("=" * 70)
macro = sw(N)
micro_one = sw(M)
n_cells = N ** 3
print(f"  macro cube contributes its own ledger ............ SW({N})      = {macro}")
print(f"  each of the {n_cells} macro cells hosts a full micro cube ... SW({M}) = {micro_one}")
print(f"  all micro cubes are IDENTICAL (weight is by position, not host)")
print(f"  micro total = {n_cells} x {micro_one} ........................... = {n_cells*micro_one}")
print(f"  GLOBAL = micro_total + macro = {n_cells*micro_one} + {macro} = {global_ledger(N,M)}")
print()

print("KEY POINT: a macro cell's value does NOT trickle down or get divided.")
print("Each scale keeps its own complete weight budget; the budgets simply ADD.")
print("A micro cube inside a CORNER cell (macro SW=27) has the same internal")
print("SW total as one inside the deep CORE cell (macro SW=0):")
for host_w in (27, 0):
    print(f"   host macro cell SW={host_w:2}  ->  its micro cube still sums to SW({M}) = {micro_one}")
print()
print("So 'going down' = repeat the same conserved 0/9/18/27 pattern one scale")
print("smaller, independently, and add it to the parent's ledger.")
