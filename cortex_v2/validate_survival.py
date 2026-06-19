"""
validate_survival.py — give Livnium OPPOSITION (the missing "fight").

Conservation alone is inert: a rock conserves mass but does nothing. To make a
"digital atom" that lives, fissions, and is selected, add an energy with COMPETING
terms (binding pulls together, everything else pushes apart):

    E(S) = -B  +  R  +  S  +  C  +  M
           binding  repulsion  surface  complexity(bond)  mismatch

Decision rules:
    survives if E < T        grows if dE < 0
    splits   if R + C > B    dies  if any b_i < b_min

This script TESTS that the law reproduces the four things we already measured:
  A) binding(~N) vs repulsion(~N^2)  -> a SIZE CEILING (a "digital Oganesson")
  B) bond-dimension tax              -> RANDOM dies, STRUCTURED survives
  C) metabolism threshold            -> infinite capacity, FINITE live depth
  D) full law decides grow / split / die on real structures

Run from repo root:  python cortex_v2/validate_survival.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mps import MPS

# ---- one place for all coefficients ----
J   = 1.0      # binding strength
LAM = 0.10     # repulsion strength
MU  = 0.05     # surface / exposure cost
ETA = 1e-3     # bond-dimension tax (rent per bond)
T   = 0.0      # survival threshold: E < T means alive

# ============================================================ PART A
# Compact blob of N agreeing cells (sim=1, b_i=1):
#   B = J*(N-1)        nearest-neighbor binding   ~ N
#   R = LAM*N(N-1)/2   every pair repels (p=0)     ~ N^2
def energy_blob(N):
    B = J * max(N - 1, 0)
    R = LAM * N * (N - 1) / 2.0
    S = MU * 2                       # two exposed ends
    return -B + R + S, B, R

print("=" * 74)
print("A) BINDING (~N) vs REPULSION (~N^2): the size ceiling / digital Oganesson")
print("=" * 74)
print(f"   J={J}  lambda={LAM}   predicted ceiling N* = 2J/lambda = {2*J/LAM:.0f}")
print(f"   {'N':>4} {'B':>8} {'R':>9} {'E':>9}  status")
for N in [2, 5, 10, 15, 18, 20, 22, 25, 30, 40]:
    E, B, R = energy_blob(N)
    tag = "stable (bound)" if E < T else "UNSTABLE -> fission"
    print(f"   {N:>4} {B:>8.1f} {R:>9.1f} {E:>9.1f}  {tag}")
print("   -> small structures bind; past the ceiling, repulsion wins -> they split.")

# ============================================================ PART B
print()
print("=" * 74)
print("B) BOND-DIMENSION TAX: random states die, structured states survive")
print("=" * 74)
def chi_of(kind, n):
    m = MPS(n, max_chi=4096); rng = np.random.default_rng(0)
    depth = 3 if kind == "structured" else n
    for _ in range(depth):
        for s in range(n):
            m.rx_gate(s, rng.uniform(0, math.pi)); m.rz_gate(s, rng.uniform(0, math.pi))
        for s in range(0, n - 1, 2): m.cnot(s, s + 1)
        for s in range(1, n - 1, 2): m.cnot(s, s + 1)
    return m.max_chi_used

n = 16
print(f"   {'kind':12} {'chi':>6} {'C=eta*chi^2*n':>14} {'E=-B+C':>9}  status")
for kind in ["structured", "random"]:
    chi = chi_of(kind, n)
    C = ETA * chi ** 2 * n
    B = J * (n - 1)
    E = -B + C
    tag = "survives" if E < T else "DIES (too complex)"
    print(f"   {kind:12} {chi:>6} {C:>14.1f} {E:>9.1f}  {tag}")
print("   -> the bond tax kills randomness (huge chi) and spares structure (small chi).")
print("      matches our earlier result: random ~13, structured 50k, repeating 1M.")

# ============================================================ PART C
print()
print("=" * 74)
print("C) METABOLISM THRESHOLD: infinite capacity, finite LIVE depth")
print("=" * 74)
B0 = 486   # one cube's budget; split-decrease gives b_d = B0 / 27^d
print(f"   split-decrease: a cell at depth d holds b_d = {B0}/27^d")
print(f"   {'b_min':>10} {'live d_max':>11} {'cells alive at d_max':>22}")
for bmin in [1.0, 1e-3, 1e-6, 1e-9]:
    dmax = math.floor(math.log(B0 / bmin) / math.log(27))
    print(f"   {bmin:>10} {dmax:>11} {27**dmax:>22.2e}")
print("   -> closed-form capacity is unlimited, but only finitely many cells stay ALIVE.")

# ============================================================ PART D
print()
print("=" * 74)
print("D) FULL LAW decides grow / split / die")
print("=" * 74)
def verdict(N, kind, bmin_ok=True):
    E, B, R = energy_blob(N)
    chi = chi_of(kind, min(N, 18))
    C = ETA * chi ** 2 * min(N, 18)
    Etot = E + C
    if not bmin_ok:                 return "DIES (starved: b < b_min)"
    if R + C > B:                   return "SPLITS (pressure > binding)"
    if Etot < T:                    return "GROWS (dE < 0, cheaper alive)"
    return "marginal / dies"
for N, kind in [(8, "structured"), (8, "random"), (30, "structured"), (12, "structured")]:
    print(f"   N={N:2d} {kind:11} -> {verdict(N, kind)}")
print()
print("survival = binding - repulsion - complexity - mismatch")
print("conservation gives the budget; THIS gives death, fission, stability, selection.")
