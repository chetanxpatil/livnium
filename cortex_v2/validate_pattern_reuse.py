"""
validate_pattern_reuse.py — testing the user's theory:

  "Real states aren't noise. Interactions are LIMITED and patterns REPEAT, so the
   amplitude table is reusable and should compress far beyond the random-noise wall."

This is essentially the ENTANGLEMENT AREA LAW. We test three regimes:

  A. LIMITED interaction (fixed shallow depth)   -> bond saturates -> huge n faithful
  B. REPEATING pattern (periodic / dimerized)    -> tiny constant bond at any n
  C. RANDOM deep (volume law = noise)            -> bond explodes -> the ~13 wall

If A and B stay faithful at large n with small bond, the theory is CONFIRMED:
the wall was an artifact of noise, not a law of nature.

Run from repo root:  python cortex_v2/validate_pattern_reuse.py
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mps import MPS

def fmt(b):
    for u in ["B","KB","MB","GB"]:
        if b < 1024: return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}TB"

# ---- A. limited interaction: FIXED shallow depth, grow n -------------------
def limited(n, depth=3, max_chi=64):
    rng = np.random.default_rng(42)
    m = MPS(n, max_chi=max_chi)
    for _ in range(depth):                      # depth fixed, NOT growing with n
        for s in range(n):
            m.rx_gate(s, rng.uniform(0, math.pi)); m.rz_gate(s, rng.uniform(0, math.pi))
        for s in range(0, n-1, 2): m.cnot(s, s+1)
        for s in range(1, n-1, 2): m.cnot(s, s+1)
    return m

# ---- B. repeating pattern: dimerized (entangle pairs, identical everywhere) -
def repeating(n, max_chi=64):
    m = MPS(n, max_chi=max_chi)
    for s in range(0, n-1, 2):                  # same Bell-pair pattern repeated
        m.hadamard(s); m.cnot(s, s+1)
    return m

# ---- C. random deep: volume law (noise) ------------------------------------
def random_deep(n, max_chi=64):
    rng = np.random.default_rng(1)
    m = MPS(n, max_chi=max_chi)
    for _ in range(n):                          # depth = n -> volume-law noise
        for s in range(n):
            m.rx_gate(s, rng.uniform(0, math.pi)); m.rz_gate(s, rng.uniform(0, math.pi))
        for s in range(0, n-1, 2): m.cnot(s, s+1)
        for s in range(1, n-1, 2): m.cnot(s, s+1)
    return m

print("="*86)
print("A. LIMITED interaction (fixed depth=3) — does it stay faithful as n grows huge?")
print("="*86)
for n in [16, 50, 100, 300, 800]:
    t0=time.time(); m=limited(n); dt=time.time()-t0
    print(f"  n={n:4d}  max_bond={m.max_chi_used:3d}/64  trunc_err={m.trunc_error:.1e}  "
          f"faithful={abs(m.trunc_error)<1e-9}  mem={fmt(m.memory_bytes()):>8}  t={dt:5.2f}s")

print()
print("="*86)
print("B. REPEATING pattern (dimerized Bell pairs) — bond at large n?")
print("="*86)
for n in [16, 100, 500, 2000]:
    t0=time.time(); m=repeating(n); dt=time.time()-t0
    print(f"  n={n:4d}  max_bond={m.max_chi_used:3d}/64  trunc_err={m.trunc_error:.1e}  "
          f"faithful={abs(m.trunc_error)<1e-9}  mem={fmt(m.memory_bytes()):>8}  t={dt:5.2f}s")

print()
print("="*86)
print("C. RANDOM deep (noise, volume law) — the worst case from before")
print("="*86)
for n in [12, 16, 20, 24]:
    t0=time.time(); m=random_deep(n); dt=time.time()-t0
    print(f"  n={n:4d}  max_bond={m.max_chi_used:3d}/64  trunc_err={m.trunc_err if hasattr(m,'trunc_err') else m.trunc_error:.1e}  "
          f"faithful={abs(m.trunc_error)<1e-9}  t={dt:5.2f}s")

print()
print("VERDICT: if A and B stay faithful at n=hundreds/thousands with small bond,")
print("the theory holds — LIMITED + REPEATING structure compresses, and the wall")
print("only appears for genuine noise (C). Reality lives in A and B.")
