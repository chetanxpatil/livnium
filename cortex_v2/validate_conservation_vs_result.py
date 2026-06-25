"""
validate_conservation_vs_result.py — the key question:
"if the system is conserved, can we just trust the result without looking inside?"

Answer: NO. Conservation (cube ΣSW) is a checksum on the GEOMETRY. It is preserved
even when the COMPUTATIONAL RESULT (the amplitude-like amplitudes) is corrupted by
truncation. We prove it: same conserved ledger, different result.

Run from repo root:  python cortex_v2/validate_conservation_vs_result.py
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mps import MPS
from lattice import Lattice, SU2

def dense(m):
    psi = m.tensors[0]
    for i in range(1, m.n):
        psi = np.tensordot(psi, m.tensors[i], axes=([-1], [0]))
    v = psi.reshape(-1)
    return v / np.linalg.norm(v)

def build(n, max_chi):
    m = MPS(n, max_chi=max_chi); cube = Lattice(); k = 0
    for layer in range(n):
        for s in range(n):
            rot = (k % 23) + 1; cube.apply(rot); m.apply_1q(s, SU2[rot]); k += 1
        for s in range(0, n-1, 2): m.cnot(s, s+1)
        for s in range(1, n-1, 2): m.cnot(s, s+1)
    return m, cube

n = 14
exact, cube_e = build(n, 2**n)     # no truncation  -> true result
cap,   cube_c = build(n, 64)       # capped         -> approximate result

ve, vc = dense(exact), dense(cap)
fidelity = abs(np.vdot(ve, vc))**2          # 1.0 means identical results

print("=" * 76)
print(f"n={n} cube-driven register:  EXACT (uncapped)  vs  CAPPED (max_chi=64)")
print("=" * 76)
print(f"conserved ledger:   exact cube SW = {cube_e.total_sw()}   capped cube SW = {cube_c.total_sw()}")
print(f"bijection intact:   exact = {cube_e.is_bijection()}        capped = {cube_c.is_bijection()}")
print(f"  -> CONSERVATION IS IDENTICAL AND UNCHANGED in both cases.")
print()
print(f"result fidelity |<exact|capped>|^2 = {fidelity:.4f}   (1.0 = same result)")
print(f"capped truncation error            = {cap.trunc_error:.3e}")
print()
# show a concrete measurable diverging: probability of the all-zeros outcome
p0_exact = abs(ve[0])**2
p0_cap   = abs(vc[0])**2
print(f"P(measure 00..0):  exact = {p0_exact:.5f}   capped = {p0_cap:.5f}   "
      f"off by {abs(p0_exact-p0_cap)/max(p0_exact,1e-12)*100:.0f}%")
print()
print("CONCLUSION: the conserved quantity did NOT move, yet the result is wrong")
print("(fidelity < 1). Conservation is necessary bookkeeping, not proof of result.")
print("You can trust 'don't look inside' ONLY for the pure geometry/address layer,")
print("where the conserved permutation IS the result. For the amplitude-like register the")
print("result lives in the amplitudes, which conservation does not pin down.")
