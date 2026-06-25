"""
validate_sites.py — empirically probe how many MPS "sites" cortex_v2 can produce,
and whether the result is FAITHFUL (a real computation) or merely cheap because the
state is trivial. Tests the retired "500-site computer" claim against the code.
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mps import MPS, ghz

def fmt_bytes(b):
    for u in ["B","KB","MB","GB"]:
        if b < 1024: return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}TB"

print("="*70)
print("TEST 1: GHZ (low-entanglement) — how big can n go cheaply & correctly?")
print("="*70)
for n in [50, 200, 500, 1000, 2000]:
    t0 = time.time()
    m = ghz(n)
    dt = time.time() - t0
    bd = m.bond_dims()
    bits = m.measure_all(np.random.default_rng(0))
    correct = len(set(bits)) == 1  # GHZ must collapse to all-0 or all-1
    print(f"n={n:5d}  max_bond={max(bd):2d}  mem={fmt_bytes(m.memory_bytes()):>8}  "
          f"build={dt:5.2f}s  GHZ_valid={correct}")

print()
print("="*70)
print("TEST 2: HIGH-entanglement random circuit — is a generic 'computation' faithful?")
print("="*70)
print("max_chi caps the bond dimension. If trunc_error > 0, the simulator is")
print("DISCARDING amplitude-like information -> NOT a faithful n-site computer.")
print()
rng = np.random.default_rng(1)
for n in [12, 20, 30, 50]:
    m = MPS(n, max_chi=64)           # default cap from the codebase
    # entangling sweep: random 1q rotations + adjacent CNOT ladder, several layers
    for layer in range(n):           # depth ~ n -> drives volume-law entanglement
        for s in range(n):
            m.rx_gate(s, rng.uniform(0, math.pi))
            m.rz_gate(s, rng.uniform(0, math.pi))
        for s in range(0, n-1, 2):
            m.cnot(s, s+1)
        for s in range(1, n-1, 2):
            m.cnot(s, s+1)
    bd = m.bond_dims()
    print(f"n={n:3d}  max_bond_used={m.max_chi_used:3d}/cap{m.max_chi}  "
          f"prune_events={m.prune_events:4d}  trunc_error={m.trunc_error:.3e}  "
          f"faithful={m.trunc_error < 1e-9}")

print()
print("="*70)
print("TEST 3: exact-statevector cross-check (the only ground truth we have)")
print("="*70)
def dense(m):
    psi = m.tensors[0]
    for i in range(1, m.n):
        psi = np.tensordot(psi, m.tensors[i], axes=([-1],[0]))
    return psi.reshape(-1)
for n in [6, 8, 10]:
    m = MPS(n, max_chi=2**n)  # no truncation
    rng2 = np.random.default_rng(7)
    # reference statevector
    ref = np.zeros([2]*n, dtype=complex); ref[(0,)*n] = 1.0
    def apply1(psi, U, s):
        return np.moveaxis(np.tensordot(U, psi, axes=([1],[s])), 0, s)
    for s in range(n):
        th = rng2.uniform(0, math.pi)
        Ux = np.array([[math.cos(th/2), -1j*math.sin(th/2)],[-1j*math.sin(th/2), math.cos(th/2)]], complex)
        m.rx_gate(s, th); ref = apply1(ref, Ux, s)
    # one CNOT ladder
    CN = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex).reshape(2,2,2,2)
    for s in range(n-1):
        m.cnot(s, s+1)
        ref = np.moveaxis(np.tensordot(CN, ref, axes=([2,3],[s,s+1])), [0,1], [s,s+1])
    err = float(np.max(np.abs(dense(m) - ref.reshape(-1))))
    print(f"n={n:2d}  max_err_vs_exact={err:.2e}  exact={'YES' if err<1e-9 else 'NO'}")
