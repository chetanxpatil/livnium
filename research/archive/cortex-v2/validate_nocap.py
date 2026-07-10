"""
validate_nocap.py — remove the bond-dimension cap and watch what happens.
With max_chi huge, the MPS is forced to keep ALL singular values -> it becomes
an exact statevector simulator. Faithfulness returns, but cost is exponential.
"""
import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mps import MPS

def fmt_bytes(b):
    for u in ["B","KB","MB","GB","TB"]:
        if b < 1024: return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}PB"

print("="*78)
print("HIGH-ENTANGLEMENT random circuit, CAP REMOVED (max_chi = 2^n, no truncation)")
print("="*78)
rng = np.random.default_rng(1)
for n in [12, 16, 18, 20, 22, 24]:
    cap = 2**n  # effectively unbounded: max possible Schmidt rank
    t0 = time.time()
    m = MPS(n, max_chi=cap)
    try:
        for layer in range(n):
            for s in range(n):
                m.rx_gate(s, rng.uniform(0, math.pi))
                m.rz_gate(s, rng.uniform(0, math.pi))
            for s in range(0, n-1, 2):
                m.cnot(s, s+1)
            for s in range(1, n-1, 2):
                m.cnot(s, s+1)
        dt = time.time() - t0
        print(f"n={n:3d}  max_bond_used={m.max_chi_used:6d} (theoretical max {2**(n//2)})  "
              f"mem={fmt_bytes(m.memory_bytes()):>9}  trunc_err={m.trunc_error:.1e}  "
              f"faithful={abs(m.trunc_error)<1e-9}  time={dt:6.2f}s", flush=True)
    except MemoryError:
        print(f"n={n:3d}  *** MemoryError — exponential blowup, cannot allocate ***")
        break
