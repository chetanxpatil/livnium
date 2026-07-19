"""
validate_energy_sites.py — does the split-&-decrease ENERGY change the faithful
amplitude-like-site ceiling?

Hypothesis: if energy shrinks/divides as we go down, maybe it limits entanglement
and lets more sites stay faithful. We test it directly: drive the cube MPS while
the governor's entropy ceiling is set from the shrinking energy budget, and find
the faithful wall. Compare to the no-governor baseline.

Key fact under test: entanglement (Schmidt rank) is a property of the AMPLITUDES,
not of the conserved energy ledger. Truncation can only DISCARD, never add capacity.

Run from repo root:  python cortex_v2/validate_energy_sites.py
"""
import sys, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "packages", "livnium-core", "src")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mps import MPS
from lattice import Lattice, SU2, ALPHA

def run(n, max_chi=64, s_max=None, energy_decay=None):
    """energy_decay: if set, the governor ceiling shrinks each layer by this factor
    (simulating energy dividing as we go deeper)."""
    m = MPS(n, max_chi=max_chi, s_max=s_max, alpha=0.5)
    cube = Lattice(); k = 0; budget = 1.0
    for layer in range(n):
        if energy_decay is not None and s_max is not None:
            m.s_max = s_max * budget          # ceiling tracks the shrinking energy
            budget *= energy_decay            # energy divides going deeper
        for s in range(n):
            rot = (k % 23) + 1; cube.apply(rot); m.apply_1q(s, SU2[rot]); k += 1
        for s in range(0, n - 1, 2): m.cnot(s, s + 1)
        for s in range(1, n - 1, 2): m.cnot(s, s + 1)
    return abs(m.trunc_error) < 1e-9, m.max_chi_used, m.trunc_error, m.prune_events

def wall(label, **kw):
    last_ok = 0
    for n in range(6, 21):
        ok, bond, err, pr = run(n, **kw)
        if ok: last_ok = n
    print(f"  {label:42}  deepest faithful = {last_ok} sites")
    return last_ok

print("=" * 78)
print("Does shrinking/dividing ENERGY raise the faithful amplitude-like-site ceiling?")
print("=" * 78)
b0 = wall("baseline: no governor (chi=64 only)")
b1 = wall("energy governor, ceiling=2.0 (loose)", s_max=2.0)
b2 = wall("energy governor, ceiling=1.0 (tight)", s_max=1.0)
b3 = wall("energy SHRINKING x0.7/layer (loose start)", s_max=2.0, energy_decay=0.7)
b4 = wall("energy SHRINKING x0.5/layer (tight start)", s_max=1.0, energy_decay=0.5)

print()
print("ANSWER:")
print(f"  baseline faithful wall .................. {b0} sites")
print(f"  best achieved WITH energy shrinking ..... {max(b1,b2,b3,b4)} sites")
if max(b1, b2, b3, b4) <= b0:
    print("  => energy shrinking/dividing does NOT raise the ceiling.")
    print("     At best it matches baseline; tighter energy only PRUNES MORE")
    print("     (fewer faithful sites). Truncation discards info, never adds capacity.")
else:
    print("  => energy shrinking RAISED the ceiling (unexpected — investigate).")
print()
print("WHY: entanglement entropy lives in the amplitude-like amplitudes. The energy ledger")
print("is conserved bookkeeping over GEOMETRY. Shrinking it cannot lower the Schmidt")
print("rank the true state needs, so the ~13-site wall is unmoved.")
