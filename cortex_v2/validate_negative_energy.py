"""
validate_negative_energy.py — "why can't the energy go negative? maybe that's the fix."

You're onto something real. A NON-NEGATIVE conserved quantity behaves like classical
probability: things can only pile up, never cancel. A SIGNED (or complex) quantity can
CANCEL — and cancellation (interference) is the whole reason quantum beats classical.

So negativity IS essential to capture amplitudes. BUT: the MPS already uses signed/
complex amplitudes. Negativity is necessary, not sufficient — it does not remove the
exponential cost. We demonstrate both halves.

Run from repo root:  python cortex_v2/validate_negative_energy.py
"""
import numpy as np

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)   # note the negative entry

print("=" * 76)
print("PART 1: why negativity matters — interference (Hadamard twice = identity)")
print("=" * 76)
psi = np.array([1.0, 0.0])                      # start in |0>
print(f"  start:        {psi}   (definitely 0)")
psi1 = H @ psi
print(f"  after H:      {np.round(psi1,3)}   amplitudes for 0 and 1, BOTH paths open")
psi2 = H @ psi1
print(f"  after H again:{np.round(psi2,3)}   -> back to |0>!  the |1> paths CANCELLED")
print()
print("  Now force energy NON-NEGATIVE (use |amplitude| like a classical count):")
clavg = np.abs(H) @ (np.abs(H) @ np.abs(psi))
print(f"  classical (no signs): {np.round(clavg,3)}   -> NO cancellation, wrong result")
print("  => negativity is REQUIRED: without it you lose interference entirely.")

print()
print("=" * 76)
print("PART 2: negativity is necessary but NOT sufficient — cost is still exponential")
print("=" * 76)
print("  The MPS tensors are already complex (signed). The wall at ~13 faithful sites")
print("  came from the NUMBER of independent amplitudes (Schmidt rank), not their sign.")
print()
# show: a maximally entangled state needs full rank REGARDLESS of sign
for n in [4, 8, 12, 16]:
    rank_needed = 2 ** (n // 2)          # Schmidt rank of a volume-law state
    print(f"  n={n:2d} sites: a fully entangled state needs bond dim {rank_needed:5d} "
          f"(signs don't reduce this)")
print()
print("READING:")
print("  * You are RIGHT that non-negative energy is the wrong model for amplitudes —")
print("    it cannot cancel, so it can never represent interference.")
print("  * Signed/complex values fix THAT (and the simulator already has them).")
print("  * They do NOT shrink the COUNT of amplitudes, so the exponential entanglement")
print("    wall remains. Negativity buys correctness of structure, not free capacity.")
