"""
validate_representation.py — what do amplitudes LOOK like, and can we just change
the representation (basis) to make the explosion go away?

Answer: you can change the basis all you want (that's exactly what MPS/compression
does — hunt for a basis where the state is simple). BUT there is an INVARIANT the
state carries across any cut — its entanglement (Schmidt rank / entropy) — and NO
change of representation can lower it. Cheap states are cheap in some basis;
genuinely entangled states are expensive in EVERY basis.

Run from repo root:  python cortex_v2/validate_representation.py
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def schmidt_rank(psi2):
    """rank across the A|B cut of a 2-qubit state (4-vector reshaped 2x2)."""
    s = np.linalg.svd(psi2.reshape(2, 2), compute_uv=False)
    return int((s > 1e-9).sum()), np.round(s, 3)

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

print("=" * 74)
print("WHAT AMPLITUDES LOOK LIKE (2 qubits -> 4 amplitudes, one per outcome)")
print("=" * 74)
states = {
    "|00>            (definite)": np.array([1, 0, 0, 0], float),
    "product (+)(+)  (separable)": np.kron([1, 1], [1, 1]) / 2.0,
    "Bell  (00+11)/v2 (entangled)": np.array([1, 0, 0, 1], float) / np.sqrt(2),
}
for name, psi in states.items():
    r, sv = schmidt_rank(psi)
    print(f"  {name}")
    print(f"     amplitudes [00,01,10,11] = {psi}")
    print(f"     Schmidt rank across cut  = {r}   (1 = separable, 2 = entangled)")
    print()

print("=" * 74)
print("CHANGE THE REPRESENTATION (apply a local basis rotation) — what's invariant?")
print("=" * 74)
for name, psi in states.items():
    # rotate qubit A's basis by a Hadamard (a different 'representation space')
    rotated = np.kron(H, np.eye(2)) @ psi
    r0, _ = schmidt_rank(psi)
    r1, _ = schmidt_rank(rotated)
    print(f"  {name}")
    print(f"     amplitudes BEFORE rotation = {psi}")
    print(f"     amplitudes AFTER  rotation = {rotated}")
    print(f"     Schmidt rank: before={r0}  after={r1}   {'INVARIANT' if r0==r1 else 'CHANGED'}")
    print()

print("=" * 74)
print("Can ANY basis make the Bell state separable? try 200 random local bases")
print("=" * 74)
bell = np.array([1, 0, 0, 1], float) / np.sqrt(2)
def random_unitary(rng):
    z = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / np.sqrt(2)
    q, r = np.linalg.qr(z); return q @ np.diag(np.diag(r) / np.abs(np.diag(r)))
rng = np.random.default_rng(0)
min_rank = 99
for _ in range(200):
    U, V = random_unitary(rng), random_unitary(rng)
    out = np.kron(U, V) @ bell
    r, _ = schmidt_rank(out)
    min_rank = min(min_rank, r)
print(f"  smallest Schmidt rank found over 200 local-basis changes: {min_rank}")
print(f"  -> the Bell state stays rank {min_rank} in EVERY local basis. You cannot")
print("     'represent away' real entanglement. That invariant IS the wall.")
print()
print("READING: representation is free to choose, and a good choice makes")
print("low-entanglement data tiny (your 500x compression). But entanglement entropy")
print("is basis-invariant, so no representation shrinks a truly entangled state.")
