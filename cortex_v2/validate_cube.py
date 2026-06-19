"""
validate_cube.py — drive the n-site MPS with the ACTUAL cube, not random gates.

Where validate_nocap.py used rng.uniform() to make up rotations, here every
single-qubit gate comes from the Livnium 3x3x3 cube via its SU(2) lift:

    word -> word_to_rotation(word) -> rot in 1..23
    cube.apply(rot)                -> alpha signal (governor)
    SU2[rot]                       -> the 2x2 gate actually applied to a site

So the cube's 24-element rotation group is the gate source, and ALPHA tunes
the MPS governor. Entanglement still needs CNOTs (the cube alone is unitary
single-site geometry; it cannot create entanglement by itself).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mps import MPS
from lattice import Lattice, SU2, ALPHA, word_to_rotation, NAMES

def fmt_bytes(b):
    for u in ["B","KB","MB","GB","TB"]:
        if b < 1024: return f"{b:.1f}{u}"
        b /= 1024
    return f"{b:.1f}PB"

# a deterministic word stream to drive the cube
WORDS = ("livnium cube observer om entropy lattice rotation collapse basin "
         "attractor geometry symbol weight conserve spinor bridge").split()

def run(n, max_chi):
    m = MPS(n, max_chi=max_chi, s_max=None)
    cube = Lattice()
    wi = 0
    rots_used = set()
    for layer in range(n):
        # single-qubit layer: every gate is a cube rotation's SU(2) lift
        for s in range(n):
            word = WORDS[wi % len(WORDS)] + str(wi); wi += 1
            rot = word_to_rotation(word)
            alpha = cube.apply(rot)        # advance cube; get governor signal
            m.alpha = alpha                # feed the cube's signal into the MPS
            m.apply_1q(s, SU2[rot])        # apply the cube gate to this site
            rots_used.add(rot)
        # entangling layers (CNOT ladders) — required for real entanglement
        for s in range(0, n-1, 2):
            m.cnot(s, s+1)
        for s in range(1, n-1, 2):
            m.cnot(s, s+1)
    return m, cube, rots_used

print("="*82)
print("CUBE-DRIVEN MPS — single-qubit gates sourced from the Livnium cube's SU(2) lift")
print("="*82)
print(f"cube facts: 24 rotations, alpha range [{ALPHA.min():.3f}, {ALPHA.max():.3f}]")
print()

print("--- capped (max_chi=64), high entanglement ---")
for n in [12, 16, 20, 30, 50]:
    t0 = time.time(); m, cube, rots = run(n, 64); dt = time.time()-t0
    print(f"n={n:3d}  cube_orient={cube.orient:2d}  rots_used={len(rots):2d}/23  "
          f"max_bond={m.max_chi_used:3d}/64  trunc_err={m.trunc_error:.2e}  "
          f"faithful={abs(m.trunc_error)<1e-9}  cube_bijection={cube.is_bijection()}  "
          f"SW={cube.total_sw()}  t={dt:5.2f}s")

print()
print("--- uncapped (max_chi=2^n), exact but exponential ---")
for n in [12, 16, 18, 20]:
    t0 = time.time(); m, cube, rots = run(n, 2**n); dt = time.time()-t0
    print(f"n={n:3d}  max_bond={m.max_chi_used:5d}  mem={fmt_bytes(m.memory_bytes()):>8}  "
          f"trunc_err={m.trunc_error:.1e}  faithful={abs(m.trunc_error)<1e-9}  t={dt:5.2f}s")
