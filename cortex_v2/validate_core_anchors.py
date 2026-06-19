"""
validate_core_anchors.py — actually WIRE Livnium Core into the collapse classifier.

The 68.92% NLI model classifies a sentence-pair vector by which of 3 learned anchors
(Entailment / Neutral / Contradiction) it collapses toward. Core was NOT used there.

Here we test: can the CUBE supply those 3 anchors? We build anchors directly from the
24 cube rotations (SW-weighted permuted-coordinate signature, a genuine Core object),
pick the most-separated triple, and compare class-separating power against:
  - random anchors (no structure)
  - ideal simplex anchors (theoretical best: pairwise cos = -1/2)

Then run nearest-anchor "collapse" classification on synthetic 3-class data at rising
noise, to see whether cube anchors form usable gravity wells.

Run from repo root:  python cortex_v2/validate_core_anchors.py
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lattice import PERMS, COORDS, SW   # genuine Livnium Core objects

def unit(v): return v / (np.linalg.norm(v) + 1e-12)

# ---- a cube ROTATION -> a vector (SW-weighted permuted coordinate signature) ----
def cube_anchor(rot):
    moved = COORDS[PERMS[rot]].astype(float)        # where each cell goes under rot
    sig = (SW[:, None] * moved).reshape(-1)          # weight by symbolic weight -> 81-dim
    return unit(sig)

CUBE = np.array([cube_anchor(r) for r in range(24)])  # 24 candidate anchors

def pairwise_max_cos(anchors):
    c = anchors @ anchors.T
    return max(c[i, j] for i in range(len(anchors)) for j in range(i+1, len(anchors)))

# ---- pick the most-separated triple of cube rotations (give Core its best shot) ----
best, best_trip = 9, None
for trip in itertools.combinations(range(24), 3):
    A = CUBE[list(trip)]
    m = pairwise_max_cos(A)
    if m < best: best, best_trip = m, trip
cube3 = CUBE[list(best_trip)]

rng = np.random.default_rng(0)
rand3 = np.array([unit(rng.standard_normal(81)) for _ in range(3)])
# ideal simplex in 81-dim: 3 vectors with pairwise cos = -1/2
simplex = np.zeros((3, 81)); simplex[0,0]=1
simplex[1,0]=-0.5; simplex[1,1]=np.sqrt(3)/2
simplex[2,0]=-0.5; simplex[2,1]=-np.sqrt(3)/2

print("="*70)
print("ANCHOR SEPARATION (lower max-pairwise-cosine = better-separated wells)")
print("="*70)
print(f"  cube (best triple {best_trip}):  max cos = {pairwise_max_cos(cube3):+.3f}")
print(f"  random triple:                    max cos = {pairwise_max_cos(rand3):+.3f}")
print(f"  ideal simplex (theoretical best): max cos = {pairwise_max_cos(simplex):+.3f}")

def classify_acc(anchors, noise, n=3000):
    """generate n points near a random class anchor + noise, classify by nearest."""
    correct = 0
    for _ in range(n):
        c = rng.integers(3)
        x = anchors[c] + noise * rng.standard_normal(anchors.shape[1])
        x = unit(x)
        pred = int(np.argmax(anchors @ x))
        correct += (pred == c)
    return 100*correct/n

print()
print("="*70)
print("COLLAPSE CLASSIFICATION accuracy vs noise (nearest-anchor on 3 classes)")
print("="*70)
print(f"  {'noise':>6} {'cube':>8} {'random':>8} {'simplex':>8}")
for noise in [0.3, 0.6, 1.0, 1.5, 2.0]:
    a = classify_acc(cube3, noise)
    b = classify_acc(rand3, noise)
    c = classify_acc(simplex, noise)
    print(f"  {noise:>6.1f} {a:>7.1f}% {b:>7.1f}% {c:>7.1f}%")
print("  (chance = 33.3%)")
print()
print("READING: if cube anchors track the simplex (and beat random), then Livnium")
print("Core CAN supply usable class wells — the cube plugs into the collapse model.")
