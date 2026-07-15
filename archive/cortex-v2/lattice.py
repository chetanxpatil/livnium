"""
cortex_v2.lattice — the Livnium 3x3x3 lattice, reduced to what it is.

KEY FACT (proved in selftest T13): the alpha signal of a rotation is
INDEPENDENT of the lattice's current orientation. A rotation R maps the
multiset of (position -> new position) pairs identically regardless of
which symbol sits where, because rotations permute the lattice
bijectively. Therefore:

    - the lattice state is ONE integer (orientation index, 0..23)
    - alpha is a table of 24 precomputed floats
    - per-word cost is O(1): two array lookups

This replaces the v1 dict-of-27-coords + per-call polarity loop
(~1,200 lines) with ~120 lines and a >1000x speedup on the hot path.
All v1 invariants (R^4=I, 24 orientations, class counts, sum SW = 486,
bijection) hold by construction and are verified in selftest.py.
"""

from __future__ import annotations

import hashlib

import numpy as np

# ---------------------------------------------------------------------------
# Coordinates and static invariants
# ---------------------------------------------------------------------------

COORDS = np.array(
    [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)],
    dtype=np.int8,
)
_INDEX = {tuple(c): i for i, c in enumerate(COORDS)}

EXPOSURE = np.abs(COORDS).sum(axis=1).astype(np.int8)  # f in {0,1,2,3}
SW = 9 * EXPOSURE  # symbolic weight
TOTAL_SW = int(SW.sum())  # 486, constant
CLASS_COUNTS = {f: int((EXPOSURE == f).sum()) for f in range(4)}  # {0:1,1:6,2:12,3:8}


def _perm(fn) -> np.ndarray:
    """Permutation p with p[i] = index of fn(COORDS[i])."""
    return np.array([_INDEX[fn(tuple(c))] for c in COORDS], dtype=np.int8)


# Generators: 90-degree rotations about z, x, y (right-handed)
_GEN_MAPS = {
    "Z": lambda c: (-c[1], c[0], c[2]),
    "X": lambda c: (c[0], -c[2], c[1]),
    "Y": lambda c: (c[2], c[1], -c[0]),
}
_GEN_MATS = {
    "Z": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
    "X": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float),
    "Y": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float),
}

# ---------------------------------------------------------------------------
# BFS over the rotation group: 24 elements, each with perm + 3x3 matrix
# ---------------------------------------------------------------------------


def _build_group():
    gen_perms = {k: _perm(f) for k, f in _GEN_MAPS.items()}
    ident = np.arange(27, dtype=np.int8)
    seen = {tuple(ident): 0}
    perms = [ident]
    mats = [np.eye(3)]
    names = ["I"]
    frontier = [0]
    while frontier:
        nxt = []
        for idx in frontier:
            for k in "ZXY":
                # apply generator AFTER existing element: i -> gen[perm[i]]
                p = gen_perms[k][perms[idx]]
                key = tuple(p)
                if key not in seen:
                    seen[key] = len(perms)
                    perms.append(p)
                    mats.append(_GEN_MATS[k] @ mats[idx])
                    names.append((k + names[idx]).replace("I", "") or "I")
                    nxt.append(len(perms) - 1)
        frontier = nxt
    assert len(perms) == 24, f"expected 24 rotations, got {len(perms)}"
    return np.stack(perms), np.stack(mats), names


PERMS, MATS, NAMES = _build_group()

# Composition table: COMPOSE[a, b] = index of (apply a, then b)
_key_to_idx = {tuple(PERMS[i]): i for i in range(24)}
COMPOSE = np.array(
    [[_key_to_idx[tuple(PERMS[b][PERMS[a]])] for b in range(24)] for a in range(24)],
    dtype=np.int8,
)
INVERSE = np.array(
    [next(b for b in range(24) if COMPOSE[a, b] == 0) for a in range(24)],
    dtype=np.int8,
)

# ---------------------------------------------------------------------------
# Alpha: mean |cos(motion, toward-observer)| over moved cells — 24 constants
# ---------------------------------------------------------------------------


def _alpha_of(perm: np.ndarray) -> float:
    old = COORDS.astype(float)
    new = COORDS[perm].astype(float)
    moved = (perm != np.arange(27)) & (np.abs(old).sum(axis=1) > 0)
    if not moved.any():
        return 0.0
    m = new[moved] - old[moved]
    o = -old[moved]
    cos = (m * o).sum(axis=1) / (np.linalg.norm(m, axis=1) * np.linalg.norm(o, axis=1))
    return float(np.abs(cos).mean())


ALPHA = np.array([_alpha_of(PERMS[i]) for i in range(24)])

# ---------------------------------------------------------------------------
# SU(2) lift (for the state-vector bridge) — axis/angle from each 3x3 matrix
# ---------------------------------------------------------------------------

_PAULI = np.array(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=complex,
)


def _axis_angle(M: np.ndarray):
    theta = float(np.arccos(np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0)))
    if theta < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(theta - np.pi) < 1e-9:
        # axis from M = 2 n n^T - I
        n = np.sqrt(np.clip((np.diag(M) + 1.0) / 2.0, 0.0, 1.0))
        # fix signs using off-diagonals
        if n[0] > 1e-8:
            n[1] = np.copysign(n[1], M[0, 1])
            n[2] = np.copysign(n[2], M[0, 2])
        elif n[1] > 1e-8:
            n[2] = np.copysign(n[2], M[1, 2])
        return n / np.linalg.norm(n), float(np.pi)
    n = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    n = n / (2.0 * np.sin(theta))
    return n / np.linalg.norm(n), theta


def _su2(n: np.ndarray, theta: float) -> np.ndarray:
    ns = n[0] * _PAULI[0] + n[1] * _PAULI[1] + n[2] * _PAULI[2]
    return np.cos(theta / 2) * np.eye(2, dtype=complex) - 1j * np.sin(theta / 2) * ns


SU2 = np.stack([_su2(*_axis_angle(MATS[i])) for i in range(24)])

# ---------------------------------------------------------------------------
# Runtime lattice: one integer of state
# ---------------------------------------------------------------------------


class Lattice:
    """Orientation-tracking lattice. apply() returns the alpha signal."""

    __slots__ = ("orient",)

    def __init__(self):
        self.orient = 0

    def apply(self, rot: int) -> float:
        self.orient = int(COMPOSE[self.orient, rot])
        return float(ALPHA[rot])

    # explicit state, for verification only
    def occupancy(self) -> np.ndarray:
        """occupancy[i] = symbol now at COORDS[i]."""
        p = PERMS[self.orient]
        occ = np.empty(27, dtype=np.int8)
        occ[p] = np.arange(27, dtype=np.int8)
        return occ

    def total_sw(self) -> int:
        return TOTAL_SW  # invariant by construction: rotations permute coords

    def is_bijection(self) -> bool:
        return len(set(self.occupancy().tolist())) == 27


def word_to_rotation(word: str) -> int:
    """Deterministic word -> rotation index in 1..23 (never identity).

    Uses a stable cryptographic digest rather than Python's built-in hash():
    str hashing is salted per-process (PYTHONHASHSEED), so hash(word) is only
    stable within a single interpreter run and differs across processes/launches.
    Note: the mapping is arbitrary, same as v1 — only the GloVe bridge is
    semantic."""
    digest = hashlib.md5(word.encode("utf-8")).hexdigest()
    return int(digest, 16) % 23 + 1  # never identity, so alpha > 0
