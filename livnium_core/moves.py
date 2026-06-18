"""
moves.py — "allowed change": face turns that rearrange the nodes inside the cube.

Livnium Core's base dynamics are the 24 RIGID whole-cube rotations (rotations.py).
This module adds the richer move set you get by allowing the cube to *rearrange
its own nodes*: the Rubik's-cube FACE TURNS. A face turn is a rotation applied to
ONE layer only, leaving the rest fixed.

Key facts (all checked in tests/test_moves.py):
  - Every face turn is a valid permutation of the 27 cells, of order 4 (T^4 = I).
  - The core Om (0,0,0) and all 6 face-centers stay fixed.
  - Face turns are CLASS-PRESERVING: a corner can only land in a corner slot,
    an edge in an edge slot, a face-center in a face-center slot.
  - Therefore total symbolic weight ΣSW is invariant under ANY sequence of
    face turns (SW depends only on exposure class), and every move is reversible.

So even though face turns generate a vast rearrangement space (the Rubik's group,
~4.3e19 states) far larger than the 24 rigid rotations, the conservation law and
the core both survive. The shell structure ("the box around the core") fixes each
node's energy; rearrangement only shuffles which node sits where.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

Cell = Tuple[int, int, int]
_CELLS: List[Cell] = list(itertools.product((-1, 0, 1), repeat=3))

# face -> (axis, layer level)
FACES = {
    "U": ("y", 1),
    "D": ("y", -1),
    "R": ("x", 1),
    "L": ("x", -1),
    "F": ("z", 1),
    "B": ("z", -1),
}
_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _rot(c: Cell, axis: str) -> Cell:
    x, y, z = c
    if axis == "x":
        return (x, -z, y)
    if axis == "y":
        return (z, y, -x)
    return (-y, x, z)  # axis == "z"


def face_permutation(face: str) -> Dict[Cell, Cell]:
    """Return the cell->cell permutation for one 90 degree face turn."""
    axis, lvl = FACES[face]
    idx = _AXIS_IDX[axis]
    return {c: (_rot(c, axis) if c[idx] == lvl else c) for c in _CELLS}


def apply_sequence(state: Dict[Cell, Cell], moves: List[str]) -> Dict[Cell, Cell]:
    """Apply a sequence of face turns to a state mapping slot->token."""
    for m in moves:
        p = face_permutation(m)
        state = {c: state[p[c]] for c in _CELLS}
    return state


def solved_state() -> Dict[Cell, Cell]:
    """Each slot holds the token whose home is that slot."""
    return {c: c for c in _CELLS}
