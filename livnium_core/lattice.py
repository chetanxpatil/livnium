"""
lattice.py — exposure classes, symbolic weight, and the conservation law.

For an odd N >= 3, the lattice is L_N = {-(N-1)/2,...,+(N-1)/2}^3 (N^3 cells).
A cell's *exposure* f is how many of its coordinates lie on the outer
boundary (+/-(N-1)/2), so f in {0,1,2,3}. The symbolic weight is SW = 9*f.

Closed forms (verified for N=3,5,7,9 in tests/test_lattice.py):

    core   cells = (N-2)^3        SW 0
    center cells = 6(N-2)^2       SW 9
    edge   cells = 12(N-2)        SW 18
    corner cells = 8             SW 27

    total symbolic weight  SW(N) = 54(N-2)^2 + 216(N-2) + 216
        SW(3)=486  SW(5)=1350  SW(7)=2646  SW(9)=4374

These quantities are invariant under the cube rotation group (see rotations.py),
which is what makes the system a conserved geometric state space.
"""

from __future__ import annotations

import itertools
from typing import Dict, Tuple


def exposure(coord: Tuple[int, int, int], N: int) -> int:
    """Boundary-exposure count f in {0,1,2,3} for a cell at `coord`."""
    half = (N - 1) // 2
    return sum(1 for c in coord if abs(c) == half)


def SW(coord: Tuple[int, int, int], N: int) -> int:
    """Symbolic weight of a cell: SW = 9 * exposure."""
    return 9 * exposure(coord, N)


def _iter_cells(N: int):
    half = (N - 1) // 2
    rng = range(-half, half + 1)
    return itertools.product(rng, rng, rng)


def class_counts(N: int) -> Dict[str, int]:
    """Return {'core','center','edge','corner'} counts via closed form."""
    if N < 3 or N % 2 == 0:
        raise ValueError("N must be an odd integer >= 3")
    return {
        "core": (N - 2) ** 3,
        "center": 6 * (N - 2) ** 2,
        "edge": 12 * (N - 2),
        "corner": 8,
    }


def symbolic_weight_total(N: int) -> int:
    """Closed-form total symbolic weight SW(N) = 54(N-2)^2 + 216(N-2) + 216."""
    if N < 3 or N % 2 == 0:
        raise ValueError("N must be an odd integer >= 3")
    return 54 * (N - 2) ** 2 + 216 * (N - 2) + 216


def brute_force_total(N: int) -> int:
    """Direct cell-by-cell sum of SW — used by tests to check the closed form."""
    return sum(SW(c, N) for c in _iter_cells(N))
