"""
rotations.py — the cube rotation group (order 24), pure integer matrices.

The allowed dynamics are 90-degree rotations about the X, Y, Z axes and their
compositions. Generated, they form a group of exactly 24 elements isomorphic
to S4 (the rotation group of the cube). Each generator satisfies R^4 = I.

Every rotation is a reversible permutation of the lattice that preserves
exposure class, class counts, symbolic weight, and the symbol-coordinate
bijection. Verified in tests/test_rotations.py.
"""
from __future__ import annotations
from typing import List, Tuple

Matrix = Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]

I: Matrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
ROT_X: Matrix = ((1, 0, 0), (0, 0, -1), (0, 1, 0))   # 90 deg about X
ROT_Y: Matrix = ((0, 0, 1), (0, 1, 0), (-1, 0, 0))   # 90 deg about Y
ROT_Z: Matrix = ((0, -1, 0), (1, 0, 0), (0, 0, 1))   # 90 deg about Z


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def matpow(a: Matrix, p: int) -> Matrix:
    r: Matrix = I
    for _ in range(p):
        r = matmul(r, a)
    return r


def rotation_group() -> List[Matrix]:
    """Close {ROT_X, ROT_Y, ROT_Z} under composition; returns all 24 elements."""
    gens = [ROT_X, ROT_Y, ROT_Z]
    seen = {I}
    frontier = [I]
    while frontier:
        m = frontier.pop()
        for g in gens:
            p = matmul(g, m)
            if p not in seen:
                seen.add(p)
                frontier.append(p)
    return sorted(seen)


def apply(rot: Matrix, coord: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Apply a rotation matrix to an integer coordinate."""
    return tuple(sum(rot[i][k] * coord[k] for k in range(3)) for i in range(3))  # type: ignore
