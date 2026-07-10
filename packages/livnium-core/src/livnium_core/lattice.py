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
import math
from dataclasses import dataclass
from typing import Dict, Tuple

from .rotations import Matrix, apply

Vec = Tuple[int, int, int]
Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]
Field = Dict[Vec, float]


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


# --------------------------------------------------------------------------- #
# multipole moments — the *distribution* of charge, beside its conserved total
# --------------------------------------------------------------------------- #
# `symbolic_weight_total` is the conserved ledger: one scalar, the total meaning
# mass inside the shell. It is the MONOPOLE — the only quantity a Gauss-style
# flux read gives you for free, and the only one that survives *every* move
# (rotation AND class-preserving rearrangement, since a permutation cannot change
# a sum). It says how much is inside, never where.
#
# To read *arrangement* off the boundary you need the higher moments of an actual
# charge field q: cell -> weight. Their transformation law is the honest part:
#
#   monopole   M    = Σ qᵢ                      (scalar)   rot-INVARIANT, perm-INVARIANT
#   dipole     D    = Σ qᵢ rᵢ                   (vector)   rot-COVARIANT  : D → R·D
#   quadrupole Q_jk = Σ qᵢ r_ij r_ik            (tensor)   rot-COVARIANT  : Q → R·Q·Rᵀ
#
# Under the 24 rigid rotations the moments rotate rigidly, so their rotation
# invariants — |D|, tr(Q), ‖Q‖_F, det(Q) — are preserved. Under a FACE TURN
# (moves.py) the interior is rearranged into positions NOT related by a global
# rotation, so M is still conserved but D and Q genuinely change: that is the
# precise code-level statement of "the total is conserved, the arrangement is
# not." Verified in tests/test_moments.py.


def sw_field(N: int) -> Field:
    """The geometric symbolic-weight distribution as a charge field.

    This field is centrally symmetric (exposure(c) == exposure(-c)) and cubically
    symmetric, so its dipole is exactly the zero vector and its quadrupole is an
    isotropic diagonal tensor — a useful fixed-point sanity check.
    """
    return {c: float(SW(c, N)) for c in _iter_cells(N)}


def monopole(field: Field) -> float:
    """0th moment: total charge / meaning mass. Invariant under any permutation."""
    return float(sum(field.values()))


def dipole(field: Field) -> Vec3:
    """1st moment D = Σ qᵢ rᵢ — center/direction of mass. Covariant: D → R·D."""
    dx = dy = dz = 0.0
    for (x, y, z), q in field.items():
        dx += q * x
        dy += q * y
        dz += q * z
    return (dx, dy, dz)


def quadrupole(field: Field) -> Mat3:
    """2nd moment Q_jk = Σ qᵢ r_ij r_ik — spread/polarity. Covariant: Q → R·Q·Rᵀ."""
    q = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for (x, y, z), w in field.items():
        r = (x, y, z)
        for j in range(3):
            for k in range(3):
                q[j][k] += w * r[j] * r[k]
    return (tuple(q[0]), tuple(q[1]), tuple(q[2]))  # type: ignore[return-value]


def vec_norm(v: Vec3) -> float:
    """Euclidean magnitude of a 3-vector (e.g. |dipole|)."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def mat_trace(m: Mat3) -> float:
    """tr(Q) = Σ qᵢ |rᵢ|² — radial spread. Rotation-invariant."""
    return m[0][0] + m[1][1] + m[2][2]


def mat_frobenius(m: Mat3) -> float:
    """‖Q‖_F. Rotation-invariant (orthogonal R preserves the Frobenius norm)."""
    return math.sqrt(sum(m[j][k] ** 2 for j in range(3) for k in range(3)))


def mat_det(m: Mat3) -> float:
    """det(Q). Rotation-invariant (det(R·Q·Rᵀ) = det Q for orthogonal R)."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def rotate_field(field: Field, rot: Matrix) -> Field:
    """Transport a charge field by a rigid rotation: the charge that was at c
    lands at R·c. Lets the covariance laws be exercised directly (see tests)."""
    return {apply(rot, c): q for c, q in field.items()}


@dataclass(frozen=True)
class MultipoleSignature:
    """A charge field collapsed to its low-order moments + rotation invariants.

    The invariants (dipole_mag, quad_trace, quad_frobenius, quad_det) are the
    boundary-readable, frame-independent summary: two fields related by any of the
    24 rigid rotations share them exactly. They are the static counterpart to the
    path Signature in ping.py — flux side vs. circulation side.
    """

    monopole: float
    dipole: Vec3
    quadrupole: Mat3
    dipole_mag: float
    quad_trace: float
    quad_frobenius: float
    quad_det: float


def multipole_signature(field: Field) -> MultipoleSignature:
    """Reduce a charge field to monopole, dipole, quadrupole and their invariants."""
    d = dipole(field)
    q = quadrupole(field)
    return MultipoleSignature(
        monopole=monopole(field),
        dipole=d,
        quadrupole=q,
        dipole_mag=vec_norm(d),
        quad_trace=mat_trace(q),
        quad_frobenius=mat_frobenius(q),
        quad_det=mat_det(q),
    )
