"""
test_moments.py — the multipole layer over a charge field.

Asserts the honest transformation law:
  - monopole (total) is invariant under EVERY permutation: rigid rotations and
    class-preserving face turns alike (a sum cannot change under a relabelling).
  - dipole/quadrupole are COVARIANT under the 24 rigid rotations, so their
    rotation invariants (|D|, tr Q, ‖Q‖_F, det Q) are preserved by rotation but
    NOT by face turns — the code-level meaning of "total conserved, arrangement
    not".
  - the geometric SW field has zero dipole and an isotropic quadrupole by symmetry.
"""

import itertools
import math
import random

from livnium_core.lattice import (
    dipole,
    mat_det,
    mat_frobenius,
    mat_trace,
    monopole,
    multipole_signature,
    quadrupole,
    rotate_field,
    sw_field,
    symbolic_weight_total,
    vec_norm,
)
from livnium_core.moves import face_permutation
from livnium_core.rotations import apply, rotation_group

CELLS_27 = list(itertools.product((-1, 0, 1), repeat=3))


def _random_field(seed: int = 0):
    rng = random.Random(seed)
    return {c: rng.uniform(-5.0, 5.0) for c in CELLS_27}


# --------------------------------------------------------------------------- #
# monopole: the conserved ledger
# --------------------------------------------------------------------------- #
def test_monopole_equals_total_sw_for_sw_field():
    assert math.isclose(monopole(sw_field(3)), float(symbolic_weight_total(3)))
    assert math.isclose(monopole(sw_field(5)), float(symbolic_weight_total(5)))


def test_monopole_invariant_under_all_rotations():
    f = _random_field(1)
    m0 = monopole(f)
    for R in rotation_group():
        assert math.isclose(monopole(rotate_field(f, R)), m0, rel_tol=1e-12)


def test_monopole_invariant_under_face_turns():
    """A face turn permutes charges among cells; the total cannot change."""
    f = _random_field(2)
    m0 = monopole(f)
    for face in ("U", "D", "R", "L", "F", "B"):
        p = face_permutation(face)
        turned = {c: f[p[c]] for c in CELLS_27}
        assert math.isclose(monopole(turned), m0, rel_tol=1e-12)


# --------------------------------------------------------------------------- #
# dipole / quadrupole: covariance and preserved invariants under rotation
# --------------------------------------------------------------------------- #
def test_dipole_is_rotation_covariant():
    """D(R·field) == R·D(field) exactly."""
    f = _random_field(3)
    D = dipole(f)
    for R in rotation_group():
        DR = dipole(rotate_field(f, R))
        RD = apply(R, D)  # apply handles float vectors fine
        assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(DR, RD))


def test_rotation_invariants_preserved_by_rotation():
    f = _random_field(4)
    s0 = multipole_signature(f)
    for R in rotation_group():
        s = multipole_signature(rotate_field(f, R))
        assert math.isclose(s.dipole_mag, s0.dipole_mag, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(s.quad_trace, s0.quad_trace, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(s.quad_frobenius, s0.quad_frobenius, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(s.quad_det, s0.quad_det, rel_tol=1e-6, abs_tol=1e-6)


def test_arrangement_changes_under_face_turn():
    """The discriminating claim: a face turn keeps the monopole but moves the
    dipole — total conserved, arrangement not. (At least one face must shift it.)"""
    f = _random_field(5)
    D0 = dipole(f)
    moved = False
    for face in ("U", "D", "R", "L", "F", "B"):
        p = face_permutation(face)
        turned = {c: f[p[c]] for c in CELLS_27}
        if vec_norm(tuple(a - b for a, b in zip(dipole(turned), D0))) > 1e-6:
            moved = True
    assert moved


# --------------------------------------------------------------------------- #
# symmetry fixed points of the geometric SW field
# --------------------------------------------------------------------------- #
def test_sw_field_dipole_is_zero():
    """exposure(c) == exposure(-c) makes the SW dipole vanish exactly."""
    D = dipole(sw_field(3))
    assert all(abs(c) < 1e-12 for c in D)
    assert all(abs(c) < 1e-12 for c in dipole(sw_field(5)))


def test_sw_field_quadrupole_is_isotropic_diagonal():
    """Cubic symmetry forces Q = (tr/3)·I: equal diagonal, zero off-diagonal."""
    Q = quadrupole(sw_field(3))
    diag = (Q[0][0], Q[1][1], Q[2][2])
    assert math.isclose(diag[0], diag[1]) and math.isclose(diag[1], diag[2])
    for j in range(3):
        for k in range(3):
            if j != k:
                assert abs(Q[j][k]) < 1e-12


def test_invariant_helpers_basic():
    f = sw_field(3)
    Q = quadrupole(f)
    assert math.isclose(mat_trace(Q), Q[0][0] + Q[1][1] + Q[2][2])
    assert mat_frobenius(Q) >= mat_trace(Q) / math.sqrt(3) - 1e-9
    # isotropic diagonal -> det = (tr/3)^3
    assert math.isclose(mat_det(Q), (mat_trace(Q) / 3) ** 3, rel_tol=1e-9)
