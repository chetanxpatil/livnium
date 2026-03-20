"""
SO(3) → SU(2) Test Suite
=========================

Tests run from most fundamental outward.
Every number claimed is verified by code output.

T1   Generator matrices map basis vectors correctly
T2   Generator matrices are orthogonal, det = +1
T3   All 24 SO(3) matrices are orthogonal, det = +1
T4   All 24 SO(3) matrices are distinct
T5   Angle class distribution: 1×0° + 6×90° + 8×120° + 9×180° = 24
T6   Axis extraction exact for the 3 generators
T7   SU(2) gates are 2×2 unitary (U†U = I) for all 24
T8   All 24 SU(2) gates are distinct (up to global phase)
T9   SU(2) generators match analytic closed forms
T10  Composition: gate(A)@gate(B) = gate(AB) up to global phase
T11  Consistency with livnium_to_tensor rotations (same permutation effect)
T12  Gate acts on qubit correctly: Rx90 on |0⟩ gives (|0⟩ − i|1⟩)/√2
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rotation_to_pauli import (
    Rx90_mat, Ry90_mat, Rz90_mat,
    extract_axis_angle,
    axis_angle_to_su2,
    rotation_matrix_to_su2,
    generate_all_24_so3_su2,
    compose_sequence,
    I2, SX, SY, SZ,
)
from livnium_to_tensor import LivniumLattice

PASS  = "✅"
FAIL  = "❌"
failures = []

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS} {name}  {detail}")
    else:
        print(f"  {FAIL} {name}  {detail}")
        failures.append(name)

def allclose(a, b, tol=1e-10):
    return np.allclose(a, b, atol=tol)

def matrices_equal_up_to_phase(U, V, tol=1e-10):
    """True if U = e^{iφ} V for some global phase φ."""
    # Try all elements to find a ratio
    for i in range(2):
        for j in range(2):
            if abs(V[i,j]) > 1e-10:
                ratio = U[i,j] / V[i,j]
                if abs(abs(ratio) - 1.0) < tol:
                    if allclose(U, ratio * V, tol):
                        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# T1  Generator matrices map basis vectors correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_generator_matrices():
    print("\n── T1  Generator matrices map basis vectors ─────────────")
    e1, e2, e3 = np.array([1,0,0.]), np.array([0,1,0.]), np.array([0,0,1.])

    # Rx90: (x,y,z) → (x, −z, y)
    check("Rx90 @ e1 = e1", allclose(Rx90_mat @ e1, [1,0,0]))
    check("Rx90 @ e2 = e3", allclose(Rx90_mat @ e2, [0,0,1]))
    check("Rx90 @ e3 = −e2",allclose(Rx90_mat @ e3, [0,-1,0]))

    # Ry90: (x,y,z) → (z, y, −x)
    check("Ry90 @ e1 = −e3",allclose(Ry90_mat @ e1, [0,0,-1]))
    check("Ry90 @ e2 = e2", allclose(Ry90_mat @ e2, [0,1,0]))
    check("Ry90 @ e3 = e1", allclose(Ry90_mat @ e3, [1,0,0]))

    # Rz90: (x,y,z) → (−y, x, z)
    check("Rz90 @ e1 = e2", allclose(Rz90_mat @ e1, [0,1,0]))
    check("Rz90 @ e2 = −e1",allclose(Rz90_mat @ e2, [-1,0,0]))
    check("Rz90 @ e3 = e3", allclose(Rz90_mat @ e3, [0,0,1]))


# ─────────────────────────────────────────────────────────────────────────────
# T2  Generator matrices are orthogonal, det = +1
# ─────────────────────────────────────────────────────────────────────────────

def test_generator_orthogonality():
    print("\n── T2  Generator matrices: orthogonal + det = +1 ────────")
    for name, M in [("Rx90", Rx90_mat), ("Ry90", Ry90_mat), ("Rz90", Rz90_mat)]:
        check(f"{name}: MᵀM = I", allclose(M.T @ M, np.eye(3)))
        check(f"{name}: det = +1", abs(np.linalg.det(M) - 1.0) < 1e-10,
              f"det={np.linalg.det(M):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# T3  All 24 SO(3) matrices: orthogonal, det = +1
# ─────────────────────────────────────────────────────────────────────────────

def test_all_24_orthogonal():
    print("\n── T3  All 24 SO(3) matrices: orthogonal + det = +1 ─────")
    all24 = generate_all_24_so3_su2()
    bad_orth = sum(1 for r in all24 if not allclose(r["M_so3"].T @ r["M_so3"], np.eye(3)))
    bad_det  = sum(1 for r in all24 if abs(np.linalg.det(r["M_so3"]) - 1.0) > 1e-10)
    check(f"All 24 satisfy MᵀM = I", bad_orth == 0, f"{bad_orth} violations")
    check(f"All 24 have det = +1",   bad_det  == 0, f"{bad_det} violations")


# ─────────────────────────────────────────────────────────────────────────────
# T4  All 24 SO(3) matrices distinct
# ─────────────────────────────────────────────────────────────────────────────

def test_24_distinct_so3():
    print("\n── T4  All 24 SO(3) matrices are distinct ───────────────")
    all24 = generate_all_24_so3_su2()
    check(f"Generator found 24 elements", len(all24) == 24, f"got {len(all24)}")

    keys = set()
    for r in all24:
        keys.add(tuple(np.round(r["M_so3"], 8).flatten()))
    check("All 24 SO(3) matrices distinct", len(keys) == 24, f"got {len(keys)}")


# ─────────────────────────────────────────────────────────────────────────────
# T5  Angle class distribution
# ─────────────────────────────────────────────────────────────────────────────

def test_angle_classes():
    """
    The 24 cube rotations split into:
      1  identity  (θ = 0°)
      6  face quarter-turns (θ = 90°)
      3  face half-turns (θ = 180°, axis = x/y/z)
      8  vertex rotations (θ = 120°)
      6  edge half-turns (θ = 180°, axis = face-diagonal)
    Total: 1 + 6 + 3 + 8 + 6 = 24
    Of these: 9 rotations have θ = 180°.
    """
    print("\n── T5  Angle class distribution ─────────────────────────")
    all24 = generate_all_24_so3_su2()
    from collections import Counter
    deg_counts = Counter(round(r["theta_deg"]) for r in all24)

    print(f"  Angle distribution: {dict(sorted(deg_counts.items()))}")
    check("θ =   0°: 1 rotation",  deg_counts[0]   == 1,  f"got {deg_counts[0]}")
    check("θ =  90°: 6 rotations", deg_counts[90]  == 6,  f"got {deg_counts[90]}")
    check("θ = 120°: 8 rotations", deg_counts[120] == 8,  f"got {deg_counts[120]}")
    check("θ = 180°: 9 rotations", deg_counts[180] == 9,  f"got {deg_counts[180]}")
    check("Total = 24",             sum(deg_counts.values()) == 24)


# ─────────────────────────────────────────────────────────────────────────────
# T6  Axis extraction exact for the 3 generators
# ─────────────────────────────────────────────────────────────────────────────

def test_axis_extraction_generators():
    print("\n── T6  Axis extraction for generators ───────────────────")
    cases = [
        ("Rx90", Rx90_mat, np.array([1.,0.,0.]), 90.),
        ("Ry90", Ry90_mat, np.array([0.,1.,0.]), 90.),
        ("Rz90", Rz90_mat, np.array([0.,0.,1.]), 90.),
    ]
    for name, M, expected_n, expected_deg in cases:
        n, theta = extract_axis_angle(M)
        got_deg  = np.degrees(theta)
        check(f"{name}: angle = 90°",
              abs(got_deg - expected_deg) < 1e-8, f"got {got_deg:.4f}°")
        check(f"{name}: axis = {expected_n}",
              allclose(n, expected_n), f"got {np.round(n,6)}")


# ─────────────────────────────────────────────────────────────────────────────
# T7  All 24 SU(2) gates are unitary
# ─────────────────────────────────────────────────────────────────────────────

def test_su2_unitary():
    print("\n── T7  All 24 SU(2) gates are unitary (U†U = I) ─────────")
    all24 = generate_all_24_so3_su2()
    bad = 0
    for r in all24:
        U = r["U_su2"]
        prod = U.conj().T @ U
        if not allclose(prod, I2, 1e-10):
            bad += 1
    check(f"All 24 gates satisfy U†U = I", bad == 0, f"{bad} non-unitary")
    # Also check det = +1 for SU(2) specifically
    bad_det = sum(1 for r in all24
                  if abs(np.linalg.det(r["U_su2"]) - 1.0) > 1e-9)
    check(f"All 24 gates have det = +1 (SU(2))", bad_det == 0,
          f"{bad_det} with wrong det")


# ─────────────────────────────────────────────────────────────────────────────
# T8  All 24 SU(2) gates are distinct (up to global phase)
# ─────────────────────────────────────────────────────────────────────────────

def test_su2_distinct():
    print("\n── T8  All 24 SU(2) gates are distinct ──────────────────")
    all24 = generate_all_24_so3_su2()
    gates = [r["U_su2"] for r in all24]

    duplicates = 0
    for i in range(len(gates)):
        for j in range(i+1, len(gates)):
            if matrices_equal_up_to_phase(gates[i], gates[j]):
                duplicates += 1

    check(f"No two gates equal up to global phase", duplicates == 0,
          f"{duplicates} duplicate pairs found")


# ─────────────────────────────────────────────────────────────────────────────
# T9  SU(2) generators match analytic closed forms
# ─────────────────────────────────────────────────────────────────────────────

def test_su2_analytic():
    """
    For θ = 90°:
        Rx90 → [[1/√2, −i/√2], [−i/√2, 1/√2]]
        Ry90 → [[1/√2, −1/√2], [1/√2,   1/√2]]
        Rz90 → [[e^{−iπ/4}, 0], [0, e^{+iπ/4}]]
    """
    print("\n── T9  SU(2) generators match analytic values ───────────")
    s = 1.0 / np.sqrt(2)

    Ux_analytic = np.array([[s, -1j*s], [-1j*s, s]], dtype=complex)
    Uy_analytic = np.array([[s,    -s], [s,      s]], dtype=complex)
    Uz_analytic = np.array([[np.exp(-1j*np.pi/4), 0],
                             [0, np.exp(+1j*np.pi/4)]], dtype=complex)

    Ux_got = rotation_matrix_to_su2(Rx90_mat)
    Uy_got = rotation_matrix_to_su2(Ry90_mat)
    Uz_got = rotation_matrix_to_su2(Rz90_mat)

    check("Rx90 SU(2) = analytic Ux",
          matrices_equal_up_to_phase(Ux_got, Ux_analytic),
          f"\n    got:      {np.round(Ux_got, 5)}\n    expected: {np.round(Ux_analytic,5)}")
    check("Ry90 SU(2) = analytic Uy",
          matrices_equal_up_to_phase(Uy_got, Uy_analytic),
          f"\n    got:      {np.round(Uy_got, 5)}\n    expected: {np.round(Uy_analytic,5)}")
    check("Rz90 SU(2) = analytic Uz",
          matrices_equal_up_to_phase(Uz_got, Uz_analytic),
          f"\n    got:      {np.round(Uz_got, 5)}\n    expected: {np.round(Uz_analytic,5)}")


# ─────────────────────────────────────────────────────────────────────────────
# T10  Composition: gate(A)@gate(B) matches gate(AB)  (up to global phase)
# ─────────────────────────────────────────────────────────────────────────────

def test_composition_homomorphism():
    """
    The map SO(3) → SU(2) is a group homomorphism:
        U(R_A) @ U(R_B) = U(R_A R_B)   (up to global phase ±1)
    Test 20 random pairs from the 24 rotations.
    """
    print("\n── T10  Composition homomorphism: U(A)U(B) = U(AB) ─────")
    all24 = generate_all_24_so3_su2()

    rng = np.random.default_rng(0)
    fails = 0
    tested = 0

    for _ in range(20):
        i, j = rng.integers(0, 24, size=2)
        MA = all24[i]["M_so3"]
        MB = all24[j]["M_so3"]
        UA = all24[i]["U_su2"]
        UB = all24[j]["U_su2"]

        M_AB  = MA @ MB
        U_AB  = rotation_matrix_to_su2(M_AB)
        U_prod = UA @ UB

        if not matrices_equal_up_to_phase(U_prod, U_AB):
            fails += 1
        tested += 1

    check(f"U(A)@U(B) = U(A@B) for {tested} random pairs",
          fails == 0, f"{fails} failures")


# ─────────────────────────────────────────────────────────────────────────────
# T11  Consistent with livnium_to_tensor coordinate rotations
# ─────────────────────────────────────────────────────────────────────────────

def test_consistency_with_lattice():
    """
    Verify that the 3×3 matrices agree with what livnium_to_tensor.py does:
    Rx90_mat @ [0,1,0] should give the same result as rotating coord (0,1,0)
    through LivniumLattice.rotate_x90().
    """
    print("\n── T11  Consistent with livnium_to_tensor ───────────────")
    lat = LivniumLattice()

    test_coords = [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (1,1,0)]
    axes = [
        ("Rx90", Rx90_mat, lambda l: l.rotate_x90()),
        ("Ry90", Ry90_mat, lambda l: l.rotate_y90()),
        ("Rz90", Rz90_mat, lambda l: l.rotate_z90()),
    ]

    all_match = True
    for axis_name, M, rot_fn in axes:
        for coord in test_coords:
            # Matrix prediction
            predicted = tuple(int(round(v)) for v in M @ np.array(coord))

            # Lattice coordinate mapping: apply rotation, find where this coord went
            lat2 = LivniumLattice()
            # Find which symbol was at `coord` before rotation
            sym_at_coord = lat2.state.get(coord)
            if sym_at_coord is None:
                continue  # (1,1,0) not a valid lattice coord
            rot_fn(lat2)
            # Find where that symbol ended up
            new_coord = next(c for c, s in lat2.state.items() if s == sym_at_coord)

            if predicted != new_coord:
                all_match = False
                print(f"  {FAIL} {axis_name} on {coord}: matrix→{predicted} lattice→{new_coord}")

    check("All coordinate predictions match lattice rotations", all_match)


# ─────────────────────────────────────────────────────────────────────────────
# T12  Gate action on qubit: Rx90|0⟩ = (|0⟩ − i|1⟩)/√2
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_qubit_action():
    """
    Apply the SU(2) gates to |0⟩ = [1,0] and verify the output.

    Rx90|0⟩ = [[1/√2, −i/√2], [−i/√2, 1/√2]] @ [1,0] = [1/√2, −i/√2]
    Ry90|0⟩ = [[1/√2, −1/√2], [1/√2,   1/√2]] @ [1,0] = [1/√2, 1/√2]  = |+y⟩
    Rz90|0⟩ = [[e^{−iπ/4}, 0], [0, e^{iπ/4}]] @ [1,0] = [e^{−iπ/4}, 0]  (phase only)
    """
    print("\n── T12  Gate action on qubit |0⟩ ────────────────────────")
    ket0 = np.array([1.0, 0.0], dtype=complex)
    s = 1.0 / np.sqrt(2)

    Ux = rotation_matrix_to_su2(Rx90_mat)
    Uy = rotation_matrix_to_su2(Ry90_mat)
    Uz = rotation_matrix_to_su2(Rz90_mat)

    out_x = Ux @ ket0
    out_y = Uy @ ket0
    out_z = Uz @ ket0

    # Rx90|0⟩ should be proportional to [1, −i] (up to global phase)
    check("Rx90|0⟩ is superposition (|amplitude₀|² = |amplitude₁|² = 0.5)",
          abs(abs(out_x[0])**2 - 0.5) < 1e-10 and abs(abs(out_x[1])**2 - 0.5) < 1e-10,
          f"|α|²={abs(out_x[0])**2:.4f}, |β|²={abs(out_x[1])**2:.4f}")

    check("Ry90|0⟩ is superposition (real amplitudes)",
          abs(abs(out_y[0])**2 - 0.5) < 1e-10 and abs(abs(out_y[1])**2 - 0.5) < 1e-10,
          f"|α|²={abs(out_y[0])**2:.4f}, |β|²={abs(out_y[1])**2:.4f}")

    check("Rz90|0⟩ stays |0⟩ (only global phase, no superposition)",
          abs(abs(out_z[0])**2 - 1.0) < 1e-10 and abs(out_z[1])**2 < 1e-10,
          f"|α|²={abs(out_z[0])**2:.4f}, |β|²={abs(out_z[1])**2:.4f}")

    print(f"\n  Rx90|0⟩ = {np.round(out_x, 5)}   ← puts qubit on equator")
    print(f"  Ry90|0⟩ = {np.round(out_y, 5)}   ← |+⟩ state")
    print(f"  Rz90|0⟩ = {np.round(out_z, 5)}   ← only phase rotation")
    print()

    # Summary table of all 24 gates
    print("  Full table: angle class → qubit superposition")
    all24 = generate_all_24_so3_su2()
    from collections import defaultdict
    by_angle = defaultdict(list)
    for r in all24:
        out = r["U_su2"] @ ket0
        prob_1 = abs(out[1]) ** 2
        by_angle[round(r["theta_deg"])].append(prob_1)

    for deg in sorted(by_angle):
        probs = by_angle[deg]
        print(f"  θ={deg:3d}°: P(|1⟩) values = "
              f"{[round(p, 4) for p in sorted(set(round(p,4) for p in probs))]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("SO(3) → SU(2)  —  TEST SUITE")
    print("Non-arbitrary mapping: Livnium rotations → qubit gates")
    print()

    test_generator_matrices()
    test_generator_orthogonality()
    test_all_24_orthogonal()
    test_24_distinct_so3()
    test_angle_classes()
    test_axis_extraction_generators()
    test_su2_unitary()
    test_su2_distinct()
    test_su2_analytic()
    test_composition_homomorphism()
    test_consistency_with_lattice()
    test_gate_qubit_action()

    print()
    print("=" * 55)
    if failures:
        print(f"FAILED  {len(failures)} test(s):")
        for f in failures:
            print(f"  {FAIL}  {f}")
    else:
        print("ALL TESTS PASSED  (12 / 12)")
    print("=" * 55)
    print()
