"""
Livnium → MPS Bridge Test Suite
================================

Tests run in order from most fundamental to most integrated.
Nothing is assumed. Every claim is verified by output.

T1  Rotation R⁴ = I  (all 3 generators)
T2  Class counts correct for 3×3×3
T3  ΣSW = 486 and invariant under all 24 rotations
T4  Bijection preserved after every rotation
T5  All 24 rotations are distinct
T6  Polarity range: cos(θ) ∈ [−1, +1]
T7  Polarity geometry: known analytic cases
T8  Identity rotation → polarity = 0 (nothing moves)
T9  Bridge α signal: high-polarity vs low-polarity rotations distinguishable
T10 Governor responds to α: tight S_max + high α → fewer prunes than low α
T11 Conservation laws hold after 100 random rotation sequences
T12 LivniumGovernedCircuit end-to-end: GHZ + Livnium polarity signal
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from mps_simulator import MPSSimulator
from polarity_governor import SemanticPolarityGovernor
from livnium_to_tensor import (
    LivniumLattice,
    boundary_exposure,
    symbolic_weight,
    generate_all_24_rotations,
    livnium_polarity_signal,
    LivniumGovernedCircuit,
    COORDS_3x3x3,
)

PASS = "✅"
FAIL = "❌"

failures = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS} {name}  {detail}")
    else:
        print(f"  {FAIL} {name}  {detail}")
        failures.append(name)


# ─────────────────────────────────────────────────────────────────────────────
# T1  R⁴ = I
# ─────────────────────────────────────────────────────────────────────────────

def test_r4_identity():
    print("\n── T1  R⁴ = I ──────────────────────────────────────────")
    for name, fn in [("Rz90", LivniumLattice.rotate_z90),
                     ("Rx90", LivniumLattice.rotate_x90),
                     ("Ry90", LivniumLattice.rotate_y90)]:
        lat = LivniumLattice()
        original = dict(lat.state)
        for _ in range(4):
            fn(lat)
        check(f"{name}⁴ = I", lat.state == original,
              f"state restored after 4× {name}")


# ─────────────────────────────────────────────────────────────────────────────
# T2  Class counts
# ─────────────────────────────────────────────────────────────────────────────

def test_class_counts():
    print("\n── T2  Class counts for 3×3×3 ──────────────────────────")
    lat = LivniumLattice()
    cc = lat.class_counts()
    check("Core   (f=0) count = 1",  cc[0] == 1,  f"got {cc[0]}")
    check("Center (f=1) count = 6",  cc[1] == 6,  f"got {cc[1]}")
    check("Edge   (f=2) count = 12", cc[2] == 12, f"got {cc[2]}")
    check("Corner (f=3) count = 8",  cc[3] == 8,  f"got {cc[3]}")
    check("Total = 27",              sum(cc.values()) == 27)


# ─────────────────────────────────────────────────────────────────────────────
# T3  ΣSW invariant
# ─────────────────────────────────────────────────────────────────────────────

def test_sw_invariant():
    print("\n── T3  ΣSW = 486, invariant under all 24 rotations ─────")
    lat = LivniumLattice()
    sw0 = lat.total_sw()
    check("Initial ΣSW = 486", sw0 == 486, f"got {sw0}")

    rotations = generate_all_24_rotations()
    all_ok = True
    for label, fn in rotations:
        test_lat = LivniumLattice()
        fn(test_lat)
        sw = test_lat.total_sw()
        if sw != 486:
            all_ok = False
            print(f"  {FAIL} ΣSW after {label}: {sw}")
            break
    check(f"ΣSW = 486 after all 24 rotations", all_ok)


# ─────────────────────────────────────────────────────────────────────────────
# T4  Bijection preserved
# ─────────────────────────────────────────────────────────────────────────────

def test_bijection():
    print("\n── T4  Bijection preserved after all 24 rotations ──────")
    rotations = generate_all_24_rotations()
    all_ok = True
    for label, fn in rotations:
        lat = LivniumLattice()
        fn(lat)
        if not lat.is_bijection():
            all_ok = False
            print(f"  {FAIL} bijection violated after {label}")
            break
    check("All 24 rotations preserve bijection", all_ok)


# ─────────────────────────────────────────────────────────────────────────────
# T5  All 24 rotations are distinct
# ─────────────────────────────────────────────────────────────────────────────

def test_24_distinct():
    print("\n── T5  All 24 rotations are distinct ───────────────────")
    rotations = generate_all_24_rotations()
    check("Generator produces exactly 24", len(rotations) == 24,
          f"got {len(rotations)}")

    def state_key(lat):
        return tuple(sorted(lat.state.items()))

    states = set()
    for _, fn in rotations:
        lat = LivniumLattice()
        fn(lat)
        states.add(state_key(lat))
    check("All 24 produce distinct states", len(states) == 24,
          f"got {len(states)} distinct")


# ─────────────────────────────────────────────────────────────────────────────
# T6  Polarity range
# ─────────────────────────────────────────────────────────────────────────────

def test_polarity_range():
    print("\n── T6  Polarity ∈ [−1, +1] for all rotations ───────────")
    rotations = generate_all_24_rotations()
    all_in_range = True
    bad = []
    for label, fn in rotations:
        lat = LivniumLattice()
        result = lat.apply_rotation_with_polarity(fn)
        for p in result["polarities"]:
            if not -1.0 - 1e-9 <= p <= 1.0 + 1e-9:
                all_in_range = False
                bad.append((label, p))
    check("All polarity values ∈ [−1, +1]", all_in_range,
          f"{len(bad)} out-of-range" if bad else "")


# ─────────────────────────────────────────────────────────────────────────────
# T7  Polarity geometry: known analytic cases
# ─────────────────────────────────────────────────────────────────────────────

def test_polarity_geometry():
    """
    Verify analytic polarity cases from the spec geometry.

    Case 1: Symbol at (1,0,0), moves to (0,1,0)  (Rz90 on a face-center)
        motion     = (0,1,0)−(1,0,0) = (−1,1,0)
        observer   = (0,0,0)−(1,0,0) = (−1,0,0)
        cos θ      = (1+0+0) / (√2 × 1) = 1/√2 ≈ 0.707

    Case 2: Symbol at (1,0,0), moves to (−1,0,0)  (180° rotation)
        motion     = (−2,0,0)
        observer   = (−1,0,0)
        cos θ      = 2 / (2 × 1) = +1.0  (moving directly toward origin and past it)

    Case 3: Symbol at (0,0,1), moves to (0,1,0)  (Rx90)
        motion     = (0,1,−1)
        observer   = (0,0,−1)
        cos θ      = (0+0+1) / (√2 × 1) = 1/√2 ≈ 0.707
    """
    print("\n── T7  Polarity geometry: known analytic cases ──────────")
    lat = LivniumLattice()

    # Case 1: (1,0,0) → (0,1,0)
    p1 = lat.semantic_polarity((1,0,0), (0,1,0))
    expected1 = 1.0 / np.sqrt(2)
    check(f"(1,0,0)→(0,1,0): cos θ = 1/√2 ≈ {expected1:.4f}",
          abs(p1 - expected1) < 1e-9, f"got {p1:.6f}")

    # Case 2: (1,0,0) → (−1,0,0)
    p2 = lat.semantic_polarity((1,0,0), (-1,0,0))
    expected2 = 1.0
    check(f"(1,0,0)→(−1,0,0): cos θ = +1.0 (moves through origin)",
          abs(p2 - expected2) < 1e-9, f"got {p2:.6f}")

    # Case 3: (0,0,1) → (0,1,0)
    p3 = lat.semantic_polarity((0,0,1), (0,1,0))
    expected3 = 1.0 / np.sqrt(2)
    check(f"(0,0,1)→(0,1,0): cos θ = 1/√2 ≈ {expected3:.4f}",
          abs(p3 - expected3) < 1e-9, f"got {p3:.6f}")

    # Case 4: origin symbol (0,0,0) → any move → polarity = 0 (no observer dir)
    p4 = lat.semantic_polarity((0,0,0), (1,0,0))
    check(f"(0,0,0)→(1,0,0): cos θ = 0.0 (origin has no observer direction)",
          abs(p4) < 1e-9, f"got {p4:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# T8  Identity rotation → polarity = 0
# ─────────────────────────────────────────────────────────────────────────────

def test_identity_polarity():
    print("\n── T8  Identity rotation → nothing moves, α = 0 ────────")
    lat = LivniumLattice()

    def identity(l): pass
    result = lat.apply_rotation_with_polarity(identity)
    check("n_moved = 0 for identity", result["n_moved"] == 0,
          f"got {result['n_moved']}")
    check("mean_abs = 0.0 for identity", result["mean_abs"] == 0.0,
          f"got {result['mean_abs']}")


# ─────────────────────────────────────────────────────────────────────────────
# T9  Bridge α signal is non-trivially distributed across rotations
# ─────────────────────────────────────────────────────────────────────────────

def test_alpha_distribution():
    print("\n── T9  Bridge α is distributed (not all equal) ─────────")
    rotations = generate_all_24_rotations()
    alphas = []
    for label, fn in rotations:
        lat = LivniumLattice()
        alpha = livnium_polarity_signal(lat, fn)
        alphas.append(alpha)

    alphas = np.array(alphas)
    print(f"  α values across 24 rotations:")
    print(f"    min  = {alphas.min():.4f}")
    print(f"    max  = {alphas.max():.4f}")
    print(f"    mean = {alphas.mean():.4f}")
    print(f"    std  = {alphas.std():.4f}")

    check("α values not all identical (std > 0)",
          alphas.std() > 1e-6, f"std={alphas.std():.6f}")
    check("α ∈ [0, 1] for all rotations",
          bool(np.all(alphas >= -1e-9) and np.all(alphas <= 1.0 + 1e-9)))
    check("At least two distinct α values",
          len(set(np.round(alphas, 6))) >= 2)


# ─────────────────────────────────────────────────────────────────────────────
# T10  Governor responds: high α → fewer prunes on structured circuit
# ─────────────────────────────────────────────────────────────────────────────

def test_governor_responds_to_alpha():
    """
    Build GHZ-20 under a tight S_max = 0.9×log(2).
    Run twice: once with high α (from a real Livnium rotation),
               once with α = 0.0 (no polarity reward).
    Expect: high α → fewer prune events.
    """
    print("\n── T10  Governor responds to α signal ───────────────────")
    n   = 20
    s_m = np.log(2) * 0.9   # tight: below GHZ entropy, so pruning is forced without α

    # High α: use the rotation with maximum polarity
    rotations = generate_all_24_rotations()
    lat_tmp = LivniumLattice()
    alpha_vals = []
    for _, fn in rotations:
        l = LivniumLattice()
        alpha_vals.append(livnium_polarity_signal(l, fn))
    max_alpha = max(alpha_vals)

    # Governor with high α
    sim_hi = MPSSimulator(n_qubits=n, max_bond_dim=64)
    gov_hi = SemanticPolarityGovernor(sim_hi, S_max=s_m, alpha=max_alpha, verbose=False)
    gov_hi.hadamard(0)
    for i in range(n - 1):
        gov_hi.cnot(i, i + 1)

    # Governor with α = 0
    sim_lo = MPSSimulator(n_qubits=n, max_bond_dim=64)
    gov_lo = SemanticPolarityGovernor(sim_lo, S_max=s_m, alpha=0.0, verbose=False)
    gov_lo.hadamard(0)
    for i in range(n - 1):
        gov_lo.cnot(i, i + 1)

    hi_prunes = len(gov_hi.pruning_log)
    lo_prunes = len(gov_lo.pruning_log)
    print(f"  Max Livnium α = {max_alpha:.4f}")
    print(f"  Prunes (α={max_alpha:.2f}): {hi_prunes}")
    print(f"  Prunes (α=0.00):            {lo_prunes}")
    check("High α → fewer prunes than α=0",
          hi_prunes < lo_prunes,
          f"{hi_prunes} < {lo_prunes}")


# ─────────────────────────────────────────────────────────────────────────────
# T11  Conservation after 100 random rotation sequences
# ─────────────────────────────────────────────────────────────────────────────

def test_conservation_stress():
    print("\n── T11  Conservation: 100 random rotation chains ────────")
    rng = np.random.default_rng(0)
    fns = [
        LivniumLattice.rotate_x90,
        LivniumLattice.rotate_y90,
        LivniumLattice.rotate_z90,
    ]

    all_valid = True
    failure_detail = ""
    for trial in range(100):
        lat = LivniumLattice()
        n_steps = int(rng.integers(5, 50))
        for _ in range(n_steps):
            rng.choice(fns)(lat)
        ok, msg = lat.is_valid()
        if not ok:
            all_valid = False
            failure_detail = f"trial {trial}, {n_steps} steps: {msg}"
            break

    check("All 100 stress chains pass conservation", all_valid, failure_detail)


# ─────────────────────────────────────────────────────────────────────────────
# T12  End-to-end: LivniumGovernedCircuit with GHZ
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end():
    print("\n── T12  End-to-end: LivniumGovernedCircuit + GHZ-15 ────")
    n = 15
    shots = 200

    lat = LivniumLattice()
    circ = LivniumGovernedCircuit(n_qubits=n, s_max=np.log(2))

    # Apply a Livnium rotation to set the polarity signal
    alpha = circ.apply_livnium_rotation(lat, lambda l: l.rotate_z90())
    print(f"  Livnium Rz90 α = {alpha:.4f}")

    # Build GHZ under polarity-governed compression
    circ.hadamard(0)
    for i in range(n - 1):
        circ.cnot(i, i + 1)

    legal = 0
    for _ in range(shots):
        r = circ.measure_all()
        bits = "".join(map(str, r))
        if bits in ("0" * n, "1" * n):
            legal += 1

    ok, msg = lat.is_valid()
    check(f"Lattice still valid after circuit",     ok, msg)
    check(f"≥ 80% legal GHZ outcomes ({legal}/{shots})", legal >= shots * 0.8,
          f"{legal}/{shots} = {legal/shots:.2%}")
    check(f"Governor prune log consistent",
          isinstance(circ.gov.pruning_log, list))

    circ.summary()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("LIVNIUM → MPS BRIDGE  —  TEST SUITE")
    print("Testing spec §1–§2 claims + bridge behaviour")
    print()

    test_r4_identity()
    test_class_counts()
    test_sw_invariant()
    test_bijection()
    test_24_distinct()
    test_polarity_range()
    test_polarity_geometry()
    test_identity_polarity()
    test_alpha_distribution()
    test_governor_responds_to_alpha()
    test_conservation_stress()
    test_end_to_end()

    print()
    print("=" * 55)
    if failures:
        print(f"FAILED  {len(failures)} test(s):")
        for f in failures:
            print(f"  {FAIL}  {f}")
    else:
        print(f"ALL TESTS PASSED  (12 / 12)")
    print("=" * 55)
    print()
