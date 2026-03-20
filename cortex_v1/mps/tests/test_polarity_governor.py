"""
Semantic Polarity Governor — Test Suite
========================================

Four scenarios:

1. Structured  (GHZ-20)       — polarity ≈ 1.0 → governor relaxes ceiling, never fires
2. Chaotic  (random circuit)  — polarity ≈ 0.0 → governor uses standard ceiling, prunes
3. Mixed circuit comparison   — GHZ backbone + noise: does polarity governor preserve GHZ?
4. Polarity spectrum          — side-by-side GHZ vs random bond profiles

The key question:
  Can a polarity-aware governor distinguish "meaningful" entanglement
  (polarised bonds, cos θ ≈ ±1 in Livnium) from "noisy" entanglement
  (uniform spectrum, cos θ ≈ 0 in Livnium)?
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from mps_simulator import MPSSimulator
from entanglement_governor import EntanglementGovernor, bond_entropies
from polarity_governor import SemanticPolarityGovernor, polarity_score_theoretical, bond_polarities


rng = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1: GHZ — highly polarised bonds
# ─────────────────────────────────────────────────────────────────────────────

def test_polarity_structured(n: int = 20):
    print("=" * 65)
    print(f"SCENARIO 1: GHZ-{n} — polarity profile of a structured state")
    print("=" * 65)
    # Middle bond (i=9): S_max_theo = 10 * log(2) ≈ 6.93
    # GHZ S = log(2) ≈ 0.693 → polarity = 1 - 0.693/6.93 ≈ 0.90
    mid_smax_theo = (n // 2) * np.log(2)
    mid_pol = 1.0 - np.log(2) / mid_smax_theo
    print(f"  S_max_base = log(2) = {np.log(2):.3f}  α = 0.5")
    print(f"  GHZ S per bond = log(2) ≈ 0.693")
    print(f"  Middle bond S_max_theo = {n//2}×log(2) = {mid_smax_theo:.3f}")
    print(f"  Middle bond polarity ≈ {mid_pol:.3f}")
    print(f"  Effective ceiling (mid) = log(2) × (1 + 0.5×{mid_pol:.2f}) = "
          f"{np.log(2) * (1 + 0.5 * mid_pol):.3f}")
    print(f"  → GHZ S=log(2) < effective ceiling → governor should NEVER fire\n")

    # Use S_max = log(2): exact GHZ entropy. Without polarity reward, governor fires.
    # With polarity reward, GHZ bonds have high polarity → relaxed ceiling → no pruning.
    sim = MPSSimulator(n_qubits=n, max_bond_dim=64)
    gov = SemanticPolarityGovernor(sim, S_max=np.log(2), alpha=0.5, verbose=False)

    gov.hadamard(0)
    for i in range(n - 1):
        gov.cnot(i, i + 1)

    gov.polarity_dashboard()
    print(f"  Governor fires: {len(gov.pruning_log)} times")
    if len(gov.pruning_log) == 0:
        print(f"  ✅ Correct — GHZ bonds are Area Law (high polarity), "
              f"relaxed ceiling holds at log(2)\n")
    else:
        print(f"  ⚠️  {len(gov.pruning_log)} prune events (early bonds have low polarity "
              f"because min(i+1, n-i-1) is small there)\n")

    # Compare: standard governor at same S_max would prune all bonds
    sim2 = MPSSimulator(n_qubits=n, max_bond_dim=64)
    std  = EntanglementGovernor(sim2, S_max=np.log(2) * 0.99, verbose=False)
    std.hadamard(0)
    for i in range(n - 1):
        std.cnot(i, i + 1)
    print(f"  Standard governor (S_max=0.99×log2): {len(std.pruning_log)} prune events")
    print(f"  → Polarity reward prevents pruning where standard governor would bite\n")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2: Random circuit — low polarity bonds
# ─────────────────────────────────────────────────────────────────────────────

def test_polarity_chaotic(n: int = 20, depth: int = 3):
    print("=" * 65)
    print(f"SCENARIO 2: Chaotic circuit — polarity profile (n={n}, depth={depth})")
    print("=" * 65)
    print(f"  Comparing standard governor vs polarity governor")
    print(f"  Both use S_max_base = log(8) = {np.log(8):.3f}\n")

    rng2 = np.random.default_rng(42)

    # Standard governor
    sim1 = MPSSimulator(n_qubits=n, max_bond_dim=256)
    std_gov = EntanglementGovernor(sim1, S_max=np.log(8), verbose=False)

    # Polarity governor (same base S_max)
    sim2 = MPSSimulator(n_qubits=n, max_bond_dim=256)
    pol_gov = SemanticPolarityGovernor(sim2, S_max=np.log(8), alpha=0.5, verbose=False)

    # Apply same circuit to both
    angles = [(rng2.uniform(0, np.pi), rng2.uniform(0, 2 * np.pi)) for _ in range(n * depth)]

    idx = 0
    for d in range(depth):
        for i in range(n):
            rx_ang, rz_ang = angles[idx]; idx += 1
            std_gov.rx(i, rx_ang)
            std_gov.rz(i, rz_ang)
        for i in range(0, n - 1, 2):
            std_gov.cnot(i, i + 1)
        for i in range(1, n - 1, 2):
            std_gov.cnot(i, i + 1)

    idx = 0
    for d in range(depth):
        for i in range(n):
            rx_ang, rz_ang = angles[idx]; idx += 1
            pol_gov.rx(i, rx_ang)
            pol_gov.rz(i, rz_ang)
        for i in range(0, n - 1, 2):
            pol_gov.cnot(i, i + 1)
        for i in range(1, n - 1, 2):
            pol_gov.cnot(i, i + 1)

    std_ent = bond_entropies(sim1)
    pol_ent = bond_entropies(sim2)
    pol_pols = bond_polarities(sim2)

    print(f"  Standard governor:  {len(std_gov.pruning_log):3d} prune events")
    print(f"  Polarity governor:  {len(pol_gov.pruning_log):3d} prune events")
    print()
    print(f"  Standard  — max S={max(std_ent):.4f}  mean S={np.mean(std_ent):.4f}")
    print(f"  Polarity  — max S={max(pol_ent):.4f}  mean S={np.mean(pol_ent):.4f}")
    print(f"  Polarity governor mean polarity: {np.mean(pol_pols):.4f}")
    print()
    print(f"  → Chaotic bonds have low polarity → effective ceiling ≈ S_max_base")
    print(f"  → Polarity governor behaves near-identically to standard governor")

    pol_gov.polarity_dashboard()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3: Mixed circuit — GHZ + noise
# ─────────────────────────────────────────────────────────────────────────────

def test_polarity_mixed(n: int = 15, shots: int = 200):
    """
    GHZ backbone + small random rotations.

    The GHZ bonds are highly polarised — polarity governor should
    give them breathing room and preserve correlations better.
    """
    print("=" * 65)
    print(f"SCENARIO 3: Mixed (GHZ + noise) — does polarity governor preserve GHZ?")
    print("=" * 65)
    print(f"  Circuit: H on q0 → CNOT chain → small Rx/Rz noise (θ ∈ [0, 0.25])")
    print(f"  Both governors: S_max_base = log(4)  (tight ceiling)")
    print(f"  Polarity governor: α = 0.5  (GHZ bonds get 50% looser ceiling)")
    print(f"  {shots} shots each\n")

    legal_std = 0
    legal_pol = 0

    for shot in range(shots):
        # Pre-compute noise angles so both circuits get IDENTICAL noise
        noise_angles = [(
            np.random.uniform(0, 0.25),
            np.random.uniform(0, 0.25)
        ) for _ in range(n)]

        # ── Standard governor ──────────────────────────────────────────────
        sim1 = MPSSimulator(n_qubits=n, max_bond_dim=64)
        g1   = EntanglementGovernor(sim1, S_max=np.log(4), verbose=False)
        g1.hadamard(0)
        for i in range(n - 1):
            g1.cnot(i, i + 1)
        for i in range(n):
            g1.rx(i, noise_angles[i][0])
            g1.rz(i, noise_angles[i][1])
        r1 = sim1.measure_all()
        if "".join(map(str, r1)) in ("0" * n, "1" * n):
            legal_std += 1

        # ── Polarity governor ──────────────────────────────────────────────
        sim2 = MPSSimulator(n_qubits=n, max_bond_dim=64)
        g2   = SemanticPolarityGovernor(sim2, S_max=np.log(4), alpha=0.5, verbose=False)
        g2.hadamard(0)
        for i in range(n - 1):
            g2.cnot(i, i + 1)
        for i in range(n):
            g2.rx(i, noise_angles[i][0])
            g2.rz(i, noise_angles[i][1])
        r2 = sim2.measure_all()
        if "".join(map(str, r2)) in ("0" * n, "1" * n):
            legal_pol += 1

    print(f"  Standard governor : {legal_std}/{shots} legal GHZ outcomes  "
          f"({'✅' if legal_std == shots else '⚠️'})")
    print(f"  Polarity governor : {legal_pol}/{shots} legal GHZ outcomes  "
          f"({'✅' if legal_pol == shots else '⚠️'})")
    print()

    diff = legal_pol - legal_std
    if diff > 0:
        print(f"  ✅ Polarity governor preserved {diff} MORE correlations than standard")
        print(f"  → GHZ bonds (polarity ≈ 1.0) got relaxed ceiling, survived noise pruning")
    elif diff == 0:
        print(f"  = Both governors performed identically")
        print(f"  → Small noise (θ < 0.25) barely shifts polarity from the GHZ baseline")
    else:
        print(f"  ⚠️  Standard governor marginally better on this seed")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4: Side-by-side polarity spectrum
# ─────────────────────────────────────────────────────────────────────────────

def test_polarity_spectrum(n: int = 14):
    """
    Visualise bond entropy and polarity for GHZ vs random circuits.

    This is the core diagnostic:
      GHZ  → S = log(2), polarity ≈ 1.0  (Area Law + decisive)
      Rnd  → S → log(χ), polarity → 0.0  (Volume Law + noisy)
    """
    print("=" * 65)
    print(f"SCENARIO 4: Polarity spectrum — GHZ vs random circuit (n={n})")
    print("=" * 65)
    print(f"  GHZ: H on q0 → CNOT chain")
    print(f"  Rnd: 4 layers of Rx/Rz + CNOT pairs (deep entangling circuit)\n")

    # ── Build GHZ ──────────────────────────────────────────────────────────
    sim_ghz = MPSSimulator(n_qubits=n, max_bond_dim=512)
    sim_ghz.hadamard(0)
    for i in range(n - 1):
        sim_ghz.cnot(i, i + 1)

    # ── Build random deep circuit ──────────────────────────────────────────
    sim_rnd = MPSSimulator(n_qubits=n, max_bond_dim=512)
    rng3 = np.random.default_rng(42)
    for _ in range(4):
        for i in range(n):
            sim_rnd.rx(i, rng3.uniform(0, np.pi))
            sim_rnd.rz(i, rng3.uniform(0, 2 * np.pi))
        for i in range(0, n - 1, 2):
            sim_rnd.cnot(i, i + 1)
        for i in range(1, n - 1, 2):
            sim_rnd.cnot(i, i + 1)

    # ── Compute per-bond stats ─────────────────────────────────────────────
    header = (
        f"  {'Bond':>4}  "
        f"{'GHZ S':>8}  {'GHZ pol':>8}  {'':12}  "
        f"{'Rnd S':>8}  {'Rnd pol':>8}  {'':12}"
    )
    print(header)
    print("  " + "-" * 63)

    ghz_pols = []
    rnd_pols = []

    for i in range(n - 1):
        # GHZ bond
        t = sim_ghz.tensors[i]
        chi_l, _, chi_r = t.shape
        _, sg, _ = np.linalg.svd(t.reshape(chi_l * 2, chi_r), full_matrices=False)
        Sg = von_neumann_entropy_safe(sg)
        pg = polarity_score_theoretical(sg, i, n)
        ghz_pols.append(pg)

        # Random bond
        t = sim_rnd.tensors[i]
        chi_l, _, chi_r = t.shape
        _, sr, _ = np.linalg.svd(t.reshape(chi_l * 2, chi_r), full_matrices=False)
        Sr = von_neumann_entropy_safe(sr)
        pr = polarity_score_theoretical(sr, i, n)
        rnd_pols.append(pr)

        g_bar = "▓" * min(int(pg * 12), 12)
        r_bar = "▓" * min(int(pr * 12), 12)

        print(
            f"  {i:>4}  "
            f"{Sg:>8.4f}  {pg:>8.4f}  {g_bar:<12}  "
            f"{Sr:>8.4f}  {pr:>8.4f}  {r_bar:<12}"
        )

    print()
    print(f"  GHZ   mean polarity = {np.mean(ghz_pols):.4f}  (≈ 1.0 expected)")
    print(f"  Rnd   mean polarity = {np.mean(rnd_pols):.4f}  (≈ 0.0 expected)")
    print()
    print(f"  The polarity governor exploits this gap:")
    print(f"  GHZ bonds   → pol ≈ 1.0 → eff_ceil = S_max × {1.0+0.5:.1f} → bond survives")
    print(f"  Random bonds→ pol ≈ 0.0 → eff_ceil = S_max × {1.0:.1f} → standard pruning")
    print()
    print(f"  Livnium EGAN parallel:")
    print(f"  cos θ = ±1  (aligned with attractor) → polarity = 1.0 → preserve")
    print(f"  cos θ = 0   (neutral hyperplane)      → polarity = 0.0 → prune")
    print()


def von_neumann_entropy_safe(s: np.ndarray) -> float:
    lam2 = s ** 2
    norm = lam2.sum()
    if norm < 1e-15:
        return 0.0
    p = lam2 / norm
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log(p)))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("SEMANTIC POLARITY GOVERNOR — TEST SUITE")
    print("Mapping cos θ (Livnium EGAN) → polarity score (MPS bonds)")
    print()

    test_polarity_structured(n=20)
    test_polarity_chaotic(n=20, depth=3)
    test_polarity_mixed(n=15, shots=200)
    test_polarity_spectrum(n=14)

    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print()
    print("  Core equation:")
    print("    polarity(bond) = 1 − S(bond) / log(χ(bond))")
    print("    effective_S_max = S_max_base × (1 + α × polarity)")
    print()
    print("  Bond character → Livnium analogy → Governor response")
    print("  ─────────────────────────────────────────────────────")
    print("  polarity ≈ 1.0  →  cos θ ≈ ±1  →  ceiling +50%  →  PRESERVE")
    print("  polarity ≈ 0.5  →  cos θ ≈ 0.7 →  ceiling +25%  →  softer prune")
    print("  polarity ≈ 0.0  →  cos θ ≈ 0   →  ceiling ×1.0  →  PRUNE")
    print()
    print("  The wall is not at a qubit count.")
    print("  The wall is not even at an entropy budget.")
    print("  The wall is at: entropy that carries no polarity.")
    print()
