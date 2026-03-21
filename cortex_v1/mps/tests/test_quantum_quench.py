"""
TFIM Quantum Quench Benchmark
==============================

Validates physical fidelity of Polarity Governor vs Fixed Governor
under identical memory constraints.

Observable: exact <Z_i> computed via environment contractions (no shot noise).
Metric:     MAE = (1/n) Σ |<Z_i>_approx - <Z_i>_reference|

Model:
    H = -J Σ Z_i Z_{i+1}  -  h Σ X_i

Trotter decomposition (first order, time step dt):
    exp(-i J dt Z_i Z_{i+1}) = CNOT(i,i+1) . Rz(i+1, -2*J*dt) . CNOT(i,i+1)
    exp(-i h dt X_i)          = Rx(i, -2*h*dt)

Three backends under identical chi_max and S_base:
    A. Reference      (chi=64, effectively unconstrained)
    B. Fixed Governor (chi=6,  alpha=0,     S_base=0.45)
    C. Polarity Gov   (chi=6,  alpha=0.8,   S_base=0.45, live alpha from lattice)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from livnium_cortex_v1 import (
    MPSSimulator,
    SemanticPolarityGovernor,
    LivniumLattice,
)


# ─────────────────────────────────────────────────────────────────────────────
# Exact <Z_i> via environment contractions (no sampling, no shot noise)
# ─────────────────────────────────────────────────────────────────────────────

def exact_z_expectations(sim: MPSSimulator) -> list:
    """
    Compute <Z_i> = P(0) - P(1) for every site using L/R environment contractions.
    Does not collapse the MPS — state is read-only.
    """
    R = sim._right_environments()
    L = np.array([[1.0 + 0j]])
    expectations = []

    for i in range(sim.n):
        t  = sim.tensors[i]   # (chi_l, 2, chi_r)
        Rn = R[i + 1]

        p0 = np.real(np.trace(L @ t[:, 0, :] @ Rn @ t[:, 0, :].conj().T))
        p1 = np.real(np.trace(L @ t[:, 1, :] @ Rn @ t[:, 1, :].conj().T))

        norm = p0 + p1
        if norm > 1e-12:
            p0, p1 = p0 / norm, p1 / norm
        else:
            p0, p1 = 0.5, 0.5

        expectations.append(float(p0 - p1))

        # Advance left environment — shape must grow with bond dimension
        chi_r = t.shape[2]
        Lnew = np.zeros((chi_r, chi_r), dtype=complex)
        for sigma in range(2):
            M = t[:, sigma, :]   # (chi_l, chi_r)
            Lnew += M.T @ L @ M.conj()  # (chi_r, chi_l) @ (chi_l, chi_l) @ (chi_l, chi_r)
        L = Lnew

    return expectations


# ─────────────────────────────────────────────────────────────────────────────
# Trotterized TFIM step through a governor
# ─────────────────────────────────────────────────────────────────────────────

def apply_tfim_trotter_step(gov, n: int, J: float, h: float, dt: float):
    """
    One first-order Trotter step of H = -J Σ Z_i Z_{i+1} - h Σ X_i.

    ZZ coupling: CNOT(i,i+1) . Rz(i+1, -2*J*dt) . CNOT(i,i+1)
    X  field:    Rx(i, -2*h*dt)
    """
    sim = gov.sim

    # ZZ coupling — enforce after each two-qubit block
    for i in range(n - 1):
        sim.cnot(i, i + 1)
        sim.rz(i + 1, -2.0 * J * dt)
        sim.cnot(i, i + 1)
        gov.enforce_all()

    # Transverse field (single-qubit, no entanglement change)
    for i in range(n):
        gov.rx(i, -2.0 * h * dt)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_quench_benchmark():
    n_qubits        = 14
    chi_ref         = 64
    chi_constrained = 8
    S_base          = 0.70
    alpha_val       = 0.8

    J     = 1.0
    h     = 0.5
    T     = 1.0
    steps = 10
    dt    = T / steps

    print("▓" * 72)
    print("  TFIM QUANTUM QUENCH BENCHMARK")
    print(f"  n={n_qubits}  J={J}  h={h}  T={T}  steps={steps}  dt={dt:.3f}")
    print(f"  chi_constrained={chi_constrained}  S_base={S_base}  alpha={alpha_val}")
    print("▓" * 72)

    # ── Instantiate three backends ────────────────────────────────────────────
    sim_ref   = MPSSimulator(n_qubits, max_bond_dim=chi_ref)
    sim_fixed = MPSSimulator(n_qubits, max_bond_dim=chi_constrained)
    sim_polar = MPSSimulator(n_qubits, max_bond_dim=chi_constrained)

    # Reference: effectively no pruning (S_max=10.0, alpha=0)
    gov_ref   = SemanticPolarityGovernor(sim_ref,   S_max=10.0,   alpha=0.0,      verbose=False)
    gov_fixed = SemanticPolarityGovernor(sim_fixed, S_max=S_base, alpha=0.0,      verbose=False)
    gov_polar = SemanticPolarityGovernor(sim_polar, S_max=S_base, alpha=alpha_val, verbose=False)

    lattice = LivniumLattice()

    # ── Prepare |+...+> initial state ────────────────────────────────────────
    for gov in [gov_ref, gov_fixed, gov_polar]:
        for i in range(n_qubits):
            gov.hadamard(i)

    print(f"\n  Evolving {steps} Trotter steps...")

    # ── Time evolution ────────────────────────────────────────────────────────
    for step in range(steps):
        # Update polarity governor's alpha from lattice geometry each step
        result = lattice.apply_rotation_with_polarity(lambda l: l.rotate_z90())
        gov_polar.alpha = result["mean_abs"]

        apply_tfim_trotter_step(gov_ref,   n_qubits, J, h, dt)
        apply_tfim_trotter_step(gov_fixed, n_qubits, J, h, dt)
        apply_tfim_trotter_step(gov_polar, n_qubits, J, h, dt)

        if (step + 1) % 5 == 0:
            print(f"  step {step+1}/{steps}  "
                  f"fixed_prune={len(gov_fixed.pruning_log)}  "
                  f"polar_prune={len(gov_polar.pruning_log)}  "
                  f"alpha={gov_polar.alpha:.4f}")

    # ── Exact observables via environment contractions ────────────────────────
    print(f"\n  Computing exact <Z_i> via environment contractions...")
    z_ref   = np.array(exact_z_expectations(sim_ref))
    z_fixed = np.array(exact_z_expectations(sim_fixed))
    z_polar = np.array(exact_z_expectations(sim_polar))

    mae_fixed = float(np.mean(np.abs(z_ref - z_fixed)))
    mae_polar = float(np.mean(np.abs(z_ref - z_polar)))

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  RESULTS  —  Physical Observable <Z_i> at t={T:.1f}")
    print(f"{'═'*72}")
    print(f"  {'Backend':<22} {'MAE vs ref':>12}  {'prune_events':>13}  {'chi_max':>8}")
    print(f"  {'─'*22} {'─'*12}  {'─'*13}  {'─'*8}")
    print(f"  {'Reference (chi=64)':<22} {'0.0000':>12}  {'—':>13}  {sim_ref.max_bond_dim_used:>8}")
    print(f"  {'Fixed Governor':<22} {mae_fixed:>12.4f}  {len(gov_fixed.pruning_log):>13}  {sim_fixed.max_bond_dim_used:>8}")
    print(f"  {'Polarity Governor':<22} {mae_polar:>12.4f}  {len(gov_polar.pruning_log):>13}  {sim_polar.max_bond_dim_used:>8}")

    print(f"\n  Site-by-site <Z_i> profile:")
    print(f"  {'site':>5}  {'Z_ref':>8}  {'Z_fixed':>9}  {'Z_polar':>9}  "
          f"{'err_fixed':>10}  {'err_polar':>10}  {'✓':>4}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*10}  {'─'*4}")

    polar_wins = 0
    for i in range(n_qubits):
        ef = abs(z_fixed[i] - z_ref[i])
        ep = abs(z_polar[i] - z_ref[i])
        win = ep < ef - 0.005
        if win:
            polar_wins += 1
        print(f"  {i:>5}  {z_ref[i]:>8.4f}  {z_fixed[i]:>9.4f}  {z_polar[i]:>9.4f}  "
              f"{ef:>10.4f}  {ep:>10.4f}  {'✓' if win else '':>4}")

    print(f"\n  {'─'*72}")
    improvement = (mae_fixed - mae_polar) / (mae_fixed + 1e-10) * 100

    if mae_polar < mae_fixed:
        print(f"\n  ✓ POLARITY GOVERNOR WINS")
        print(f"    MAE:  fixed={mae_fixed:.4f}  →  polar={mae_polar:.4f}")
        print(f"    Improvement: {improvement:.1f}%")
        print(f"    Sites where polarity governor is closer to reference: {polar_wins}/{n_qubits}")
        print(f"\n    Physical interpretation:")
        print(f"    Polarity-aware pruning preserves the TFIM magnetization profile")
        print(f"    better than fixed entropy limiting under identical chi_max={chi_constrained}")
        print(f"    and S_base={S_base}. The geometry-derived alpha signal biases")
        print(f"    truncation toward volume-law bonds, protecting structured entanglement.")
    else:
        print(f"\n  ⚠  Fixed governor matched or beat polarity governor")
        print(f"     MAE: fixed={mae_fixed:.4f}  polar={mae_polar:.4f}")
        print(f"     Try tighter S_base or larger alpha to enter conflict regime")

    print(f"\n{'▓'*72}")

    return {
        "z_ref":        z_ref,
        "z_fixed":      z_fixed,
        "z_polar":      z_polar,
        "mae_fixed":    mae_fixed,
        "mae_polar":    mae_polar,
        "prune_fixed":  len(gov_fixed.pruning_log),
        "prune_polar":  len(gov_polar.pruning_log),
        "polar_wins":   polar_wins,
        "improvement":  improvement,
    }


if __name__ == "__main__":
    run_quench_benchmark()
