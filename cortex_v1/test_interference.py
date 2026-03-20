"""
Quantum Interference Test for Livnium Lattice Extension
========================================================

Brutal question: does the observed interference effect require complex phase,
or can a classical stochastic ensemble reproduce it?

Method:
  1. Define two complex amplitude states ψ_A, ψ_B on an N³ lattice
  2. Compute coherent superposition:  ψ_S = (ψ_A + ψ_B) / √2
  3. Compute classical mixture:        P_mix = ½P(ψ_A) + ½P(ψ_B)
  4. Measure L1 distance Δ = ||P_S − P_mix||₁
     - If Δ > 0 due to Re(ψ_A ψ_B*) cross-term → genuine interference
  5. Decoherence sweep: average over K random-phase trials
     - If Δ ~ 1/√K → classical Monte Carlo noise (NOT quantum)
     - If Δ collapses faster or has a different signature → investigate further

If a classical random-phase model reproduces Δ ~ 1/√K,
the "quantum-like" claim is NOT supported.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Lattice setup
# ---------------------------------------------------------------------------

def make_lattice_sites(N: int) -> np.ndarray:
    """Return all (x,y,z) coordinates for an N³ lattice."""
    coords = np.array(
        [(x, y, z)
         for x in range(-(N-1)//2, (N-1)//2 + 1)
         for y in range(-(N-1)//2, (N-1)//2 + 1)
         for z in range(-(N-1)//2, (N-1)//2 + 1)],
        dtype=int
    )
    return coords   # shape (N³, 3)


def random_unit_amplitude(n_sites: int, rng: np.random.Generator) -> np.ndarray:
    """Random complex unit vector on C^{n_sites}."""
    real = rng.standard_normal(n_sites)
    imag = rng.standard_normal(n_sites)
    psi = real + 1j * imag
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# Probability distribution from amplitude
# ---------------------------------------------------------------------------

def prob(psi: np.ndarray) -> np.ndarray:
    """Born rule: P(i) = |ψ(i)|²."""
    return np.abs(psi) ** 2


# ---------------------------------------------------------------------------
# Interference measurement
# ---------------------------------------------------------------------------

def coherent_superposition(psi_A: np.ndarray, psi_B: np.ndarray) -> np.ndarray:
    return (psi_A + psi_B) / np.sqrt(2)


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))


def interference_delta(psi_A: np.ndarray, psi_B: np.ndarray) -> float:
    """
    Δ = ||P_coherent − P_mixture||₁

    P_coherent = |ψ_A + ψ_B|² / 2  = ½(|ψ_A|² + |ψ_B|² + 2Re(ψ_A ψ_B*))
    P_mixture  = ½(|ψ_A|² + |ψ_B|²)

    So Δ directly measures the Re(ψ_A ψ_B*) cross-term.
    """
    P_coh = prob(coherent_superposition(psi_A, psi_B))
    P_mix = 0.5 * (prob(psi_A) + prob(psi_B))
    return l1_distance(P_coh, P_mix)


# ---------------------------------------------------------------------------
# Decoherence sweep
# ---------------------------------------------------------------------------

def decoherence_sweep(
    psi_A: np.ndarray,
    psi_B: np.ndarray,
    K_values: list,
    rng: np.random.Generator,
    n_repeats: int = 5,
) -> dict:
    """
    Average over K random-phase trials:
        ψ_k = (ψ_A + e^{iθ_k} ψ_B) / √2,  θ_k ~ Uniform[0, 2π)

    Returns mean Δ and std for each K.
    """
    results = {}
    for K in K_values:
        deltas = []
        for _ in range(n_repeats):
            P_avg = np.zeros(len(psi_A))
            for _ in range(K):
                theta = rng.uniform(0, 2 * np.pi)
                psi_k = (psi_A + np.exp(1j * theta) * psi_B) / np.sqrt(2)
                P_avg += prob(psi_k)
            P_avg /= K
            P_mix = 0.5 * (prob(psi_A) + prob(psi_B))
            deltas.append(l1_distance(P_avg, P_mix))
        results[K] = (np.mean(deltas), np.std(deltas))
    return results


# ---------------------------------------------------------------------------
# Classical baseline: can a stochastic ensemble reproduce the same Δ?
# ---------------------------------------------------------------------------

def classical_baseline(
    n_sites: int,
    K_values: list,
    rng: np.random.Generator,
    n_repeats: int = 5,
) -> dict:
    """
    Replace ψ_A, ψ_B with fresh random amplitudes each trial.
    If Δ matches the quantum sweep, the effect is classical.
    """
    results = {}
    for K in K_values:
        deltas = []
        for _ in range(n_repeats):
            P_avg = np.zeros(n_sites)
            psi_A = random_unit_amplitude(n_sites, rng)
            psi_B = random_unit_amplitude(n_sites, rng)
            for _ in range(K):
                theta = rng.uniform(0, 2 * np.pi)
                psi_k = (psi_A + np.exp(1j * theta) * psi_B) / np.sqrt(2)
                P_avg += prob(psi_k)
            P_avg /= K
            P_mix = 0.5 * (prob(psi_A) + prob(psi_B))
            deltas.append(l1_distance(P_avg, P_mix))
        results[K] = (np.mean(deltas), np.std(deltas))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 3
    sites = make_lattice_sites(N)
    n_sites = len(sites)   # 27 for N=3

    print(f"Lattice: {N}³ = {n_sites} sites")
    print("=" * 60)

    rng = np.random.default_rng(seed=42)

    # Fixed ψ_A, ψ_B for reproducibility
    psi_A = random_unit_amplitude(n_sites, rng)
    psi_B = random_unit_amplitude(n_sites, rng)

    # --- Single coherent measurement ---
    delta_coherent = interference_delta(psi_A, psi_B)
    cross_term = float(np.sum(np.real(psi_A * np.conj(psi_B))))
    print(f"\nSingle coherent Δ : {delta_coherent:.4f}")
    print(f"Re(ψ_A · ψ_B*)    : {cross_term:.4f}  ← this IS the interference term")

    # --- Decoherence sweep ---
    K_values = [1, 4, 16, 64, 128, 512]
    print("\nDecoherence sweep (quantum amplitudes, fixed ψ_A/ψ_B):")
    print(f"{'K':>6}  {'mean Δ':>10}  {'std':>8}  {'1/√K':>8}  {'ratio Δ/(1/√K)':>16}")
    sweep = decoherence_sweep(psi_A, psi_B, K_values, rng)
    for K, (mean_d, std_d) in sweep.items():
        inv_sqrtK = 1.0 / np.sqrt(K)
        ratio = mean_d / inv_sqrtK if inv_sqrtK > 0 else float("nan")
        print(f"{K:>6}  {mean_d:>10.4f}  {std_d:>8.4f}  {inv_sqrtK:>8.4f}  {ratio:>16.4f}")

    # --- Classical baseline ---
    print("\nClassical baseline (fresh random amplitudes each trial):")
    print(f"{'K':>6}  {'mean Δ':>10}  {'std':>8}")
    baseline = classical_baseline(n_sites, K_values, rng)
    for K, (mean_d, std_d) in baseline.items():
        print(f"{K:>6}  {mean_d:>10.4f}  {std_d:>8.4f}")

    # --- Verdict ---
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    quantum_K1   = sweep[1][0]
    classical_K1 = baseline[1][0]
    ratio = quantum_K1 / classical_K1 if classical_K1 > 0 else float("nan")

    print(f"Quantum Δ (K=1)   : {quantum_K1:.4f}")
    print(f"Classical Δ (K=1) : {classical_K1:.4f}")
    print(f"Ratio              : {ratio:.3f}")

    if ratio > 1.5:
        print("\n→ Quantum amplitudes produce STRONGER interference than classical baseline.")
        print("  The effect is NOT fully explained by classical ensemble averaging.")
        print("  Worth investigating further.")
    elif 0.67 < ratio < 1.5:
        print("\n→ Quantum and classical baselines produce SIMILAR Δ.")
        print("  The 1/√K scaling is consistent with classical Monte Carlo noise.")
        print("  The 'quantum-like' claim is NOT supported by this test.")
    else:
        print("\n→ Unexpected ratio. Inspect the data manually.")

    # --- Scaling check: is Δ ~ 1/√K? ---
    print("\nScaling check: fit Δ ~ C / √K")
    Ks = np.array(K_values, dtype=float)
    deltas = np.array([sweep[K][0] for K in K_values])
    # Linear fit on log-log
    log_K = np.log(Ks)
    log_d = np.log(deltas + 1e-12)
    slope, intercept = np.polyfit(log_K, log_d, 1)
    print(f"  log-log slope: {slope:.3f}  (quantum prediction: −0.5)")
    if abs(slope + 0.5) < 0.15:
        print("  → Slope consistent with 1/√K. Classical averaging is sufficient explanation.")
    else:
        print(f"  → Slope {slope:.3f} deviates from −0.5. Investigate the dynamics.")
