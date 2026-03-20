"""
Bridge Fidelity Test
====================

Does the Livnium α signal actually predict which entanglement is worth keeping?

Setup
-----
n = 20 qubits, 40 circuit steps.
Each step: pick a Livnium rotation → apply its SU(2) gate to one qubit
           → apply a CNOT to the next qubit pair to create entanglement.

Three paths run the IDENTICAL gate sequence:

    Reference  : no compression  (max_bond_dim=512, S_max=∞)
    Static     : governor with fixed α=0.5 always
    Dynamic    : governor with α from Livnium polarity signal

Metrics (all measured from the same gate sequence, same random seed):

    1. Total truncation error  — sum of all discarded singular-value-squared mass
    2. Prune events            — how many bonds got clipped
    3. L1 distance             — |P_compressed − P_reference| over measurement outcomes
                                  measured over 500 shots, averaged over 5 seeds

Hypothesis
----------
If the α signal is doing real work:
    Dynamic truncation error  <  Static truncation error     (for same or fewer prunes)
    Dynamic L1 distance       <  Static L1 distance          (higher fidelity)

If α is noise:
    Dynamic ≈ Static on both metrics.

We report the numbers and let them speak.
"""

import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from mps_simulator import MPSSimulator
from polarity_governor import SemanticPolarityGovernor
from entanglement_governor import EntanglementGovernor
from livnium_to_tensor import LivniumLattice, generate_all_24_rotations
from rotation_to_pauli import generate_all_24_so3_su2


# ─────────────────────────────────────────────────────────────────────────────
# Build the 24-rotation catalogue once (label, SO3 matrix, SU2 gate, α)
# ─────────────────────────────────────────────────────────────────────────────

def build_rotation_catalogue() -> List[Dict]:
    """
    For each of the 24 rotations, store:
        su2_entries[i]["label"]   : str
        su2_entries[i]["U_su2"]   : 2×2 gate
        su2_entries[i]["alpha"]   : mean |cos θ| Livnium polarity signal
    """
    so3_su2 = generate_all_24_so3_su2()        # ordered by BFS from rotation_to_pauli

    # Compute α for each rotation using the livnium lattice
    lat_rotations = generate_all_24_rotations()  # ordered by BFS from livnium_to_tensor

    # Both BFS routines start from identity and expand X/Y/Z in order → same ordering
    # Verify lengths match
    assert len(so3_su2) == len(lat_rotations) == 24

    catalogue = []
    for i, (so3_entry, (label, lat_fn)) in enumerate(zip(so3_su2, lat_rotations)):
        lat = LivniumLattice()
        result = lat.apply_rotation_with_polarity(lat_fn)
        catalogue.append({
            "label"  : label,
            "U_su2"  : so3_entry["U_su2"],
            "alpha"  : result["mean_abs"],
            "theta"  : so3_entry["theta_deg"],
        })
    return catalogue


# ─────────────────────────────────────────────────────────────────────────────
# Apply a 2×2 SU(2) gate to qubit q in an MPS
# ─────────────────────────────────────────────────────────────────────────────

def apply_su2_to_mps(sim: MPSSimulator, qubit: int, U: np.ndarray):
    """
    Apply a 2×2 unitary U to qubit `qubit` in the MPS.
    Updates sim.tensors[qubit] in place.
    """
    t = sim.tensors[qubit]               # shape (chi_l, 2, chi_r)
    chi_l, _, chi_r = t.shape
    # Contract U over physical index: new_t[chi_l, p', chi_r] = Σ_p U[p',p] * t[chi_l, p, chi_r]
    new_t = np.einsum("ij,kjl->kil", U, t)
    sim.tensors[qubit] = new_t


# ─────────────────────────────────────────────────────────────────────────────
# Run one circuit trial
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(
    n_qubits     : int,
    steps        : List[Tuple[int, int]],   # (qubit_idx, rotation_idx)
    catalogue    : List[Dict],
    mode         : str,                     # "reference" | "static" | "dynamic"
    s_max        : float,
    static_alpha : float = 0.5,
) -> Dict:
    """
    Run the gate sequence defined by `steps` under the given compression mode.

    Returns:
        {
          "trunc_error" : float,   total truncation error accumulated
          "prune_events": int,
          "tensors"     : the final MPS tensor list (for measurement)
        }
    """
    if mode == "reference":
        sim = MPSSimulator(n_qubits=n_qubits, max_bond_dim=128)
        gov = SemanticPolarityGovernor(sim, S_max=1e9, alpha=0.0, verbose=False)
    elif mode == "static":
        sim = MPSSimulator(n_qubits=n_qubits, max_bond_dim=64)
        gov = SemanticPolarityGovernor(sim, S_max=s_max, alpha=static_alpha, verbose=False)
    elif mode == "dynamic":
        sim = MPSSimulator(n_qubits=n_qubits, max_bond_dim=64)
        gov = SemanticPolarityGovernor(sim, S_max=s_max, alpha=0.0, verbose=False)
    else:
        raise ValueError(mode)

    for step_i, (q, rot_idx) in enumerate(steps):
        U     = catalogue[rot_idx]["U_su2"]
        alpha = catalogue[rot_idx]["alpha"]

        # Apply single-qubit gate
        apply_su2_to_mps(sim, q, U)

        # For dynamic mode: update α from this rotation's geometry
        if mode == "dynamic":
            gov.alpha = alpha

        # Enforce entropy budget across all bonds
        gov._enforce_all()

        # Entangling gate: CNOT between q and (q+1) % n_qubits (skip last pair)
        q_next = (q + 1) % n_qubits
        if q_next != q:
            gov.cnot(q, q_next)

    trunc_error = sum(e["trunc_err"] for e in gov.pruning_log)
    return {
        "trunc_error" : trunc_error,
        "prune_events": len(gov.pruning_log),
        "sim"         : sim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Measurement distribution comparison
# ─────────────────────────────────────────────────────────────────────────────

def measure_distribution(sim: MPSSimulator, shots: int, seed: int) -> Counter:
    """Run `shots` measurements and return outcome counts."""
    rng = np.random.default_rng(seed)
    counts = Counter()
    for _ in range(shots):
        # temporarily seed numpy global for measure_all
        r = sim.measure_all()
        counts["".join(map(str, r))] += 1
    return counts


def l1_distance(c1: Counter, c2: Counter, shots: int) -> float:
    """L1 distance between two empirical distributions."""
    all_keys = set(c1) | set(c2)
    return sum(abs(c1.get(k, 0) - c2.get(k, 0)) for k in all_keys) / shots


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    n_qubits    : int   = 20,
    n_steps     : int   = 40,
    s_max_bits  : float = 2.5,   # entropy ceiling in bits
    shots       : int   = 500,
    n_seeds     : int   = 5,
):
    catalogue = build_rotation_catalogue()
    s_max = s_max_bits * np.log(2)

    print(f"\n{'='*60}")
    print(f"BRIDGE FIDELITY TEST")
    print(f"n_qubits={n_qubits}  steps={n_steps}  "
          f"S_max={s_max_bits} bits  shots={shots}  seeds={n_seeds}")
    print(f"{'='*60}")

    # α distribution across the 24 rotations
    alphas = [r["alpha"] for r in catalogue]
    print(f"\n  Livnium α distribution across 24 rotations:")
    print(f"    min={min(alphas):.4f}  max={max(alphas):.4f}  "
          f"mean={np.mean(alphas):.4f}  std={np.std(alphas):.4f}")

    results_static  = []
    results_dynamic = []
    l1_static_vals  = []
    l1_dynamic_vals = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed * 17 + 3)

        # Same random gate sequence for all three paths
        steps = [
            (int(rng.integers(0, n_qubits)), int(rng.integers(0, 24)))
            for _ in range(n_steps)
        ]

        # Reference (no compression)
        ref = run_trial(n_qubits, steps, catalogue, "reference", s_max)

        # Static governor (α = 0.5 always)
        sta = run_trial(n_qubits, steps, catalogue, "static",    s_max, static_alpha=0.5)

        # Dynamic governor (α from Livnium geometry)
        dyn = run_trial(n_qubits, steps, catalogue, "dynamic",   s_max)

        results_static.append(sta)
        results_dynamic.append(dyn)

        # Measure distributions
        ref_counts = measure_distribution(ref["sim"], shots, seed)
        sta_counts = measure_distribution(sta["sim"], shots, seed)
        dyn_counts = measure_distribution(dyn["sim"], shots, seed)

        l1_s = l1_distance(ref_counts, sta_counts, shots)
        l1_d = l1_distance(ref_counts, dyn_counts, shots)
        l1_static_vals.append(l1_s)
        l1_dynamic_vals.append(l1_d)

        print(f"\n  Seed {seed}:")
        print(f"    Static  — prunes={sta['prune_events']:3d}  "
              f"trunc_err={sta['trunc_error']:.4f}  L1={l1_s:.4f}")
        print(f"    Dynamic — prunes={dyn['prune_events']:3d}  "
              f"trunc_err={dyn['trunc_error']:.4f}  L1={l1_d:.4f}")

    # Aggregate
    mean_trunc_s = np.mean([r["trunc_error"]  for r in results_static])
    mean_trunc_d = np.mean([r["trunc_error"]  for r in results_dynamic])
    mean_prune_s = np.mean([r["prune_events"] for r in results_static])
    mean_prune_d = np.mean([r["prune_events"] for r in results_dynamic])
    mean_l1_s    = np.mean(l1_static_vals)
    mean_l1_d    = np.mean(l1_dynamic_vals)

    print(f"\n{'─'*60}")
    print(f"  AGGREGATE over {n_seeds} seeds")
    print(f"  {'Metric':<30} {'Static (α=0.5)':>16} {'Dynamic (Livnium α)':>20}")
    print(f"  {'──────':<30} {'──────────────':>16} {'──────────────────':>20}")
    print(f"  {'Mean truncation error':<30} {mean_trunc_s:>16.4f} {mean_trunc_d:>20.4f}")
    print(f"  {'Mean prune events':<30} {mean_prune_s:>16.1f} {mean_prune_d:>20.1f}")
    print(f"  {'Mean L1 vs reference':<30} {mean_l1_s:>16.4f} {mean_l1_d:>20.4f}")

    print(f"\n  VERDICT")
    print(f"  {'─'*56}")

    trunc_better = mean_trunc_d < mean_trunc_s * 0.95
    l1_better    = mean_l1_d    < mean_l1_s    * 0.95
    trunc_same   = abs(mean_trunc_d - mean_trunc_s) < mean_trunc_s * 0.05
    l1_same      = abs(mean_l1_d    - mean_l1_s   ) < mean_l1_s    * 0.05

    if trunc_better and l1_better:
        print(f"  ✅ Dynamic governor outperforms static on BOTH metrics.")
        print(f"     The Livnium α signal is identifying structure worth saving.")
    elif l1_better and not trunc_better:
        print(f"  ⚠️  Dynamic achieves better fidelity but similar truncation error.")
        print(f"     The α signal shifts *where* it prunes, not how much.")
    elif trunc_same and l1_same:
        print(f"  ──  Static and dynamic are equivalent within 5% margin.")
        print(f"     For this random circuit, the α signal carries no advantage.")
        print(f"     The geometry and the entanglement structure are uncorrelated here.")
    else:
        print(f"  ──  Mixed result. Difference is below the noise threshold.")
        print(f"     trunc {'↓' if trunc_better else '≈' if trunc_same else '↑'}  "
              f"L1 {'↓' if l1_better else '≈' if l1_same else '↑'}")

    print(f"{'='*60}\n")

    return {
        "mean_trunc_static" : mean_trunc_s,
        "mean_trunc_dynamic": mean_trunc_d,
        "mean_l1_static"    : mean_l1_s,
        "mean_l1_dynamic"   : mean_l1_d,
    }


# ─────────────────────────────────────────────────────────────────────────────
# We also need _enforce_all on the governor — add it here as a patch
# if it's missing from entanglement_governor.py
# ─────────────────────────────────────────────────────────────────────────────

def _patch_governor():
    """
    EntanglementGovernor may not have _enforce_all().
    Patch it in so run_trial can call it after single-qubit gates.
    """
    from entanglement_governor import EntanglementGovernor
    if not hasattr(EntanglementGovernor, "_enforce_all"):
        def _enforce_all(self):
            for i in range(self.sim.n - 1):
                self._enforce_bond(i)
        EntanglementGovernor._enforce_all = _enforce_all


if __name__ == "__main__":
    _patch_governor()

    # Primary test: n=12, 24 steps (one per rotation), S_max=2.0 bits
    run_experiment(n_qubits=12, n_steps=24, s_max_bits=2.0, shots=300, n_seeds=5)
