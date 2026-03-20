"""
Guided Alpha Utility Test
=========================

Tests whether the Livnium α signal is a FUNCTIONAL control knob when
the circuit is designed to correlate with it.

Key finding from bridge_fidelity_test.py:
    α distribution: min=0.0, max=0.841, mean=0.704
    LOW  α ≈ 0.595: the 6 quarter-turns (90°) — Rx, Ry, Rz type
    HIGH α ≈ 0.836: the 9 half-turns (180°) — Pauli / Hadamard family

Circuit design (engineered correlation):
    Phase 1 — STRUCTURE (n steps using HIGH-α rotations + CNOTs)
        High-α rotations are the 180° family: they flip qubits or create
        Hadamard-like superpositions. Combined with CNOTs they build
        structured entanglement across bonds.
        Dynamic governor sees high α → relaxes ceiling → preserves bonds.

    Phase 2 — NOISE (n steps using LOW-α rotations, no new CNOTs)
        Low-α rotations are the 90° family: they add superposition on top
        of already-entangled qubits, creating phase noise across bonds.
        Dynamic governor sees low α → standard ceiling → prunes aggressively.

Three paths on identical gate sequences:
    Reference  : no compression  (max_bond_dim=128)
    Static     : fixed α=0.5 throughout both phases
    Dynamic    : α from Livnium geometry (high in Phase 1, low in Phase 2)

Metrics:
    trunc_phase1   : truncation error accumulated during Phase 1
    trunc_phase2   : truncation error accumulated during Phase 2
    prunes_phase1  : prune events in Phase 1
    prunes_phase2  : prune events in Phase 2
    L1_final       : L1 vs reference distribution after both phases (300 shots)

Hypothesis:
    Dynamic trunc_phase1 < Static trunc_phase1  (high α protects Phase 1 bonds)
    Dynamic trunc_phase2 > Static trunc_phase2  (low α prunes Phase 2 harder)
    Dynamic L1_final     < Static L1_final      (net fidelity advantage)

If the hypothesis holds: α is a functional control knob.
If not: the signal has no mechanical advantage even with engineered correlation.
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
# Build catalogue and identify high / low α rotation groups
# ─────────────────────────────────────────────────────────────────────────────

def build_catalogue() -> Tuple[List[Dict], List[int], List[int]]:
    """
    Returns (catalogue, high_alpha_indices, low_alpha_indices).
    high: α > mean + 0.5σ  (180° half-turn family)
    low : α < mean − 0.5σ  (90° quarter-turn family, excluding identity)
    """
    lat_rots = generate_all_24_rotations()
    so3_rots = generate_all_24_so3_su2()

    catalogue = []
    for (label, lat_fn), so3 in zip(lat_rots, so3_rots):
        lat    = LivniumLattice()
        result = lat.apply_rotation_with_polarity(lat_fn)
        catalogue.append({
            "label"    : label,
            "U_su2"    : so3["U_su2"],
            "alpha"    : result["mean_abs"],
            "theta_deg": so3["theta_deg"],
        })

    alphas     = np.array([r["alpha"] for r in catalogue])
    mean_a     = alphas.mean()
    std_a      = alphas.std()
    high_idx   = [i for i, r in enumerate(catalogue) if r["alpha"] > mean_a + 0.5 * std_a]
    low_idx    = [i for i, r in enumerate(catalogue)
                  if 0 < r["alpha"] < mean_a - 0.5 * std_a]  # exclude identity (α=0)

    return catalogue, high_idx, low_idx


# ─────────────────────────────────────────────────────────────────────────────
# Apply SU(2) gate to one qubit of the MPS
# ─────────────────────────────────────────────────────────────────────────────

def apply_su2(sim: MPSSimulator, qubit: int, U: np.ndarray):
    t = sim.tensors[qubit]
    sim.tensors[qubit] = np.einsum("ij,kjl->kil", U, t)


def patch_enforce_all():
    if not hasattr(EntanglementGovernor, "_enforce_all"):
        def _enforce_all(self):
            for i in range(self.sim.n - 1):
                self._enforce_bond(i)
        EntanglementGovernor._enforce_all = _enforce_all


# ─────────────────────────────────────────────────────────────────────────────
# Run one trial through both phases
# ─────────────────────────────────────────────────────────────────────────────

def run_guided_trial(
    n_qubits      : int,
    phase1_steps  : List[Tuple[int, int]],   # (qubit, rot_idx) — HIGH α
    phase2_steps  : List[Tuple[int, int]],   # (qubit, rot_idx) — LOW  α
    catalogue     : List[Dict],
    mode          : str,                     # "reference" | "static" | "dynamic"
    s_max         : float,
) -> Dict:
    if mode == "reference":
        sim = MPSSimulator(n_qubits=n_qubits, max_bond_dim=128)
        gov = SemanticPolarityGovernor(sim, S_max=1e9, alpha=0.0, verbose=False)
    else:
        sim = MPSSimulator(n_qubits=n_qubits, max_bond_dim=64)
        alpha_init = 0.5 if mode == "static" else 0.0
        gov = SemanticPolarityGovernor(sim, S_max=s_max, alpha=alpha_init, verbose=False)

    def step(rot_idx, qubit, with_cnot):
        U     = catalogue[rot_idx]["U_su2"]
        alpha = catalogue[rot_idx]["alpha"]
        if mode == "dynamic":
            gov.alpha = alpha
        apply_su2(sim, qubit, U)
        gov._enforce_all()
        if with_cnot:
            q_next = (qubit + 1) % n_qubits
            gov.cnot(qubit, q_next)

    # ── Phase 1: high-α rotations + CNOTs ────────────────────────────────
    prunes_before_p2 = 0
    trunc_before_p2  = 0.0
    for q, rot_idx in phase1_steps:
        step(rot_idx, q, with_cnot=True)

    prunes_phase1 = len(gov.pruning_log) - prunes_before_p2
    trunc_phase1  = sum(e["trunc_err"] for e in gov.pruning_log[:prunes_phase1])

    # ── Phase 2: low-α rotations, NO new CNOTs ───────────────────────────
    n_after_p1 = len(gov.pruning_log)
    for q, rot_idx in phase2_steps:
        step(rot_idx, q, with_cnot=False)

    prunes_phase2 = len(gov.pruning_log) - n_after_p1
    trunc_phase2  = sum(e["trunc_err"] for e in gov.pruning_log[n_after_p1:])

    return {
        "sim"          : sim,
        "prunes_phase1": prunes_phase1,
        "prunes_phase2": prunes_phase2,
        "trunc_phase1" : trunc_phase1,
        "trunc_phase2" : trunc_phase2,
        "prunes_total" : len(gov.pruning_log),
        "trunc_total"  : sum(e["trunc_err"] for e in gov.pruning_log),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_l1(sim_ref: MPSSimulator, sim_cmp: MPSSimulator,
               shots: int) -> float:
    ref_counts: Counter = Counter()
    cmp_counts: Counter = Counter()
    for _ in range(shots):
        ref_counts["".join(map(str, sim_ref.measure_all()))] += 1
        cmp_counts["".join(map(str, sim_cmp.measure_all()))] += 1
    all_keys = set(ref_counts) | set(cmp_counts)
    return sum(abs(ref_counts.get(k, 0) - cmp_counts.get(k, 0))
               for k in all_keys) / shots


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    n_qubits   : int   = 12,
    n_phase    : int   = 10,    # steps per phase
    s_max_bits : float = 2.0,
    shots      : int   = 300,
    n_seeds    : int   = 5,
):
    patch_enforce_all()
    catalogue, high_idx, low_idx = build_catalogue()
    s_max = s_max_bits * np.log(2)

    high_alphas = [catalogue[i]["alpha"] for i in high_idx]
    low_alphas  = [catalogue[i]["alpha"] for i in low_idx]

    print(f"\n{'='*62}")
    print(f"GUIDED ALPHA UTILITY TEST")
    print(f"n_qubits={n_qubits}  phase_steps={n_phase}  "
          f"S_max={s_max_bits} bits  shots={shots}  seeds={n_seeds}")
    print(f"{'='*62}")
    print(f"\n  HIGH-α rotations ({len(high_idx)} total, all 180°):")
    print(f"    indices={high_idx}")
    print(f"    α values: min={min(high_alphas):.4f} max={max(high_alphas):.4f}")
    print(f"  LOW-α  rotations ({len(low_idx)} total, all 90°):")
    print(f"    indices={low_idx}")
    print(f"    α values: min={min(low_alphas):.4f} max={max(low_alphas):.4f}")
    print(f"\n  Phase 1: HIGH-α gates + CNOTs → build entanglement")
    print(f"  Phase 2: LOW-α  gates, no CNOTs → add phase noise\n")

    agg = {k: [] for k in [
        "trunc_p1_sta", "trunc_p1_dyn",
        "trunc_p2_sta", "trunc_p2_dyn",
        "prune_p1_sta", "prune_p1_dyn",
        "prune_p2_sta", "prune_p2_dyn",
        "l1_sta", "l1_dyn",
    ]}

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed * 31 + 7)

        # Same gate sequence for all paths
        p1_steps = [
            (int(rng.integers(0, n_qubits)), int(rng.choice(high_idx)))
            for _ in range(n_phase)
        ]
        p2_steps = [
            (int(rng.integers(0, n_qubits)), int(rng.choice(low_idx)))
            for _ in range(n_phase)
        ]

        ref = run_guided_trial(n_qubits, p1_steps, p2_steps, catalogue,
                               "reference", s_max)
        sta = run_guided_trial(n_qubits, p1_steps, p2_steps, catalogue,
                               "static",    s_max)
        dyn = run_guided_trial(n_qubits, p1_steps, p2_steps, catalogue,
                               "dynamic",   s_max)

        l1_s = measure_l1(ref["sim"], sta["sim"], shots)
        l1_d = measure_l1(ref["sim"], dyn["sim"], shots)

        agg["trunc_p1_sta"].append(sta["trunc_phase1"])
        agg["trunc_p1_dyn"].append(dyn["trunc_phase1"])
        agg["trunc_p2_sta"].append(sta["trunc_phase2"])
        agg["trunc_p2_dyn"].append(dyn["trunc_phase2"])
        agg["prune_p1_sta"].append(sta["prunes_phase1"])
        agg["prune_p1_dyn"].append(dyn["prunes_phase1"])
        agg["prune_p2_sta"].append(sta["prunes_phase2"])
        agg["prune_p2_dyn"].append(dyn["prunes_phase2"])
        agg["l1_sta"].append(l1_s)
        agg["l1_dyn"].append(l1_d)

        print(f"  Seed {seed}:")
        print(f"    Phase1  trunc  static={sta['trunc_phase1']:.4f}  "
              f"dynamic={dyn['trunc_phase1']:.4f}  "
              f"({'DYN lower ✓' if dyn['trunc_phase1'] < sta['trunc_phase1'] else 'no diff'})")
        print(f"    Phase2  trunc  static={sta['trunc_phase2']:.4f}  "
              f"dynamic={dyn['trunc_phase2']:.4f}  "
              f"({'DYN higher ✓' if dyn['trunc_phase2'] > sta['trunc_phase2'] else 'no diff'})")
        print(f"    L1      static={l1_s:.4f}  dynamic={l1_d:.4f}  "
              f"({'DYN better ✓' if l1_d < l1_s else 'no diff'})")

    def m(key): return np.mean(agg[key])

    print(f"\n{'─'*62}")
    print(f"  AGGREGATE (mean over {n_seeds} seeds)")
    print(f"\n  {'Metric':<35} {'Static':>10} {'Dynamic':>10}  Result")
    print(f"  {'──────':<35} {'──────':>10} {'───────':>10}  ──────")

    def verdict(static_val, dynamic_val, want_lower):
        diff_pct = abs(dynamic_val - static_val) / max(static_val, 1e-9) * 100
        if diff_pct < 5:
            return "≈ same"
        if want_lower:
            return "DYN ↓ ✅" if dynamic_val < static_val else "DYN ↑ ❌"
        else:
            return "DYN ↑ ✅" if dynamic_val > static_val else "DYN ↓ ❌"

    rows = [
        ("Phase1 trunc error (want DYN ↓)",
         m("trunc_p1_sta"), m("trunc_p1_dyn"), True),
        ("Phase1 prune events (want DYN ↓)",
         m("prune_p1_sta"), m("prune_p1_dyn"), True),
        ("Phase2 trunc error (want DYN ↑)",
         m("trunc_p2_sta"), m("trunc_p2_dyn"), False),
        ("Phase2 prune events (want DYN ↑)",
         m("prune_p2_sta"), m("prune_p2_dyn"), False),
        ("Final L1 vs reference (want DYN ↓)",
         m("l1_sta"), m("l1_dyn"), True),
    ]
    for label, sv, dv, want_lower in rows:
        print(f"  {label:<35} {sv:>10.4f} {dv:>10.4f}  {verdict(sv, dv, want_lower)}")

    # Count wins
    wins = sum(1 for _, sv, dv, wl in rows
               if abs(dv-sv)/max(sv, 1e-9)*100 >= 5
               and ((wl and dv < sv) or (not wl and dv > sv)))
    losses = sum(1 for _, sv, dv, wl in rows
                 if abs(dv-sv)/max(sv, 1e-9)*100 >= 5
                 and not ((wl and dv < sv) or (not wl and dv > sv)))
    ties = len(rows) - wins - losses

    print(f"\n  VERDICT")
    print(f"  {'─'*58}")
    print(f"  Wins={wins}  Losses={losses}  Ties={ties}  (threshold: 5% margin)")
    print()
    if wins >= 3 and losses == 0:
        print(f"  ✅ α signal is a functional control knob.")
        print(f"     Phase 1 bonds are better preserved under high-α guidance.")
        print(f"     Phase 2 noise is more aggressively pruned under low-α guidance.")
        print(f"     Claim: Livnium geometry is a valid heuristic for Schmidt-spectrum")
        print(f"     preservation when the circuit is designed to match it.")
    elif wins > losses:
        print(f"  ⚠️  Partial evidence. α helps on some metrics but not all.")
        print(f"     The signal has mechanical effect but it is not consistent.")
    else:
        print(f"  ──  No advantage. Even with engineered correlation, α does not")
        print(f"     outperform a static governor. The mechanism does not hold.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    run_experiment(n_qubits=12, n_phase=10, s_max_bits=2.0, shots=300, n_seeds=5)
