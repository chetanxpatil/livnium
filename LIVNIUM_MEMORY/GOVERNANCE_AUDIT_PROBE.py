#!/usr/bin/env python3
"""Independent probes for the Livnium rule/economy/governance family.

This script is deliberately read-only with respect to the source archive.  It
imports the archived experiment modules without calling their ``main``
functions, recomputes exact boundaries, and checks whether the reported
mechanisms use the information described in their narratives.

Usage:
    python GOVERNANCE_AUDIT_PROBE.py
    python GOVERNANCE_AUDIT_PROBE.py --source /path/to/Rule-Economy-Governance
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Importing an archived module must never leave __pycache__ in the source tree.
sys.dont_write_bytecode = True


DEFAULT_SOURCE = Path(
    "/Users/chetanpatil/Desktop/test/"
    "_ORGANIZED/02_Experiments/Rule-Economy-Governance"
)
DEFAULT_ROOT = Path("/Users/chetanpatil/Desktop/test")


def load_module(path: Path, label: str):
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_duplicate_audit(source: Path, root: Path) -> dict[str, Any]:
    rows = []
    for canonical in sorted(source.iterdir()):
        if not canonical.is_file():
            continue
        duplicate = root / canonical.name
        canonical_hash = sha256(canonical)
        duplicate_hash = sha256(duplicate) if duplicate.is_file() else None
        rows.append(
            {
                "name": canonical.name,
                "canonical_sha256": canonical_hash,
                "root_exists": duplicate.is_file(),
                "root_exact_copy": duplicate_hash == canonical_hash,
            }
        )
    return {
        "canonical_files": len(rows),
        "root_exact_copies": sum(row["root_exact_copy"] for row in rows),
        "rows": rows,
    }


def source_weighted_median(values, weights) -> float:
    values = np.asarray(values)
    weights = np.asarray(weights)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def anchor_boundary() -> dict[str, Any]:
    # Idealized source construction: 12 honest zeros, 28 cartel fives, one
    # exact-zero anchor carrying lambda weight.
    rows = {}
    for lam in (15.0, 15.999999, 16.0, 16.000001, 17.0):
        values = np.r_[np.zeros(12), np.full(28, 5.0), 0.0]
        weights = np.r_[np.ones(40), lam]
        rows[str(lam)] = source_weighted_median(values, weights)
    return {
        "p": 0.70,
        "N": 40,
        "printed_threshold": 16.0,
        "source_tie_break_results": rows,
        "finding": (
            "With the source's lower-endpoint tie convention, lambda=16 returns "
            "truth. The implementation therefore holds at >=16 in the ideal "
            "point-mass case; the median set is non-unique exactly at equality."
        ),
    }


def majority_anchor_probe(module) -> dict[str, Any]:
    rows = {}
    for n_anchor in (0, 1, 3, 5):
        module.rng = np.random.default_rng(1729)
        _, _, err = module.trial(
            0.70, coordinated=True, n_anchor=n_anchor, reps=6000
        )
        rows[str(n_anchor)] = float(err)
    return {
        "anchored_error_70pct_cartel": rows,
        "anchor_generation": "V_TRUE + HONEST_NOISE * random_normal",
        "selection_window": "absolute distance from anchor median < LIE_SHIFT/2",
        "finding": (
            "One trusted sample generated directly around truth already supplies "
            "almost the full result; additional anchors mainly reduce oracle noise."
        ),
    }


def exact_deterrence(
    penalty: float,
    q: float,
    stake: float,
    gain: float = 1.0,
    horizon: int = 40,
) -> float:
    """Exact expectation for the archived detection/slashing process."""
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    r = 1.0 - q
    expected = 0.0
    for t in range(1, horizon + 1):
        caught_probability = (r ** (t - 1)) * q
        cumulative_gain = t * gain
        slash = min(penalty * cumulative_gain, stake)
        expected += caught_probability * (cumulative_gain - slash)
    expected += (r**horizon) * (horizon * gain)
    return expected


def unlimited_stake_threshold(q: float, horizon: int) -> float:
    r = 1.0 - q
    expected_gain = sum((r ** (t - 1)) * 1.0 for t in range(1, horizon + 1))
    expected_caught_gain = sum(
        (r ** (t - 1)) * q * t for t in range(1, horizon + 1)
    )
    return expected_gain / expected_caught_gain


def deterrence_probe() -> dict[str, Any]:
    thresholds = {}
    for q in (1.0, 0.40, 0.15):
        thresholds[str(q)] = {
            "single_shot_exact": 1.0 / q,
            "forty_round_exact": unlimited_stake_threshold(q, 40),
            "escape_probability_40": (1.0 - q) ** 40,
        }
    stake_rows = {
        str(stake): exact_deterrence(
            penalty=100.0, q=0.10, stake=float(stake), gain=20.0, horizon=1
        )
        for stake in (2, 5, 10, 20, 40, 80, 200)
    }
    return {
        "thresholds": thresholds,
        "stake_cap_exact_net": stake_rows,
        "one_shot_stake_break_even": 200.0,
        "finding": (
            "These thresholds follow exactly from the stipulated exogenous "
            "detection probability and slash formula. At q=.10 and gain=20, "
            "stake 200—not 80—is the one-shot break-even cap."
        ),
    }


def shared_fate_probe(source: Path) -> dict[str, Any]:
    saved = json.loads((source / "livnium_shared_fate.json").read_text())
    capturing = {
        float(p): float(error)
        for p, error in saved["Ecurve"].items()
        if float(p) > 0.5
    }
    exact_thresholds = {str(p): 1.0 / error for p, error in capturing.items()}
    return {
        "saved_grid_kstar": saved["kstar"],
        "stated_g_over_W": saved["theory_kstar"],
        "exact_per_saved_point_thresholds": exact_thresholds,
        "identity": "liar_net = G - kappa * Ecurve[p] * retained",
        "finding": (
            "Negative payoff is a direct consequence of inserting the shared-loss "
            "term into the payoff definition. No agents choose, adapt, coordinate, "
            "exit, hedge, or update in this experiment."
        ),
    }


def matrix_condition(B: np.ndarray) -> tuple[int, float, list[float]]:
    singular = np.linalg.svd(B, compute_uv=False)
    rank = int(np.linalg.matrix_rank(B, tol=1e-10))
    condition = float(singular[0] / singular[-1])
    return rank, condition, singular.tolist()


def same_layer_probe(module) -> dict[str, Any]:
    n_ties = 40
    block = np.argsort([x for x, _y, _z in module.COORDS])[:n_ties]
    B_block = module.EVEC[block, : module.KLOW]
    block_rank, block_condition, block_singular = matrix_condition(B_block)

    conds = []
    ranks = []
    rng = np.random.default_rng(20260726)
    for _ in range(500):
        idx = rng.choice(module.NC, n_ties, replace=False)
        rank, condition, _ = matrix_condition(module.EVEC[idx, : module.KLOW])
        ranks.append(rank)
        conds.append(condition)

    Vtrue = module.smooth_field(1)
    Vfake = module.smooth_field(2)

    def corruption_comparison(frac: float, trials: int = 160):
        source_errors = []
        aligned_errors = []
        mismatch_counts = []
        local_rng = np.random.default_rng(914 + int(frac * 100))
        nc = int(frac * n_ties)
        for _ in range(trials):
            idx = local_rng.choice(module.NC, n_ties, replace=False)
            noise = 0.3 * local_rng.standard_normal(n_ties)
            chosen = local_rng.choice(n_ties, nc, replace=False)
            source_values = Vtrue[idx] + noise
            aligned_values = source_values.copy()
            # Archived assignment: selected rows get fake values from idx[:nc].
            source_values[chosen] = Vfake[idx[:nc]]
            # Cell-aligned assignment: selected rows get the fake value at that cell.
            aligned_values[chosen] = Vfake[idx[chosen]]
            mismatch_counts.append(int(np.sum(idx[:nc] != idx[chosen])))
            source_errors.append(module.rel(module.reconstruct(idx, source_values), Vtrue))
            aligned_errors.append(
                module.rel(module.reconstruct(idx, aligned_values), Vtrue)
            )
        return {
            "source_misaligned_mean_error": float(np.mean(source_errors)),
            "cell_aligned_mean_error": float(np.mean(aligned_errors)),
            "mean_wrong_cell_assignments": float(np.mean(mismatch_counts)),
            "corrupted_rows": nc,
        }

    return {
        "known_truth_dimension": int(module.KLOW),
        "same_layer_ties": n_ties,
        "concentrated_block": {
            "rank": block_rank,
            "condition_number": block_condition,
            "singular_values": block_singular,
        },
        "random_distributed_500_draws": {
            "all_ranks": sorted(set(ranks)),
            "condition_median": float(np.median(conds)),
            "condition_p95": float(np.quantile(conds, 0.95)),
            "condition_max": float(np.max(conds)),
        },
        "corruption_assignment": {
            "20pct": corruption_comparison(0.20),
            "40pct": corruption_comparison(0.40),
        },
        "finding": (
            "The experiment is sparse low-rank sensing with the exact generating "
            "basis known to the decoder. Distributed samples are better conditioned "
            "than one spatial block. The same-layer corruption branch also assigns "
            "fake values from different cells than the rows it corrupts."
        ),
    }


def scalar_orbits(perms: list[np.ndarray], n_cells: int) -> list[list[int]]:
    unseen = set(range(n_cells))
    orbits = []
    while unseen:
        seed = min(unseen)
        orbit = {int(perm[seed]) for perm in perms}
        orbits.append(sorted(orbit))
        unseen -= orbit
    return orbits


def selector_probe(module) -> dict[str, Any]:
    perms = module.rotation_perms()
    P = module.sym_projector(perms)
    rank = int(np.linalg.matrix_rank(P, tol=1e-10))
    orbits = scalar_orbits(perms, module.NC)
    orbit_sizes = Counter(len(orbit) for orbit in orbits)
    sigma = 1.0 / math.sqrt(module.NC)
    return {
        "shape": list(P.shape),
        "rank": rank,
        "trace": float(np.trace(P)),
        "symmetry_error_fro": float(np.linalg.norm(P - P.T)),
        "idempotence_error_fro": float(np.linalg.norm(P @ P - P)),
        "rotation_orbits": len(orbits),
        "orbit_size_histogram": {str(k): v for k, v in sorted(orbit_sizes.items())},
        "expected_rms_projected_white_noise_for_unit_signal": float(
            sigma * math.sqrt(rank)
        ),
        "finding": (
            "This is an exact orthogonal projection onto the 21-dimensional "
            "rotation-invariant subspace. The clean signal is generated inside "
            "that same subspace, so the selector's denoising advantage is the "
            "standard matched-subspace projection result, applied once—not a "
            "learned or multilevel natural-selection process."
        ),
    }


def vector_probe(module) -> dict[str, Any]:
    character_terms = []
    for rotation, perm in zip(module.MATS, module.PERMS):
        fixed_cells = int(np.sum(perm == np.arange(module.NC)))
        character_terms.append(fixed_cells * int(np.trace(rotation)))
    exact_rank = int(round(sum(character_terms) / len(character_terms)))

    rng = np.random.default_rng(8675309)
    A = rng.standard_normal((module.NC, 3))
    B = rng.standard_normal((module.NC, 3))
    PA = module.P_equiv(A)
    PB = module.P_equiv(B)
    self_adjoint_error = abs(float(np.vdot(A, PB) - np.vdot(PA, B)))
    idempotence_error = float(np.linalg.norm(module.P_equiv(PA) - PA))

    orbits = scalar_orbits(module.PERMS, module.NC)
    orbit_sizes = Counter(len(orbit) for orbit in orbits)
    return {
        "raw_dimension": module.NC * 3,
        "exact_equivariant_rank_by_character_trace": exact_rank,
        "archived_sample_rank_cap": 60,
        "idempotence_error": idempotence_error,
        "self_adjoint_inner_product_error": self_adjoint_error,
        "cell_orbits": len(orbits),
        "orbit_size_histogram": {str(k): v for k, v in sorted(orbit_sizes.items())},
        "finding": (
            "The 42-dimensional equivariant subspace is real and the projector is "
            "valid. But cells fall into 21 orbits of several sizes, not 24 copies "
            "for every cell. Projection corrects off-subspace stochastic corruption; "
            "arbitrary equivariant/in-subspace lies remain indistinguishable."
        ),
    }


def function_name_loads(path: Path, function_name: str, variable: str) -> int:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return sum(
                isinstance(child, ast.Name)
                and child.id == variable
                and isinstance(child.ctx, ast.Load)
                for child in ast.walk(node)
            )
    raise KeyError(f"{function_name} not found in {path}")


def economy_probe(source: Path, module) -> dict[str, Any]:
    members = np.arange(8)
    base_e = np.array([3.0, 0.94, 0.88, 0.82, 0.76, 0.70, 0.64, 0.58])
    obs_truth_first = np.array([0.0, 4.0, -2.0, 3.0, 8.0, -4.0, 1.0, 9.0])
    obs_truth_last = obs_truth_first[::-1].copy()
    result_a = module.run_region(members, obs_truth_first, base_e, rounds=60)
    result_b = module.run_region(members, obs_truth_last, base_e, rounds=60)
    trace_equal = all(
        np.array_equal(result_a[1][key], result_b[1][key])
        for key in ("coop", "fav_gini", "stable")
    )
    return {
        "obs_loads_inside_run_region": function_name_loads(
            source / "livnium_cube_economy.py", "run_region", "obs"
        ),
        "winner_with_first_observations": int(result_a[0]),
        "winner_with_reversed_observations": int(result_b[0]),
        "trajectory_exactly_equal_after_reversing_observations": trace_equal,
        "final_cooperation_metric": float(result_a[1]["coop"][-1]),
        "final_favor_gini": float(result_a[1]["fav_gini"][-1]),
        "finding": (
            "Reported values are not read anywhere in the election dynamics. "
            "Reversing them leaves the winner and every trajectory identical. "
            "The process is a rich-get-richer selection mechanism; information "
            "quality is measured only after the winner is fixed."
        ),
    }


def judge_probe(source: Path) -> dict[str, Any]:
    path = source / "livnium_cube_economy_judges.py"
    return {
        "judge_identity_loads_inside_run_region": function_name_loads(
            path, "run_region", "_j"
        ),
        "consensus_expression": "median(obs[cand])",
        "each_judge_vote_expression": (
            "anomaly * (1 + 0.1 * independent_normal_noise)"
        ),
        "finding": (
            "Judge identities, observations, histories, and beliefs are not used. "
            "Every judge emits the same global anomaly rule plus noise, so more "
            "judges only average that injected noise. This is a useful median-based "
            "outlier filter with unelectable monitors, not yet a judge community."
        ),
    }


def purge_and_silence_probe(source: Path) -> dict[str, Any]:
    visibility = {
        str(deletions): 1.0 - (1.0 - 0.30) ** deletions
        for deletions in (1, 2, 3, 5, 10)
    }
    silence_path = source / "livnium_purge_and_rest.py"
    return {
        "majority_without_anchor_detection_rule": "q_eff = 0.0 when p > .5",
        "visibility_reform_probability_at_visibility_point_3": visibility,
        "silence_sweep_time_variable_loads": function_name_loads(
            silence_path, "silence_sweep", "t"
        ),
        "silence_mask": "computed once from oracle-known noise levels",
        "voters_at_95pct_silence": 41 - int(round(0.95 * 41)),
        "hardcoded_below_quorum_error": 9.99,
        "finding": (
            "Majority failure and public deterrence are explicit transition rules: "
            "detection is set to zero and deletions directly convert remaining liars. "
            "The silence experiment has no rounds, cooldown, or staggering; it "
            "permanently removes the oracle-known noisiest voters for every replicate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    source = args.source.resolve()
    root = args.root.resolve()

    majority = load_module(source / "livnium_majority_capture.py", "audit_majority")
    same_layer = load_module(source / "livnium_same_layer.py", "audit_same_layer")
    selector = load_module(source / "livnium_selector_clean.py", "audit_selector")
    vector = load_module(source / "livnium_vector_decode.py", "audit_vector")
    economy = load_module(source / "livnium_cube_economy.py", "audit_economy")

    results = {
        "source": str(source),
        "exact_duplicates": exact_duplicate_audit(source, root),
        "anchor_boundary": anchor_boundary(),
        "majority_anchor": majority_anchor_probe(majority),
        "deterrence": deterrence_probe(),
        "shared_fate": shared_fate_probe(source),
        "same_layer": same_layer_probe(same_layer),
        "selector": selector_probe(selector),
        "vector": vector_probe(vector),
        "economy": economy_probe(source, economy),
        "judges": judge_probe(source),
        "purge_and_silence": purge_and_silence_probe(source),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
