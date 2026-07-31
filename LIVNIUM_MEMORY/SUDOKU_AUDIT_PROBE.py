#!/usr/bin/env python3
"""Read-only reproducibility probes for the archived Livnium Sudoku family.

This script imports the historical modules without calling their ``main``
functions, never writes into the source tree, and prints one JSON evidence
record.  It is intentionally independent of the historical verdict documents.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import binomtest


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_complete_9x9(grid: np.ndarray) -> bool:
    target = set(range(1, 10))
    if grid.shape != (9, 9):
        return False
    for i in range(9):
        if set(map(int, grid[i])) != target:
            return False
        if set(map(int, grid[:, i])) != target:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if set(map(int, grid[br : br + 3, bc : bc + 3].ravel())) != target:
                return False
    return True


def candidates_9x9(grid: np.ndarray, r: int, c: int) -> list[int]:
    used = set(map(int, grid[r]))
    used.update(map(int, grid[:, c]))
    used.update(map(int, grid[3 * (r // 3) : 3 * (r // 3) + 3, 3 * (c // 3) : 3 * (c // 3) + 3].ravel()))
    return [d for d in range(1, 10) if d not in used]


def count_solutions_9x9(puzzle: np.ndarray, cap: int = 2) -> int:
    """Independent MRV solution counter, stopping at ``cap``."""
    grid = puzzle.copy()
    count = 0

    def visit() -> None:
        nonlocal count
        if count >= cap:
            return
        best = None
        best_candidates: list[int] | None = None
        for r, c in np.argwhere(grid == 0):
            rr, cc = int(r), int(c)
            cand = candidates_9x9(grid, rr, cc)
            if not cand:
                return
            if best_candidates is None or len(cand) < len(best_candidates):
                best = (rr, cc)
                best_candidates = cand
                if len(cand) == 1:
                    break
        if best is None:
            if valid_complete_9x9(grid):
                count += 1
            return
        r, c = best
        assert best_candidates is not None
        for d in best_candidates:
            grid[r, c] = d
            visit()
            grid[r, c] = 0
            if count >= cap:
                return

    visit()
    return count


def ast_findings(paths: dict[str, Path]) -> dict[str, object]:
    findings: dict[str, object] = {}
    for name, path in paths.items():
        tree = ast.parse(path.read_text())
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        defs = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        findings[name] = {
            "function_defs": defs,
            "symmetry_variant_call_count": calls.count("symmetry_variant"),
            "discounted_returns_call_count": calls.count("discounted_returns"),
        }
    return findings


def puzzle_generation_probe(pure, generator: str, per_level: int) -> dict[str, object]:
    pure.rng = np.random.default_rng(20260726)
    out: dict[str, object] = {}
    # 51/36/26 reproduce the original script's 30/45/55-hole labels;
    # 40/32/26/23 reproduce the later pure/hybrid labels.
    for givens in (51, 40, 36, 32, 26, 23):
        counts = []
        forced_fractions = []
        random_legal_expected = []
        smallest_legal_correct = []
        unique_solutions = set()
        for _ in range(per_level):
            solution = pure.full_solution(generator=generator)
            unique_solutions.add(solution.tobytes())
            puzzle = pure.make_puzzle(solution, givens)
            counts.append(count_solutions_9x9(puzzle, cap=2))
            forced = 0
            expected = []
            smallest = []
            for r, c in np.argwhere(puzzle == 0):
                rr, cc = int(r), int(c)
                cand = candidates_9x9(puzzle, rr, cc)
                if len(cand) == 1:
                    forced += 1
                expected.append(1.0 / len(cand))
                smallest.append(int(cand[0] == int(solution[rr, cc])))
            forced_fractions.append(forced / max(1, len(expected)))
            random_legal_expected.append(float(np.mean(expected)))
            smallest_legal_correct.append(float(np.mean(smallest)))
        out[str(givens)] = {
            "puzzles": per_level,
            "distinct_generated_solutions": len(unique_solutions),
            "unique_puzzle_rate_pct": 100.0 * sum(c == 1 for c in counts) / per_level,
            "multiple_solution_rate_pct": 100.0 * sum(c >= 2 for c in counts) / per_level,
            "mean_initial_naked_single_fraction_pct": 100.0 * float(np.mean(forced_fractions)),
            "mean_random_legal_digit_expected_accuracy_pct": 100.0 * float(np.mean(random_legal_expected)),
            "mean_smallest_legal_digit_accuracy_pct": 100.0 * float(np.mean(smallest_legal_correct)),
        }
    return out


def original_model_probe(original, held_out_solutions: int = 20, masks_per_solution: int = 4) -> dict[str, object]:
    """Replay the original training score and measure a genuinely fresh cell split."""
    original.rng = np.random.default_rng(0)
    x_train, y_train = original.make_training()
    model = original.MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=120,
        random_state=0,
    ).fit(x_train, y_train)
    train_accuracy = float(model.score(x_train, y_train))

    correct = 0
    total = 0
    legal_correct = 0
    legal_total = 0
    for _ in range(held_out_solutions):
        solution = original.random_solution()
        for _ in range(masks_per_solution):
            keep = original.rng.random((9, 9)) < original.rng.uniform(0.3, 0.8)
            grid = np.where(keep, solution, 0)
            wrapped = original.wrap(grid)
            for r, c in np.argwhere(grid == 0):
                rr, cc = int(r), int(c)
                probabilities = model.predict_proba(original.cell_feat(wrapped, rr, cc)[None])[0]
                predicted = int(probabilities.argmax()) + 1
                correct += int(predicted == int(solution[rr, cc]))
                total += 1
                cand = candidates_9x9(grid, rr, cc)
                legal_prediction = max(cand, key=lambda d: probabilities[d - 1])
                legal_correct += int(legal_prediction == int(solution[rr, cc]))
                legal_total += 1
    return {
        "training_examples": int(len(x_train)),
        "reported_held_out_style_is_training_resubstitution": True,
        "training_resubstitution_cell_accuracy_pct": 100.0 * train_accuracy,
        "fresh_solutions": held_out_solutions,
        "fresh_masks_per_solution": masks_per_solution,
        "fresh_unmasked_argmax_cell_accuracy_pct": 100.0 * correct / total,
        "fresh_legality_masked_argmax_cell_accuracy_pct": 100.0 * legal_correct / legal_total,
    }


def hybrid_solve_stats(hybrid, grid: np.ndarray, ordering: str, model=None, randomizer=None):
    """Independent wrapper measuring attempts and actual failed branches."""
    counters = {"attempts": 0, "failed_branches": 0}

    def visit(current: np.ndarray):
        current = current.copy()
        if not hybrid.propagate(current):
            return None
        empties = [(r, c) for r in range(9) for c in range(9) if current[r, c] == 0]
        if not empties:
            return current
        r, c = min(empties, key=lambda rc: len(hybrid.candidates(current, *rc)))
        cand = set(hybrid.candidates(current, r, c))
        if ordering == "ascending":
            order = sorted(cand)
        elif ordering == "learned":
            p = model.predict_proba(hybrid.present_feat(current, r, c)[None])[0]
            order = sorted(cand, key=lambda d: -p[d - 1])
        elif ordering == "lcv":
            peers = set((r, j) for j in range(9))
            peers.update((i, c) for i in range(9))
            peers.update(hybrid.units_of(r, c))

            def eliminations(d: int) -> int:
                total = 0
                for rr, cc in peers:
                    if current[rr, cc] == 0 and (rr, cc) != (r, c):
                        total += int(d in hybrid.candidates(current, rr, cc))
                return total

            order = sorted(cand, key=lambda d: (eliminations(d), d))
        elif ordering == "random":
            order = list(cand)
            randomizer.shuffle(order)
        else:
            raise ValueError(ordering)
        for d in order:
            counters["attempts"] += 1
            child = current.copy()
            child[r, c] = d
            solved = visit(child)
            if solved is not None:
                return solved
            counters["failed_branches"] += 1
        return None

    return visit(grid), counters


def bootstrap_mean_difference_ci(differences: np.ndarray, seed: int = 20260726) -> list[float]:
    randomizer = np.random.default_rng(seed)
    draws = randomizer.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def hybrid_replay(hybrid, per_level: int) -> dict[str, object]:
    hybrid.rng = np.random.default_rng(0)
    model = hybrid.train_model()
    out: dict[str, object] = {}
    levels = {"easy_40": 40, "medium_32": 32, "hard_26": 26, "expert_23": 23}
    randomizer = np.random.default_rng(20260726)
    for label, givens in levels.items():
        rows = []
        for _ in range(per_level):
            source_solution = hybrid.full_solution()
            puzzle = hybrid.make_puzzle(source_solution, givens)
            uniqueness = count_solutions_9x9(puzzle, cap=2)
            row: dict[str, object] = {"solution_count_capped_2": uniqueness}
            for ordering in ("ascending", "learned", "lcv", "random"):
                solved, counters = hybrid_solve_stats(
                    hybrid,
                    puzzle,
                    ordering,
                    model=model,
                    randomizer=randomizer,
                )
                row[ordering] = counters
                row[f"{ordering}_valid"] = bool(
                    solved is not None
                    and valid_complete_9x9(solved)
                    and np.array_equal(solved[puzzle > 0], puzzle[puzzle > 0])
                )
            rows.append(row)

        asc = np.array([int(row["ascending"]["attempts"]) for row in rows])
        learned = np.array([int(row["learned"]["attempts"]) for row in rows])
        lcv = np.array([int(row["lcv"]["attempts"]) for row in rows])
        random_attempts = np.array([int(row["random"]["attempts"]) for row in rows])
        differences = asc - learned
        non_ties = differences[differences != 0]
        learned_wins = int((differences > 0).sum())
        asc_wins = int((differences < 0).sum())
        sign_p = float(binomtest(min(learned_wins, asc_wins), len(non_ties), 0.5).pvalue) if len(non_ties) else 1.0
        out[label] = {
            "puzzles": per_level,
            "all_solver_outputs_valid": all(
                bool(row[f"{ordering}_valid"])
                for row in rows
                for ordering in ("ascending", "learned", "lcv", "random")
            ),
            "multiple_solution_rate_pct": 100.0
            * sum(int(row["solution_count_capped_2"]) >= 2 for row in rows)
            / per_level,
            "attempt_means": {
                "ascending": float(asc.mean()),
                "learned": float(learned.mean()),
                "lcv": float(lcv.mean()),
                "random_one_draw": float(random_attempts.mean()),
            },
            "actual_failed_branch_means": {
                ordering: float(np.mean([int(row[ordering]["failed_branches"]) for row in rows]))
                for ordering in ("ascending", "learned", "lcv", "random")
            },
            "learned_vs_ascending": {
                "learned_wins": learned_wins,
                "ties": int((differences == 0).sum()),
                "ascending_wins": asc_wins,
                "mean_attempt_reduction_ascending_minus_learned": float(differences.mean()),
                "bootstrap_95pct_ci_for_mean_reduction": bootstrap_mean_difference_ci(differences),
                "two_sided_exact_sign_test_p": sign_p,
            },
            "learned_vs_lcv_mean_attempt_difference": float(learned.mean() - lcv.mean()),
        }
    return out


def rl_replay(rl, held_out: int, episodes: int) -> dict[str, object]:
    rl.rng = np.random.default_rng(0)
    training = [rl.gen_puzzle(6) for _ in range(3)]
    q_table, solves = rl.train(training, episodes=episodes)
    training_success = sum(rl.greedy_trace(q_table, *puzzle)[1] for puzzle in training)
    held = [rl.gen_puzzle(6) for _ in range(held_out)]
    initial_states_seen = sum(puzzle[0].tobytes() in q_table for puzzle in held)
    held_success = sum(rl.greedy_trace(q_table, *puzzle)[1] for puzzle in held)
    window = min(500, len(solves))
    return {
        "episodes": episodes,
        "q_states": len(q_table),
        "training_puzzles": 3,
        "training_greedy_solved": training_success,
        "last_window_training_solve_rate_pct": 100.0 * float(np.mean(solves[-window:])),
        "held_out_puzzles": held_out,
        "held_out_initial_states_seen_in_q": initial_states_seen,
        "held_out_greedy_solved": held_success,
        "action_count_implemented": 64,
        "docstring_claimed_action_count": 80,
    }


def policy_artifact_probe(policy_json: Path) -> dict[str, object]:
    artifact = json.loads(policy_json.read_text())
    example = artifact["example"]
    puzzle = np.array(example["puzzle"])
    trace = np.array(example["trace"])
    solution = np.array(example["solution"])
    return {
        "training_curve_length": len(artifact["train_curve"]),
        "training_curve_sum": int(sum(artifact["train_curve"])),
        "example_steps": int(example["steps"]),
        "initial_holes": int((puzzle == 0).sum()),
        "cells_changed_from_puzzle": int((trace != puzzle).sum()),
        "final_cells_correct_beyond_givens": int(((trace == solution) & (puzzle == 0)).sum()),
        "final_grid_equals_solution": bool(np.array_equal(trace, solution)),
        "source_mechanism": "a wrong greedy action leaves state unchanged, so deterministic argmax repeats until max_steps",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sudoku-dir", type=Path, required=True)
    parser.add_argument("--generation-samples", type=int, default=20)
    parser.add_argument("--hybrid-samples", type=int, default=45)
    parser.add_argument("--rl-held-out", type=int, default=100)
    parser.add_argument("--rl-episodes", type=int, default=30000)
    parser.add_argument("--include-original-model", action="store_true")
    args = parser.parse_args()

    paths = {
        "original": args.sudoku_dir / "livnium_sudoku.py",
        "pure": args.sudoku_dir / "livnium_sudoku_pure.py",
        "hybrid": args.sudoku_dir / "livnium_sudoku_hybrid.py",
        "rl": args.sudoku_dir / "livnium_sudoku_rl.py",
        "policy_rl": args.sudoku_dir / "livnium_sudoku_policy_rl.py",
    }
    pure = load_module(paths["pure"], "sudoku_pure_audit")
    hybrid = load_module(paths["hybrid"], "sudoku_hybrid_audit")
    rl = load_module(paths["rl"], "sudoku_rl_audit")
    original = load_module(paths["original"], "sudoku_original_audit")

    record = {
        "probe_version": 1,
        "ast": ast_findings(paths),
        "puzzle_generation": {
            generator: puzzle_generation_probe(pure, generator, args.generation_samples)
            for generator in ("builtin", "py-sudoku")
        },
        "hybrid_replay": hybrid_replay(hybrid, args.hybrid_samples),
        "tabular_rl_replay": rl_replay(rl, args.rl_held_out, args.rl_episodes),
        "policy_rl_saved_artifact": policy_artifact_probe(
            args.sudoku_dir / "livnium_sudoku_policy_rl_g60_e5000.json"
        ),
    }
    if args.include_original_model:
        record["original_model_replay"] = original_model_probe(original)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
