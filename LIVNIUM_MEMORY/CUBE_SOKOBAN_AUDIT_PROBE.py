#!/usr/bin/env python3
"""Read-only probes for the archived Cube/Sokoban symmetry experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("cube_sokoban_audit_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def group_probe(perms: list[np.ndarray]) -> dict[str, object]:
    keys = {tuple(map(int, p)) for p in perms}
    closure_failures = 0
    for left in perms:
        for right in perms:
            composed = tuple(map(int, right[left]))
            if composed not in keys:
                closure_failures += 1
    return {
        "count": len(perms),
        "unique": len(keys),
        "all_bijections": all(np.array_equal(np.sort(p), np.arange(len(p))) for p in perms),
        "closure_failures_of_576": closure_failures,
    }


def nearest_neighbor(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    predictions = []
    for row in test_x:
        distances = np.count_nonzero(train_x != row, axis=1)
        predictions.append(int(train_y[int(np.argmin(distances))]))
    return np.array(predictions)


def mlp_result(source, worlds, rotations, canonical: bool, random_state: int = 0):
    x_train, y_train = source.build(worlds, rotations, c=canonical)
    model = source.MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=400,
        random_state=random_state,
    ).fit(x_train, y_train)
    return model, x_train, y_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model-seeds", type=int, default=10)
    args = parser.parse_args()

    source = load_module(args.source)
    source.rng = np.random.default_rng(0)
    worlds = source.make_worlds(source.C)
    seen = [0, 1, 2, 3, 4, 5]
    unseen = [i for i in range(24) if i not in seen]

    raw_seen_x, raw_seen_y = source.build(worlds, seen, c=False)
    raw_unseen_x, raw_unseen_y = source.build(worlds, unseen, c=False)
    can_seen_x, can_seen_y = source.build(worlds, seen, c=True)
    can_unseen_x, can_unseen_y = source.build(worlds, unseen, c=True)

    train_keys = {row.tobytes() for row in can_seen_x}
    key_to_labels: dict[bytes, set[int]] = {}
    for row, label in zip(can_seen_x, can_seen_y):
        key_to_labels.setdefault(row.tobytes(), set()).add(int(label))
    hash_predictions = np.array(
        [next(iter(key_to_labels[row.tobytes()])) for row in can_unseen_x]
    )

    # Exact historical replay.
    replay = {}
    for name, rotations, canonical in (
        ("plain_1", [0], False),
        ("plain_6", seen, False),
        ("canonical_6", seen, True),
    ):
        model, _, _ = mlp_result(source, worlds, rotations, canonical)
        seen_x = can_seen_x if canonical else raw_seen_x
        unseen_x = can_unseen_x if canonical else raw_unseen_x
        seen_y = can_seen_y if canonical else raw_seen_y
        unseen_y = can_unseen_y if canonical else raw_unseen_y
        replay[name] = {
            "seen_pct": 100.0 * float(model.score(seen_x, seen_y)),
            "unseen_pct": 100.0 * float(model.score(unseen_x, unseen_y)),
        }

    # The verdict says one effective canonical view. Test literal one-row/class
    # training, then the source's six exact duplicates per class.
    canonical_one_model, can_one_x, can_one_y = mlp_result(source, worlds, [0], True)
    plain_one_x, plain_one_y = source.build(worlds, [0], c=False)
    plain_one_repeated_x = np.repeat(plain_one_x, 6, axis=0)
    plain_one_repeated_y = np.repeat(plain_one_y, 6, axis=0)
    plain_repeated_model = source.MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=400,
        random_state=0,
    ).fit(plain_one_repeated_x, plain_one_repeated_y)

    seed_results = []
    for seed in range(args.model_seeds):
        plain_model, _, _ = mlp_result(source, worlds, seen, False, seed)
        canonical_model, _, _ = mlp_result(source, worlds, seen, True, seed)
        seed_results.append(
            {
                "seed": seed,
                "plain_unseen_pct": 100.0
                * float(plain_model.score(raw_unseen_x, raw_unseen_y)),
                "canonical_unseen_pct": 100.0
                * float(canonical_model.score(can_unseen_x, can_unseen_y)),
            }
        )

    # A standard orbit-aware template baseline: canonicalize one template per
    # class and use exact lookup/nearest neighbor. No MLP is needed.
    canonical_template_x, canonical_template_y = source.build(worlds, [0], c=True)
    canonical_nn_predictions = nearest_neighbor(
        canonical_template_x, canonical_template_y, can_unseen_x
    )
    raw_nn_predictions = nearest_neighbor(raw_seen_x, raw_seen_y, raw_unseen_x)

    # Standard raw augmentation control: train on 23 orientations and withhold
    # one. This is not cheaper than canonicalization, but bounds the claim that
    # a plain model necessarily needs the exact view.
    leave_one_out_train = list(range(23))
    leave_one_out_test = [23]
    loo_model, _, _ = mlp_result(source, worlds, leave_one_out_train, False)
    loo_test_x, loo_test_y = source.build(worlds, leave_one_out_test, c=False)

    result = {
        "group": group_probe(source.PERMS),
        "dataset": {
            "worlds": len(worlds),
            "cells": source.NCELL,
            "walls_per_world": source.NWALL,
            "raw_train_rows": len(raw_seen_x),
            "raw_test_rows": len(raw_unseen_x),
            "canonical_train_rows": len(can_seen_x),
            "canonical_train_unique_rows": len(train_keys),
            "canonical_test_rows": len(can_unseen_x),
            "canonical_test_rows_exactly_present_in_train_pct": 100.0
            * sum(row.tobytes() in train_keys for row in can_unseen_x)
            / len(can_unseen_x),
            "canonical_keys_with_conflicting_labels": sum(
                len(labels) > 1 for labels in key_to_labels.values()
            ),
            "all_24_rotations_share_one_canonical_form_per_world": all(
                len(
                    {
                        source.canon(world[source.PERMS[index]]).tobytes()
                        for index in range(24)
                    }
                )
                == 1
                for world in worlds
            ),
        },
        "historical_replay": replay,
        "literal_one_view_mlp": {
            "canonical_train_rows": len(can_one_x),
            "canonical_seen_pct": 100.0
            * float(canonical_one_model.score(can_seen_x, can_seen_y)),
            "canonical_unseen_pct": 100.0
            * float(canonical_one_model.score(can_unseen_x, can_unseen_y)),
            "plain_one_view_repeated_six_times_seen_pct": 100.0
            * float(plain_repeated_model.score(raw_seen_x, raw_seen_y)),
            "plain_one_view_repeated_six_times_unseen_pct": 100.0
            * float(plain_repeated_model.score(raw_unseen_x, raw_unseen_y)),
        },
        "nonlearned_controls": {
            "canonical_hash_lookup_unseen_pct": 100.0
            * float(np.mean(hash_predictions == can_unseen_y)),
            "canonical_one_template_1nn_unseen_pct": 100.0
            * float(np.mean(canonical_nn_predictions == can_unseen_y)),
            "raw_six_rotation_1nn_unseen_pct": 100.0
            * float(np.mean(raw_nn_predictions == raw_unseen_y)),
        },
        "plain_23_rotations_test_1_unseen_pct": 100.0
        * float(loo_model.score(loo_test_x, loo_test_y)),
        "model_seed_results": seed_results,
        "model_seed_summary": {
            "plain_unseen_mean_pct": float(
                np.mean([row["plain_unseen_pct"] for row in seed_results])
            ),
            "plain_unseen_min_pct": float(
                np.min([row["plain_unseen_pct"] for row in seed_results])
            ),
            "plain_unseen_max_pct": float(
                np.max([row["plain_unseen_pct"] for row in seed_results])
            ),
            "canonical_unseen_mean_pct": float(
                np.mean([row["canonical_unseen_pct"] for row in seed_results])
            ),
        },
        "task_boundary": {
            "has_player": False,
            "has_crates": False,
            "has_goals": False,
            "has_moves_or_reachability": False,
            "label_is_world_identity": True,
            "same_underlying_worlds_in_train_and_test": True,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
