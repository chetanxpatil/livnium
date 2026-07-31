#!/usr/bin/env python3
"""Read-only probes for the archived Cube-and-Geometry companion experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def directional_probe() -> dict[str, object]:
    rows = []
    for n in range(3, 22, 2):
        m = (n - 1) // 2
        value = 1 + 6 * m + 12 * m**2 + 8 * m**3
        rows.append({"n": n, "m": m, "partition_total": value, "n_cubed": n**3})
    return {
        "odd_sizes_checked": len(rows),
        "all_match": all(row["partition_total"] == row["n_cubed"] for row in rows),
        "identity": "(2m+1)^3 = 1 + 6m + 12m^2 + 8m^3",
        "rows": rows,
    }


def geometry_probe(geometry) -> dict[str, object]:
    lap = geometry.L
    operator = np.eye(geometry.NC) + geometry.LAM * lap
    eigenvalues = np.linalg.eigvalsh(lap)

    true = geometry.smooth_field(1)
    fake = geometry.smooth_field(2)
    clean_decoded = geometry.geometric_decode(true)
    fake_decoded = geometry.geometric_decode(fake)

    block = np.array(
        [
            i
            for i, (x, y, z) in enumerate(geometry.COORDS)
            if x >= 1 and y >= 1 and z >= 1
        ]
    )
    local_reports = true.copy()
    local_reports[block] = true[block] + 4.0
    local_decoded = geometry.geometric_decode(local_reports)
    social_constructed = np.where(
        np.isin(np.arange(geometry.NC), block), local_reports, true
    )

    # A prior-mismatch check: clean checkerboard data have no corruption but
    # strongly violate the smoothness preference.
    checker = np.array(
        [(-1.0) ** (x + y + z) for x, y, z in geometry.COORDS],
        dtype=float,
    )
    checker /= np.linalg.norm(checker)
    checker_decoded = geometry.geometric_decode(checker)

    return {
        "cells": geometry.NC,
        "laplacian_symmetric": bool(np.allclose(lap, lap.T)),
        "laplacian_rank": int(np.linalg.matrix_rank(lap)),
        "laplacian_min_eigenvalue": float(eigenvalues.min()),
        "smoothing_operator_rank": int(np.linalg.matrix_rank(operator)),
        "smoothing_operator_dimension": geometry.NC,
        "generated_smooth_family_is_full_dimensional": bool(
            np.linalg.matrix_rank(operator) == geometry.NC
        ),
        "clean_smooth_truth_relative_decode_error": geometry.rel(clean_decoded, true),
        "clean_checkerboard_relative_decode_error": geometry.rel(
            checker_decoded, checker
        ),
        "local_block_cells": int(len(block)),
        "local_naive_relative_error": geometry.rel(local_reports, true),
        "local_geometry_relative_error": geometry.rel(local_decoded, true),
        "constructed_social_array_equals_naive_reports": bool(
            np.array_equal(social_constructed, local_reports)
        ),
        "fake_distance_to_true": geometry.rel(fake, true),
        "fake_decode_error_to_true": geometry.rel(fake_decoded, true),
        "fake_decoder_relative_change": geometry.rel(fake_decoded, fake),
    }


def autoencoder_probe(autoencoder, seeds: int, random_partitions: int) -> dict[str, object]:
    digits = autoencoder.load_digits()
    images = (
        digits.images[:, :7, :7]
        .reshape(len(digits.images), -1)
        .astype(float)
    )
    autoencoder.rng = np.random.default_rng(0)
    autoencoder.rng.shuffle(images)
    train, test = images[:1300], images[1300:]
    train = train - train.mean(0)
    test = test - test.mean(0)
    regions = autoencoder.directional_regions()
    sizes = np.bincount(regions)

    rows = []
    for seed in range(seeds):
        autoencoder.rng = np.random.default_rng(seed)
        directional = autoencoder.train_ae(
            train, test, regions, 9, nonlin=False, epochs=4000
        )
        random_scores = []
        for partition_seed in range(random_partitions):
            random_regions = autoencoder.random_regions(sizes, 100 + partition_seed)
            autoencoder.rng = np.random.default_rng(seed)
            random_scores.append(
                autoencoder.train_ae(
                    train,
                    test,
                    random_regions,
                    9,
                    nonlin=False,
                    epochs=4000,
                )
            )
        rows.append(
            {
                "model_seed": seed,
                "directional": float(directional),
                "random_mean": float(np.mean(random_scores)),
                "random_min": float(np.min(random_scores)),
                "random_max": float(np.max(random_scores)),
                "directional_minus_random_mean": float(
                    directional - np.mean(random_scores)
                ),
                "directional_wins_of_random_partitions": int(
                    sum(directional > score for score in random_scores)
                ),
                "random_scores": [float(score) for score in random_scores],
            }
        )
    all_differences = [
        row["directional"] - random_score
        for row in rows
        for random_score in row["random_scores"]
    ]
    return {
        "seeds": seeds,
        "random_partitions_per_seed": random_partitions,
        "pca_9": float(autoencoder.pca(train, test, 9)),
        "mean_directional": float(
            autoencoder.mean_pool_ls(train, test, regions, 9)
        ),
        "directional_mean": float(np.mean([row["directional"] for row in rows])),
        "random_mean": float(
            np.mean(
                [
                    score
                    for row in rows
                    for score in row["random_scores"]
                ]
            )
        ),
        "directional_minus_random_mean": float(np.mean(all_differences)),
        "directional_win_fraction": float(
            np.mean(np.array(all_differences) > 0)
        ),
        "rows": rows,
        "test_centered_using_its_own_mean": True,
    }


def rotation_architecture_probe(rotation) -> dict[str, object]:
    randomizer = np.random.default_rng(20260726)
    canonical_filters = {
        region: randomizer.standard_normal(49) * rotation.CMASK[region]
        for region in rotation.CANON.values()
    }
    encoder = rotation.tied_Wenc(canonical_filters)
    inverse_perm = np.argsort(rotation.PERM)

    # Find the output-code permutation induced by one input quarter-turn.
    transformed_rows = encoder[:, inverse_perm]
    code_permutation = []
    for row in transformed_rows:
        distances = np.linalg.norm(encoder - row, axis=1)
        code_permutation.append(int(np.argmin(distances)))
    code_permutation = np.array(code_permutation)

    x = randomizer.standard_normal(49)
    rotated_x = rotation.rot_vec(x, 1)
    code = encoder @ x
    rotated_code = encoder @ rotated_x
    encoder_weight_only_error = float(
        np.linalg.norm(rotated_code - code[code_permutation])
    )

    independent_bias = randomizer.standard_normal(9)
    biased_code = code + independent_bias
    biased_rotated_code = rotated_code + independent_bias
    encoder_with_independent_bias_error = float(
        np.linalg.norm(
            biased_rotated_code - biased_code[code_permutation]
        )
    )

    decoder = randomizer.standard_normal((49, 9))
    decoder_bias = randomizer.standard_normal(49)

    def reconstruct(vector: np.ndarray) -> np.ndarray:
        return (encoder @ vector + independent_bias) @ decoder.T + decoder_bias

    full_mapping_equivariance_error = float(
        np.linalg.norm(
            reconstruct(rotated_x) - rotation.rot_vec(reconstruct(x), 1)
        )
        / np.linalg.norm(reconstruct(x))
    )

    tied_total = 13 + 9 + 49 * 9 + 49
    untied_total = 49 + 9 + 49 * 9 + 49
    return {
        "weight_only_encoder_equivariance_error": encoder_weight_only_error,
        "encoder_error_after_independent_bias": encoder_with_independent_bias_error,
        "full_mapping_relative_equivariance_error_with_dense_decoder": full_mapping_equivariance_error,
        "reported_encoder_weight_counts": {"tied": 13, "untied": 49},
        "actual_total_parameter_counts_including_bias_and_dense_decoder": {
            "tied": tied_total,
            "untied": untied_total,
        },
        "total_parameter_reduction_pct": 100.0 * (untied_total - tied_total) / untied_total,
        "whole_autoencoder_is_guaranteed_equivariant": False,
        "reason": "encoder biases are untied across orbits and decoder weights/biases are dense and untied",
    }


def om_lo_algebra_probe(samples: int = 1000) -> dict[str, object]:
    randomizer = np.random.default_rng(20260726)
    maximum = 0.0
    for _ in range(samples):
        premise = randomizer.standard_normal(3)
        hypothesis = randomizer.standard_normal(3)
        pnorm = np.linalg.norm(premise)
        hnorm = np.linalg.norm(hypothesis)
        dot = float(premise @ hypothesis)
        cosine = dot / (pnorm * hnorm)
        shift = hypothesis - premise
        shift_norm = np.linalg.norm(shift)
        lo = np.array(
            [
                shift_norm,
                float(shift @ (-premise) / (shift_norm * pnorm)),
                float(hypothesis @ (-premise) / (hnorm * pnorm)),
            ]
        )
        reconstructed = np.array(
            [
                np.sqrt(pnorm**2 + hnorm**2 - 2 * dot),
                (pnorm**2 - dot) / (shift_norm * pnorm),
                -cosine,
            ]
        )
        maximum = max(maximum, float(np.max(np.abs(lo - reconstructed))))
    return {
        "samples": samples,
        "max_absolute_reconstruction_error": maximum,
        "lo_features_add_information_beyond_om_features": False,
        "identities": [
            "||H-P||^2 = ||P||^2 + ||H||^2 - 2 P.H",
            "cos(H-P,-P) = (||P||^2 - P.H) / (||H-P|| ||P||)",
            "cos(H,-P) = -cos(P,H)",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--ae-seeds", type=int, default=5)
    parser.add_argument("--ae-random-partitions", type=int, default=8)
    args = parser.parse_args()

    geometry = load_module(
        args.directory / "livnium_geometry_direct.py", "geometry_direct_audit"
    )
    autoencoder = load_module(
        args.directory / "livnium_learned_cube_ae.py", "learned_cube_ae_audit"
    )
    rotation = load_module(
        args.directory / "livnium_rotation_equivariant.py", "rotation_ae_audit"
    )

    result = {
        "directional_decomposition": directional_probe(),
        "geometry_direct": geometry_probe(geometry),
        "learned_cube_autoencoder": autoencoder_probe(
            autoencoder, args.ae_seeds, args.ae_random_partitions
        ),
        "rotation_architecture": rotation_architecture_probe(rotation),
        "om_lo_algebra": om_lo_algebra_probe(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
