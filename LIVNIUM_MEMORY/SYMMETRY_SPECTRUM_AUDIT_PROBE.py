#!/usr/bin/env python3
"""Read-only audit probe for the 7x7x7 symmetry-spectrum experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True

DEFAULT_SOURCE = Path(
    "/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Symmetry"
)
DEFAULT_ROOT = Path("/Users/chetanpatil/Desktop/test")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("symmetry_archive", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(values: np.ndarray, tol: float = 1e-6):
    order = np.argsort(values)
    output = []
    current = [int(order[0])]
    last = float(values[order[0]])
    for idx in order[1:]:
        value = float(values[idx])
        if value - last <= tol:
            current.append(int(idx))
        else:
            output.append(current)
            current = [int(idx)]
        last = value
    output.append(current)
    return output


def signed_axis_matrices():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            M = np.zeros((3, 3), dtype=int)
            for row, col in enumerate(perm):
                M[row, col] = signs[row]
            mats.append(M)
    return mats


def coordinate_permutation(module, M):
    perm = np.empty(module.NC, dtype=int)
    for i, coordinate in enumerate(module.COORDS):
        transformed = tuple(int(x) for x in M @ np.asarray(coordinate))
        perm[i] = module.IDX[transformed]
    return perm


def connected_random_laplacian(n: int, n_edges: int, seed: int):
    """Uniform-ish connected control: random spanning tree, then random edges."""
    rng = np.random.default_rng(seed)
    edges = set()
    order = rng.permutation(n)
    for k in range(1, n):
        i = int(order[k])
        j = int(order[rng.integers(0, k)])
        edges.add(tuple(sorted((i, j))))
    while len(edges) < n_edges:
        i, j = (int(v) for v in rng.integers(0, n, 2))
        if i != j:
            edges.add(tuple(sorted((i, j))))
    L = np.zeros((n, n))
    for i, j in edges:
        L[i, j] = L[j, i] = -1
        L[i, i] += 1
        L[j, j] += 1
    return L


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    source = args.source.resolve()
    root = args.root.resolve()
    module = load_module(source / "livnium_symmetry_spectrum.py")

    duplicate_rows = []
    for path in sorted(source.iterdir()):
        if path.is_file():
            mirror = root / path.name
            duplicate_rows.append(
                {
                    "file": path.name,
                    "sha256": sha256(path),
                    "root_exact": mirror.is_file() and sha256(mirror) == sha256(path),
                }
            )

    L = module.cube_laplacian()
    numerical = np.linalg.eigvalsh(L)
    one_d = 2 - 2 * np.cos(np.arange(module.N) * np.pi / module.N)
    triples = list(itertools.product(range(module.N), repeat=3))
    analytic = np.array([one_d[i] + one_d[j] + one_d[k] for i, j, k in triples])
    spectrum_groups = groups(analytic)

    collision_rows = []
    for group in spectrum_groups:
        unordered = sorted({tuple(sorted(triples[index])) for index in group})
        if len(unordered) > 1:
            collision_rows.append(
                {
                    "multiplicity": len(group),
                    "unordered_triples": [list(row) for row in unordered],
                }
            )

    matrices = signed_axis_matrices()
    proper = [M for M in matrices if round(np.linalg.det(M)) == 1]
    improper = [M for M in matrices if round(np.linalg.det(M)) == -1]

    def max_commutator(mats):
        errors = []
        for M in mats:
            perm = coordinate_permutation(module, M)
            errors.append(float(np.linalg.norm(L[np.ix_(perm, perm)] - L)))
        return max(errors)

    module.rng = np.random.default_rng(0)
    n_edges = int(np.count_nonzero(np.triu(L == -1, 1)))
    archived_random = module.random_laplacian(n_edges)
    archived_random_evals = np.linalg.eigvalsh(archived_random)
    archived_random_groups = groups(archived_random_evals)
    components = int(np.sum(np.abs(archived_random_evals) < 1e-8))

    connected_counts = []
    for seed in range(10):
        evals = np.linalg.eigvalsh(
            connected_random_laplacian(module.NC, n_edges, seed)
        )
        connected_counts.append(len(groups(evals)))

    anisotropic = np.array(
        [
            one_d[i] + math.sqrt(2) * one_d[j] + math.pi * one_d[k]
            for i, j, k in triples
        ]
    )

    multiplicities = Counter(len(group) for group in spectrum_groups)
    result = {
        "source": str(source),
        "duplicates": {
            "canonical_files": len(duplicate_rows),
            "exact_root_copies": sum(row["root_exact"] for row in duplicate_rows),
            "rows": duplicate_rows,
        },
        "cube": {
            "nodes": module.NC,
            "edges": n_edges,
            "analytic_numeric_max_abs_error": float(
                np.max(np.abs(np.sort(analytic) - numerical))
            ),
            "distinct_speeds": len(spectrum_groups),
            "multiplicity_histogram": {
                str(k): v for k, v in sorted(multiplicities.items())
            },
            "unordered_axis_triples": math.comb(module.N + 2, 3),
            "extra_sum_collision_groups": len(collision_rows),
            "collision_rows": collision_rows,
            "proper_rotation_commutator_max": max_commutator(proper),
            "improper_reflection_commutator_max": max_commutator(improper),
            "proper_symmetries_checked": len(proper),
            "improper_symmetries_checked": len(improper),
        },
        "controls": {
            "archived_random_distinct": len(archived_random_groups),
            "archived_random_components": components,
            "archived_random_max_multiplicity": max(
                len(group) for group in archived_random_groups
            ),
            "connected_random_distinct_10_seeds": connected_counts,
            "anisotropic_separable_distinct": len(groups(anisotropic)),
        },
        "interpretation": [
            "The 70-level spectrum is exact for the Cartesian product P7 x P7 x P7.",
            "There are 84 unordered axis-mode triples before extra eigenvalue-sum collisions.",
            "Both 24 proper rotations and 24 improper signed-axis symmetries commute with L.",
            "Multiplicities 15 and 18 are not dimensions of cube-group irreducible representations.",
            "The archived random graph's largest multiplicity is its five connected components at eigenvalue zero.",
            "Breaking equality between axis operators makes all 343 separable sums distinct at tolerance 1e-6.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
