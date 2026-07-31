#!/usr/bin/env python3
"""Read-only probes for ARCH_ARCHIVE_ROOT_AUDIT.md.

The default run verifies the missed branch mirrors, top-level artifacts,
Core-C's historical tests, and the complete archived market-data boundary.
Use --full-core-tests to replay the intentionally failing base-core suite.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd


ROOTS = {
    "A": Path(
        "/Users/chetanpatil/Desktop/test/lab/infected/python/"
        "clean-nova-livnium/archives-local/arch-archive"
    ),
    "B": Path(
        "/Users/chetanpatil/Desktop/test/lab/infected/python/"
        "clean=noba=back/arch-archive"
    ),
    "C": Path(
        "/Users/chetanpatil/Desktop/test/lab/infected/workspace/"
        "clean-nova-livnium/archives-local/arch-archive"
    ),
}

EXPECTED_COUNTS = {
    "brain": 26,
    "core": 136,
    "core-c": 8,
    "language": 1,
    "market-killer": 509,
}

EXPECTED_TOP_HASHES = {
    "important.md": "9bf465893c14371e2715f66776c459f3e733cea5505ed1b913f7288401a1f7d9",
    "snli_geometry.png": "333f321e8ef40d9b92f7510ff6a69074b0ecf30d7195067b273a1b870a9b04f3",
    "snli_geometry_brain_zoom.png": (
        "423f7d82589fd5967e32a614b0d387289ed39c5d7418daf84da120df36b1bf7b"
    ),
    "livnium_efficiency_dark.png": (
        "5cc3c04ef66ee069043d1ea76e2f1ce5eb0f0895fd0f2e3a1370bfcfda1c3a38"
    ),
    "livnium_efficiency_plot.py": (
        "38c792839b4d75f31263ec74064c5406d729da53fe19c7d045ae915576a3fde4"
    ),
}


def ignored(path: Path) -> bool:
    return (
        path.name == ".DS_Store"
        or path.suffix == ".pyc"
        or "__pycache__" in path.parts
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_fingerprint(root: Path, branch: str) -> tuple[int, str]:
    base = root / branch
    files = sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and not path.is_symlink() and not ignored(path)
    )
    digest = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix().encode()
        digest.update(rel)
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\n")
    return len(files), digest.hexdigest()


def verify_mirrors() -> None:
    print("== missed-branch mirrors ==")
    for label, root in ROOTS.items():
        assert root.is_dir(), root
        print(f"{label}: {root}")

    for branch, expected_count in EXPECTED_COUNTS.items():
        results: dict[str, tuple[int, str] | None] = {}
        for label, root in ROOTS.items():
            if not (root / branch).is_dir():
                results[label] = None
            else:
                results[label] = tree_fingerprint(root, branch)
        print(branch, results)

        if branch == "core":
            assert results["A"] is None
            assert results["B"] == results["C"]
            assert results["B"] is not None and results["B"][0] == expected_count
        else:
            assert results["A"] == results["B"] == results["C"]
            assert results["A"] is not None and results["A"][0] == expected_count

    print("== top-level hashes ==")
    for name, expected in EXPECTED_TOP_HASHES.items():
        values = {label: sha256_file(root / name) for label, root in ROOTS.items()}
        print(name, values)
        assert set(values.values()) == {expected}


def run_core_c_tests() -> None:
    print("== Core-C historical tests ==")
    source = ROOTS["C"] / "core-c"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NUMBA_DISABLE_JIT"] = "1"
    with tempfile.TemporaryDirectory(prefix="arch_archive_core_c_") as tmp:
        linked = Path(tmp) / "core_c"
        linked.symlink_to(source, target_is_directory=True)
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(linked / "tests" / "test_livnium_c.py"),
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        assert result.returncode == 0
        assert "11 passed" in result.stdout


def build_market_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    work = df.copy()
    work["return"] = work["Close"].pct_change()
    work["vol"] = work["return"].rolling(14).std()
    work["volume_z"] = (
        work["Volume"] - work["Volume"].rolling(14).mean()
    ) / work["Volume"].rolling(14).std()
    work["range"] = (work["High"] - work["Low"]) / work["Open"]
    work["ratio"] = work["Close"] / work["Open"] - 1
    feat = work.dropna().copy()
    columns = ["return", "vol", "volume_z", "range", "ratio"]
    values = feat[columns].to_numpy(float)
    values = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-8)
    return feat, values


def ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    output = []
    state = None
    for value in values:
        state = value.copy() if state is None else alpha * value + (1 - alpha) * state
        output.append(state.copy())
    return np.asarray(output)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 50:
        return float("nan")
    return float(np.corrcoef(left[mask], right[mask])[0, 1])


def probe_market() -> None:
    print("== market archive ==")
    market_root = ROOTS["C"] / "market-killer" / "market"
    csvs = sorted(market_root.glob("*.csv"))
    assert len(csvs) == 503
    assert not (market_root / "SPY.csv").exists()

    finite = 0
    nonfinite = 0
    evaluated: list[tuple[str, int, float, float, float]] = []
    first_dates = []
    last_dates = []

    for path in csvs:
        df = pd.read_csv(path)
        dates = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        first_dates.append(dates.min())
        last_dates.append(dates.max())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feat, values = build_market_features(df)

        if not np.isfinite(values).all():
            nonfinite += 1
            continue
        finite += 1
        if len(feat) < 100:
            continue

        states_ema = ema(values)
        current = values[20:]
        prior = states_ema[19:-1]
        current = current / (np.linalg.norm(current, axis=1, keepdims=True) + 1e-8)
        prior = prior / (np.linalg.norm(prior, axis=1, keepdims=True) + 1e-8)
        alignment = np.sum(current * prior, axis=1)
        tension = np.abs(0.38 - alignment)

        raw_return_full = df["Close"].pct_change()
        raw_volatility_full = raw_return_full.rolling(14).std()
        raw_return = raw_return_full.reindex(feat.index)
        raw_volatility = raw_volatility_full.reindex(feat.index)
        next_abs_return = raw_return.shift(-1).abs().to_numpy()[20:]
        abs_current_return = raw_return.abs().to_numpy()[20:]
        rolling_volatility = raw_volatility.to_numpy()[20:]

        tension_corr = correlation(tension, next_abs_return)
        return_corr = correlation(abs_current_return, next_abs_return)
        volatility_corr = correlation(rolling_volatility, next_abs_return)
        if np.isfinite(tension_corr):
            evaluated.append(
                (
                    path.stem,
                    len(df),
                    tension_corr,
                    return_corr,
                    volatility_corr,
                )
            )

    print(
        f"csv=503 finite={finite} nonfinite={nonfinite} "
        f"corrected_evaluable={len(evaluated)}"
    )
    assert finite == 321
    assert nonfinite == 182
    assert len(evaluated) == 320

    metrics = np.asarray([[row[2], row[3], row[4]] for row in evaluated])
    print(f"observation_range={min(first_dates)} .. {max(last_dates)}")
    for index, name in enumerate(
        ["tension", "abs_current_return", "rolling_volatility_14"]
    ):
        values = metrics[:, index]
        print(
            f"{name}: mean={np.mean(values):.4f} "
            f"median={np.median(values):.4f} "
            f"q25={np.percentile(values, 25):.4f} "
            f"q75={np.percentile(values, 75):.4f}"
        )

    by_symbol = {row[0]: row for row in evaluated}
    for symbol in ["AAPL", "TSLA", "MSFT"]:
        print(symbol, by_symbol[symbol])

    # Archived bug: the last shifted target is NaN and is passed unmasked.
    aapl = pd.read_csv(market_root / "AAPL.csv")
    feat, values = build_market_features(aapl)
    states_ema = ema(values)
    current = values[20:]
    prior = states_ema[19:-1]
    current = current / (np.linalg.norm(current, axis=1, keepdims=True) + 1e-8)
    prior = prior / (np.linalg.norm(prior, axis=1, keepdims=True) + 1e-8)
    tension = np.abs(0.38 - np.sum(current * prior, axis=1))
    unmasked_target = (
        aapl["Close"].pct_change().reindex(feat.index).shift(-1).abs().to_numpy()[20:]
    )
    archived_corr = np.corrcoef(tension, unmasked_target)[0, 1]
    assert np.isnan(archived_corr)
    print("archived_AAPL_corr=nan (unmasked shifted target)")

    # For C=0.38 and alignment in [0.4, 1], tension is at most 0.62.
    max_euphoria_tension = max(abs(0.38 - 0.4), abs(0.38 - 1.0))
    assert max_euphoria_tension < 0.8
    print(
        "euphoria_unreachable: "
        f"max_tension_when_alignment>=0.4 is {max_euphoria_tension:.2f} < 0.80"
    )


def run_full_core_tests() -> None:
    print("== full historical base-core pytest ==")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NUMBA_DISABLE_JIT"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOTS["C"] / "core"),
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)
    assert result.returncode != 0
    assert "25 failed" in result.stdout
    assert "252 passed" in result.stdout
    assert "6 errors" in result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-core-tests",
        action="store_true",
        help="also replay the known-failing 283-test/collection base-core surface",
    )
    args = parser.parse_args()

    verify_mirrors()
    run_core_c_tests()
    probe_market()
    if args.full_core_tests:
        run_full_core_tests()
    print("ARCH_ARCHIVE_ROOT_AUDIT_PROBE: PASS")


if __name__ == "__main__":
    main()
