"""
Power-law scaling helper for Livnium experiments.

Use two measured points (K0, A0) and (K1, A1) to estimate alpha and predict
the K needed to hit a target accuracy with the curve

    A(K) = A_inf - (A_inf - A0) * (K0 / K) ** alpha

You can treat K as any capacity knob (e.g., number of collapse layers,
hidden size, or basin count in a sweep).
"""

import argparse
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScalingResult:
    alpha: float
    k_target: float
    current_gap: float
    target_gap: float


def estimate_alpha(
    k0: float, acc0: float, k1: float, acc1: float, a_inf: float
) -> float:
    """Estimate alpha from two observations."""
    gap0 = a_inf - acc0
    gap1 = a_inf - acc1
    if gap0 <= 0 or gap1 <= 0:
        raise ValueError("Accuracies must be below A_inf to estimate alpha.")
    ratio = gap1 / gap0
    if ratio <= 0:
        raise ValueError("Gap ratio must be positive.")
    base = k0 / k1
    if base <= 0:
        raise ValueError("K values must be positive.")
    return math.log(ratio) / math.log(base)


def predict_k_target(
    k0: float, acc0: float, acc_target: float, alpha: float, a_inf: float
) -> ScalingResult:
    """Compute K_target given alpha."""
    if acc_target >= a_inf:
        raise ValueError("Target accuracy must be below A_inf.")
    current_gap = a_inf - acc0
    target_gap = a_inf - acc_target
    if current_gap <= 0 or target_gap <= 0:
        raise ValueError("Accuracies must be below A_inf to predict K.")
    k_target = k0 * (current_gap / target_gap) ** (1.0 / alpha)
    return ScalingResult(
        alpha=alpha,
        k_target=k_target,
        current_gap=current_gap,
        target_gap=target_gap,
    )


def run_cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Apply power-law scaling to estimate K needed for a target accuracy."
    )
    parser.add_argument("--k0", type=float, required=True, help="Baseline K (e.g., num_layers).")
    parser.add_argument("--acc0", type=float, required=True, help="Accuracy at K0 (0-1).")
    parser.add_argument("--k1", type=float, required=True, help="Second K.")
    parser.add_argument("--acc1", type=float, required=True, help="Accuracy at K1 (0-1).")
    parser.add_argument(
        "--target-acc",
        type=float,
        required=True,
        help="Desired accuracy to hit (0-1).",
    )
    parser.add_argument(
        "--a-inf",
        type=float,
        default=1.0,
        help="Asymptotic accuracy ceiling (default=1.0 for classification).",
    )
    args = parser.parse_args(argv)

    alpha = estimate_alpha(args.k0, args.acc0, args.k1, args.acc1, args.a_inf)
    result = predict_k_target(args.k0, args.acc0, args.target_acc, alpha, args.a_inf)

    print(f"Estimated alpha: {result.alpha:.4f}")
    print(f"K needed for target accuracy: {result.k_target:.2f}")
    print(f"Current gap: {result.current_gap:.4f}")
    print(f"Target gap: {result.target_gap:.4f}")


if __name__ == "__main__":
    run_cli()
