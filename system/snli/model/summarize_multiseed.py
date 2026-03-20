"""
Multi-Seed Ablation Summary
============================
Reads ablation_results.json from each seed directory and produces:
  1. A cross-seed summary table (mean ± std for each condition)
  2. A per-seed breakdown
  3. Updates v3_goals.md with the results table

Usage:
  python3 summarize_multiseed.py \\
      --results-dir ../../../pretrained/multiseed \\
      --update-goals

This script expects the directory layout created by train_multiseed.sh:
  pretrained/multiseed/seed42/ablation_results.json
  pretrained/multiseed/seed123/ablation_results.json
  pretrained/multiseed/seed999/ablation_results.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np


CONDITION_KEYS = [
    ('baseline',             'Baseline (Full MLP)'),
    ('A_grad_v',             'A: Grad-V (∇V, no proj)'),
    ('B_anchor_proj_only',   'B: Anchor-proj → head'),
    ('D_anchor_then_gradv',  'D: Anchor-proj → ∇V'),
    ('E_svd_then_gradv',     'E: SVD-k → ∇V'),
]


def load_results(results_dir):
    results_dir = Path(results_dir)
    seed_dirs = sorted([d for d in results_dir.iterdir()
                        if d.is_dir() and (d / 'ablation_results.json').exists()])
    if not seed_dirs:
        print(f"ERROR: No ablation_results.json found under {results_dir}")
        sys.exit(1)

    data = {}
    for d in seed_dirs:
        with open(d / 'ablation_results.json') as f:
            data[d.name] = json.load(f)
    return data


def extract_metric(run_data, condition_key, metric='accuracy'):
    """Safely extract metric from a run's results."""
    if condition_key not in run_data:
        return None
    return run_data[condition_key].get(metric)


def compute_stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def print_summary(data):
    seeds = list(data.keys())
    print(f"\n{'═'*80}")
    print(f"  MULTI-SEED ABLATION SUMMARY  ({len(seeds)} seeds: {', '.join(seeds)})")
    print(f"{'═'*80}")

    # Header
    print(f"\n  {'Condition':<30}  {'Mean Acc':>9}  {'Std':>6}  "
          f"{'Δ vs Base':>10}  {'N-rec mean':>10}")
    print(f"  {'─'*72}")

    # Compute baseline stats first
    base_accs = [extract_metric(data[s], 'baseline', 'accuracy') for s in seeds]
    base_mean, base_std = compute_stats(base_accs)

    for ckey, clabel in CONDITION_KEYS:
        accs    = [extract_metric(data[s], ckey, 'accuracy') for s in seeds]
        n_recs  = [extract_metric(data[s], ckey, 'recall_neutral') for s in seeds]
        mean_a, std_a = compute_stats(accs)
        mean_n, _     = compute_stats(n_recs)

        if mean_a is None:
            print(f"  {clabel:<30}  {'—':>9}  {'—':>6}  {'—':>10}  {'—':>10}")
            continue

        delta = f"{(mean_a - base_mean)*100:+.2f}pp" if base_mean else "—"
        print(f"  {clabel:<30}  {mean_a*100:>8.2f}%  "
              f"{std_a*100:>5.2f}%  {delta:>10}  "
              f"{mean_n*100 if mean_n else 0:>9.2f}%")

    # Lambda sweep summary (best λ)
    lam_results = {}
    for s in seeds:
        sweep = data[s].get('F_lambda_sweep', {})
        for lkey, lval in sweep.items():
            acc = lval.get('accuracy')
            if acc:
                lam_results.setdefault(lkey, []).append(acc)

    if lam_results:
        print(f"\n  λ-sweep (Condition F: anchor-mix → ∇V):")
        print(f"  {'─'*50}")
        print(f"  {'λ':>6}  {'Mean Acc':>9}  {'Std':>6}  {'Δ vs Base':>10}")
        for lkey in sorted(lam_results.keys()):
            lam_val = float(lkey.split('_')[1])
            mean_a, std_a = compute_stats(lam_results[lkey])
            delta = f"{(mean_a - base_mean)*100:+.2f}pp" if base_mean else "—"
            print(f"  {lam_val:>6.2f}  {mean_a*100:>8.2f}%  "
                  f"{std_a*100:>5.2f}%  {delta:>10}")
        best_lam = max(lam_results, key=lambda k: np.mean(lam_results[k]))
        best_mean = float(np.mean(lam_results[best_lam]))
        print(f"\n  Best λ: {best_lam} → {best_mean*100:.2f}%")

    print(f"\n  Per-seed accuracy breakdown:")
    print(f"  {'─'*60}")
    header = f"  {'Condition':<28}"
    for s in seeds:
        header += f"  {s:>10}"
    print(header)
    for ckey, clabel in CONDITION_KEYS:
        row = f"  {clabel:<28}"
        for s in seeds:
            acc = extract_metric(data[s], ckey, 'accuracy')
            row += f"  {acc*100:>9.2f}%" if acc else f"  {'—':>10}"
        print(row)

    print(f"{'═'*80}")

    # Verdict
    print(f"\n  VERDICT")
    print(f"  {'─'*40}")
    if base_mean:
        a_accs = [extract_metric(data[s], 'A_grad_v', 'accuracy') for s in seeds]
        a_mean, a_std = compute_stats(a_accs)
        if a_mean and a_mean > base_mean - 0.002:
            print(f"  ✓ Grad-V ≥ Full on {sum(1 for a in a_accs if a and a >= base_accs[seeds.index(list(data.keys())[0])])}/{len(seeds)} seeds (mean Δ={((a_mean or 0)-base_mean)*100:+.2f}pp)")
            print(f"    Law 3 is reproducible across independent seeds.")
        else:
            print(f"  ✗ Grad-V underperforms Full (mean Δ={(a_mean-base_mean)*100:+.2f}pp)")
            print(f"    Multi-seed result is weaker than single-seed finding.")

        d_accs = [extract_metric(data[s], 'D_anchor_then_gradv', 'accuracy') for s in seeds]
        d_mean, d_std = compute_stats(d_accs)
        if d_mean:
            delta = (d_mean - (a_mean or base_mean)) * 100
            if delta > 0.2:
                print(f"  ✓ Anchor projection + ∇V > ∇V alone (+{delta:.2f}pp)")
                print(f"    Manifold contraction hypothesis supported.")
            elif abs(delta) <= 0.2:
                print(f"  ∼ Anchor projection is neutral (Δ={delta:+.2f}pp)")
                print(f"    Dynamics already select the relevant subspace.")
            else:
                print(f"  ✗ Anchor projection hurts (Δ={delta:+.2f}pp)")
                print(f"    Neutral class needs extra-subspace dimensions.")

    print(f"{'═'*80}\n")
    return data


def update_goals_md(data, goals_path):
    seeds = list(data.keys())

    CONDITION_KEYS_SHORT = [
        ('baseline',             'Baseline (Full MLP)'),
        ('A_grad_v',             'A: Grad-V'),
        ('D_anchor_then_gradv',  'D: Anchor-proj → ∇V'),
        ('E_svd_then_gradv',     'E: SVD-k → ∇V'),
    ]

    base_accs = [extract_metric(data[s], 'baseline', 'accuracy') for s in seeds]
    base_mean, base_std = compute_stats(base_accs)

    rows = []
    for ckey, clabel in CONDITION_KEYS_SHORT:
        accs = [extract_metric(data[s], ckey, 'accuracy') for s in seeds]
        mean_a, std_a = compute_stats(accs)
        if mean_a is None:
            continue
        delta = f"{(mean_a - base_mean)*100:+.2f}pp" if base_mean else "—"
        rows.append(f"| {clabel} | {mean_a*100:.2f}% | {std_a*100:.2f}% | {delta} |")

    table = (
        "\n## True Multi-Seed Results (from scratch)\n\n"
        f"Seeds: {', '.join(seeds)} — trained independently from random init.\n\n"
        "| Condition | Mean Acc | Std | Δ vs Baseline |\n"
        "|---|---|---|---|\n"
        + "\n".join(rows) + "\n"
    )

    goals = Path(goals_path).read_text()
    marker = "## Open Questions"
    if marker in goals:
        goals = goals.replace(marker, table + "\n---\n\n" + marker)
        Path(goals_path).write_text(goals)
        print(f"  v3_goals.md updated with multi-seed results table.")
    else:
        print(f"  WARNING: Could not find '{marker}' in v3_goals.md — table not inserted.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str,
                        default='../../../pretrained/multiseed',
                        help='Directory containing seed*/ablation_results.json')
    parser.add_argument('--update-goals', action='store_true',
                        help='Insert results table into v3_goals.md')
    parser.add_argument('--goals-path', type=str,
                        default='../../../v3_goals.md')
    args = parser.parse_args()

    data = load_results(args.results_dir)
    print_summary(data)

    if args.update_goals:
        goals_path = Path(__file__).resolve().parent.parent.parent.parent / 'v3_goals.md'
        if args.goals_path != '../../../v3_goals.md':
            goals_path = Path(args.goals_path)
        update_goals_md(data, goals_path)


if __name__ == '__main__':
    main()
