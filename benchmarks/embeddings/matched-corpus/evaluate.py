"""evaluate.py — score one model's vectors on all similarity benchmarks.

Datasets: simlex999 (full), simlex_nouns (POS == N, comparable to the
published 0.362), ws353, men3000. Correlation: tie-aware Spearman ONLY.

For every (model, dataset) it writes per-pair predictions to CSV
(work/results/{tag}.{dataset}.pairs.csv) so report.py can recompute rho on
the exact pair INTERSECTION shared by all models — equal-coverage comparison.

Usage:
    python3 evaluate.py --model work/models/collapse_v2_seed0.npz
"""

import argparse
import csv
import os

from common import WORK, load_eval_set, save_json, spearman

DATASETS = ["simlex999", "simlex_nouns", "ws353", "men3000"]


def load_vectors(path):
    import numpy as np
    z = np.load(path, allow_pickle=True)
    X = z["vectors"].astype(np.float64)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X, {w: i for i, w in enumerate(z["words"].tolist())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="work/models/*.npz")
    args = ap.parse_args()

    X, idx = load_vectors(args.model)
    tag = os.path.splitext(os.path.basename(args.model))[0]
    rdir = os.path.join(WORK, "results")
    os.makedirs(rdir, exist_ok=True)

    summary = {"model": tag, "datasets": {}}
    meta_path = args.model.replace(".npz", ".meta.json")
    if os.path.exists(meta_path):
        from common import load_json
        summary["meta"] = load_json(meta_path)

    for name in DATASETS:
        pairs = load_eval_set(name)
        rows, sims, gold = [], [], []
        for w1, w2, g in pairs:
            covered = w1 in idx and w2 in idx
            s = float(X[idx[w1]] @ X[idx[w2]]) if covered else ""
            rows.append((w1, w2, g, s, int(covered)))
            if covered:
                sims.append(s); gold.append(g)
        with open(os.path.join(rdir, f"{tag}.{name}.pairs.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["word1", "word2", "gold", "model_cosine", "covered"])
            w.writerows(rows)
        rho = spearman(sims, gold) if len(gold) >= 10 else None
        summary["datasets"][name] = {
            "rho_own_coverage": rho, "covered": len(gold), "total": len(pairs)}
        print(f"{tag:28s} {name:13s} rho={rho if rho is None else f'{rho:.4f}'}  "
              f"coverage {len(gold)}/{len(pairs)}")

    save_json(os.path.join(rdir, f"{tag}.json"), summary)


if __name__ == "__main__":
    main()
