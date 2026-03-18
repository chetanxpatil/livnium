#!/usr/bin/env python3
"""
multi_seed_eval.py — Multi-seed reproducibility test for the grad-V law.

For each checkpoint provided, runs:
  1. Full (MLP + forces) accuracy
  2. Grad-V accuracy at beta in {1, 5, 20}
  3. Basin agreement: % samples where full and grad-V land in same basin
  4. Anchor geometry: cos(A_E,A_C), cos(A_E,A_N), cos(A_C,A_N)

Usage:
  python3 multi_seed_eval.py \
    --checkpoints ckpt_seed42.pt ckpt_seed1337.pt ckpt_seed7.pt \
    --snli-dev ../../../data/snli/snli_1.0_dev.jsonl \
    --n-samples 9842

To train additional seeds first:
  python3 train_livnium_joint.py --seed 1337 --output-dir ../../../pretrained/livnium-seed1337
  python3 train_livnium_joint.py --seed 7    --output-dir ../../../pretrained/livnium-seed7
"""

import argparse
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Reuse collapse logic from test_gradient_collapse ────────────────────────

def load_checkpoint(ckpt_path, device):
    """Load encoder, collapse engine, head, and anchors from checkpoint."""
    # Import from the same directory
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_gradient_collapse import load_model_components
    return load_model_components(ckpt_path, device)


def encode_samples(encoder, samples, device, batch_size=64):
    """Encode premise/hypothesis pairs → h0, v_p, v_h tensors."""
    h0_list, vp_list, vh_list, label_list = [], [], [], []
    label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        premises = [s["sentence1"] for s in batch]
        hypotheses = [s["sentence2"] for s in batch]
        labels = [label_map[s["gold_label"]] for s in batch]

        with torch.no_grad():
            vp, vh = encoder.encode_batch(premises, hypotheses, device)
            h0 = vh - vp

        h0_list.append(h0.cpu())
        vp_list.append(vp.cpu())
        vh_list.append(vh.cpu())
        label_list.extend(labels)

    return (
        torch.cat(h0_list),
        torch.cat(vp_list),
        torch.cat(vh_list),
        torch.tensor(label_list),
    )


def collapse_grad_v(h0, anchors, beta, alpha, steps=6):
    """Pure analytical grad-V collapse — no MLP, no learned params."""
    h = h0.clone()
    A = anchors  # (3, 768)

    for _ in range(steps):
        h_norm = F.normalize(h, dim=-1)
        A_norm = F.normalize(A, dim=-1)

        # cos(h, A_k) for each anchor
        cos_scores = (h_norm @ A_norm.T)           # (N, 3)
        weights = F.softmax(beta * cos_scores, dim=-1)  # Boltzmann weights

        # gradient of V(h) w.r.t. h
        h_len = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        grad = torch.zeros_like(h)
        for k in range(3):
            # ∇_h cos(h, A_k) = (A_k - h·cos(h,A_k)) / ||h||
            cos_k = cos_scores[:, k:k+1]
            grad_cos_k = (A_norm[k].unsqueeze(0) - h_norm * cos_k) / h_len
            grad -= weights[:, k:k+1] * grad_cos_k  # ∇V = -Σ w_k ∇cos

        h = h - alpha * grad

    return h


def collapse_full(h0, engine, steps=6):
    """Full trained collapse (MLP + forces)."""
    with torch.no_grad():
        h = h0.clone()
        for _ in range(steps):
            h = engine.step(h)
    return h


def get_predictions(h_final, head, v_p, v_h):
    """Run classification head → predicted labels."""
    with torch.no_grad():
        logits = head(h_final, v_p, v_h)
        return logits.argmax(dim=-1)


def accuracy_and_recall(preds, labels):
    acc = (preds == labels).float().mean().item()
    recall = {}
    for cls, name in enumerate(["E", "C", "N"]):
        mask = labels == cls
        if mask.sum() > 0:
            recall[name] = (preds[mask] == labels[mask]).float().mean().item()
        else:
            recall[name] = float("nan")
    return acc, recall


def get_anchor_geometry(anchors):
    """Compute pairwise cosine similarities between anchors."""
    A = F.normalize(anchors, dim=-1).cpu()
    return {
        "cos(E,C)": (A[0] @ A[1]).item(),
        "cos(E,N)": (A[0] @ A[2]).item(),
        "cos(C,N)": (A[1] @ A[2]).item(),
    }


def basin_agreement(preds_full, preds_gradv):
    """% samples where full and grad-V land in same predicted class."""
    return (preds_full == preds_gradv).float().mean().item()


# ─── Main ────────────────────────────────────────────────────────────────────

def run_checkpoint(ckpt_path, samples, args, device):
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    encoder, engine, head, anchors = load_checkpoint(ckpt_path, device)
    encoder.eval(); head.eval()

    print("Encoding samples...")
    h0, v_p, v_h, labels = encode_samples(encoder, samples, device)
    h0 = h0.to(device)
    v_p = v_p.to(device)
    v_h = v_h.to(device)
    labels = labels.to(device)
    A = anchors.to(device)

    results = {}

    # 1. Full (MLP + forces)
    print("Running Full (MLP + forces)...")
    h_full = collapse_full(h0, engine)
    preds_full = get_predictions(h_full.to(device), head, v_p, v_h)
    acc_full, recall_full = accuracy_and_recall(preds_full, labels)
    results["full"] = {"acc": acc_full, "recall": recall_full}
    print(f"  Full:   acc={acc_full:.4f}  E={recall_full['E']:.3f}  N={recall_full['N']:.3f}  C={recall_full['C']:.3f}")

    # 2. Grad-V at each beta
    best_beta = None
    best_gradv_acc = -1
    results["grad_v"] = {}

    for beta in args.betas:
        print(f"Running Grad-V (beta={beta})...")
        h_gv = collapse_grad_v(h0, A, beta=beta, alpha=args.alpha, steps=args.steps)
        preds_gv = get_predictions(h_gv.to(device), head, v_p, v_h)
        acc_gv, recall_gv = accuracy_and_recall(preds_gv, labels)
        agree = basin_agreement(preds_full, preds_gv)

        results["grad_v"][beta] = {
            "acc": acc_gv,
            "recall": recall_gv,
            "basin_agreement": agree,
        }
        print(f"  Grad-V β={beta}: acc={acc_gv:.4f}  N={recall_gv['N']:.3f}  agree={agree:.3f}")

        if acc_gv > best_gradv_acc:
            best_gradv_acc = acc_gv
            best_beta = beta

    results["best_beta"] = best_beta
    results["best_gradv_acc"] = best_gradv_acc
    results["delta_vs_full"] = best_gradv_acc - acc_full

    # 3. Anchor geometry
    geom = get_anchor_geometry(A)
    results["geometry"] = geom
    print(f"  Anchors: cos(E,C)={geom['cos(E,C)']:+.4f}  cos(E,N)={geom['cos(E,N)']:+.4f}  cos(C,N)={geom['cos(C,N)']:+.4f}")

    return results


def print_summary(all_results):
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Checkpoint':<35} {'Full':>6} {'Best β':>6} {'Grad-V':>7} {'Δ':>6} {'Agree':>7}")
    print("-" * 70)

    best_betas = []
    agreements = []
    deltas = []
    geoms = defaultdict(list)

    for ckpt, res in all_results.items():
        name = Path(ckpt).stem[:34]
        full_acc = res["full"]["acc"]
        best_beta = res["best_beta"]
        best_acc = res["best_gradv_acc"]
        delta = res["delta_vs_full"]
        agree = res["grad_v"][best_beta]["basin_agreement"]

        best_betas.append(best_beta)
        agreements.append(agree)
        deltas.append(delta)
        for k, v in res["geometry"].items():
            geoms[k].append(v)

        print(f"{name:<35} {full_acc:>6.4f} {best_beta:>6} {best_acc:>7.4f} {delta:>+6.4f} {agree:>7.3f}")

    print(f"\n--- Reproducibility ---")
    print(f"Best β across seeds: {best_betas}  {'✅ CONSISTENT' if len(set(best_betas))==1 else '⚠️  VARIES'}")
    print(f"Basin agreement:     mean={np.mean(agreements):.3f}  std={np.std(agreements):.4f}")
    print(f"Accuracy delta:      mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}")

    print(f"\n--- Anchor Geometry ---")
    for k, vals in geoms.items():
        print(f"  {k}: {[f'{v:+.4f}' for v in vals]}  mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}")

    # Verdict
    print(f"\n--- Verdict ---")
    beta_consistent = len(set(best_betas)) == 1
    agree_high = np.mean(agreements) > 0.90
    delta_small = abs(np.mean(deltas)) < 0.005

    if beta_consistent and agree_high and delta_small:
        print("✅ LAW IS STRUCTURAL: same β optimal, >90% basin agreement, <0.5% accuracy delta across seeds.")
    elif agree_high and delta_small:
        print("🟡 LAW HOLDS but β varies — energy function is robust, sharpness parameter isn't pinned.")
    else:
        print("⚠️  INCONSISTENT — law may be seed-specific. Check basin agreement and delta per seed.")


def load_snli(path, n_samples):
    samples = []
    label_set = {"entailment", "contradiction", "neutral"}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("gold_label") in label_set:
                samples.append(obj)
                if len(samples) >= n_samples:
                    break
    return samples


def main():
    parser = argparse.ArgumentParser(description="Multi-seed grad-V reproducibility test")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to checkpoint files (one per seed)")
    parser.add_argument("--snli-dev", required=True,
                        help="Path to SNLI dev jsonl")
    parser.add_argument("--n-samples", type=int, default=9842,
                        help="Number of dev samples to evaluate")
    parser.add_argument("--betas", nargs="+", type=float, default=[1.0, 5.0, 20.0],
                        help="Beta values to sweep for grad-V")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Step size for grad-V")
    parser.add_argument("--steps", type=int, default=6,
                        help="Number of collapse steps")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save results as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.n_samples} SNLI dev samples...")
    samples = load_snli(args.snli_dev, args.n_samples)
    print(f"Loaded {len(samples)} samples.")

    all_results = {}
    for ckpt in args.checkpoints:
        all_results[ckpt] = run_checkpoint(ckpt, samples, args, device)

    print_summary(all_results)

    if args.output:
        # Convert tensors/non-serializable to plain Python
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        with open(args.output, "w") as f:
            json.dump(clean(all_results), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
