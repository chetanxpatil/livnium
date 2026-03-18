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
"""

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

_here = Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from core import VectorCollapseEngine
from tasks.snli import BERTSNLIEncoder, SNLIHead
from train import load_snli_data


# ─── Model loading ───────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = ckpt['args']

    encoder = BERTSNLIEncoder(freeze=True).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    engine = VectorCollapseEngine(
        dim=model_args.dim,
        num_layers=model_args.num_layers,
        strength_entail=getattr(model_args, 'strength_entail', 0.1),
        strength_contra=getattr(model_args, 'strength_contra', 0.1),
        strength_neutral=getattr(model_args, 'strength_neutral', 0.05),
        strength_neutral_boost=getattr(model_args, 'strength_neutral_boost', 0.05),
    ).to(device)
    engine.load_state_dict(ckpt['collapse_engine'])
    engine.eval()

    head = SNLIHead(dim=model_args.dim).to(device)
    head.load_state_dict(ckpt['head'])
    head.eval()

    # Extract anchors from collapse engine
    anchors = torch.stack([
        engine.anchor_entail,
        engine.anchor_contra,
        engine.anchor_neutral,
    ]).detach()  # (3, 768)

    return encoder, engine, head, anchors


# ─── Encoding ────────────────────────────────────────────────────────────────

def encode_samples(encoder, raw, device, batch_size=64):
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    all_h0, all_vp, all_vh, all_labels = [], [], [], []

    with torch.no_grad():
        for b in tqdm(range(math.ceil(len(raw) / batch_size)), desc='Encoding'):
            batch = raw[b*batch_size:(b+1)*batch_size]
            premises   = [s['premise']    for s in batch]
            hypotheses = [s['hypothesis'] for s in batch]
            labels_b   = [label_map.get(s['gold_label'], 2) for s in batch]

            h0, vp, vh = encoder.build_initial_state(
                premises, hypotheses, add_noise=False, device=device
            )
            all_h0.append(h0.cpu())
            all_vp.append(vp.cpu())
            all_vh.append(vh.cpu())
            all_labels.extend(labels_b)

    return (
        torch.cat(all_h0),
        torch.cat(all_vp),
        torch.cat(all_vh),
        torch.tensor(all_labels),
    )


# ─── Collapse modes ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_full(engine, h0):
    return engine.collapse(h0)[0]


@torch.no_grad()
def run_grad_v(h0, anchors, beta, alpha, steps=6):
    """
    Pure gradient flow matching test_gradient_collapse.py exactly:
      h_{t+1} = h_t - alpha * ||h_t|| * grad_V(h_t)

    where grad_V is the tangent-plane projected gradient of
      V(h) = -1/beta * logsumexp(beta * cos(h, anchor_i))
    """
    h = h0.clone()
    A = F.normalize(anchors, dim=-1)   # (3, dim)

    for _ in range(steps):
        h_n = F.normalize(h, dim=-1)           # (N, dim)
        alignments = h_n @ A.T                 # (N, 3)
        weights = torch.softmax(beta * alignments, dim=-1)   # (N, 3)

        # weighted target direction in normalised space
        target = weights @ A                   # (N, dim)

        # tangent-plane gradient of V w.r.t. h
        grad_v = -(target - h_n * (h_n * target).sum(dim=-1, keepdim=True))

        # step scaled by current norm (matches test_gradient_collapse.py)
        h_norm = h.norm(p=2, dim=-1, keepdim=True)
        h = h - alpha * h_norm * grad_v

        # norm control
        h_norm2 = h.norm(p=2, dim=-1, keepdim=True)
        h = torch.where(h_norm2 > 10.0, h * (10.0 / (h_norm2 + 1e-8)), h)

    return h


# ─── Metrics ─────────────────────────────────────────────────────────────────

def get_predictions(h_final, head, v_p, v_h, device, batch_size=256):
    preds = []
    n = h_final.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            logits = head(
                h_final[i:i+batch_size].to(device),
                v_p[i:i+batch_size].to(device),
                v_h[i:i+batch_size].to(device),
            )
            preds.append(logits.argmax(dim=-1).cpu())
    return torch.cat(preds)


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
    A = F.normalize(anchors.cpu(), dim=-1)
    return {
        "cos(E,C)": (A[0] @ A[1]).item(),
        "cos(E,N)": (A[0] @ A[2]).item(),
        "cos(C,N)": (A[1] @ A[2]).item(),
    }


def basin_agreement(preds_a, preds_b):
    return (preds_a == preds_b).float().mean().item()


# ─── Per-checkpoint eval ─────────────────────────────────────────────────────

def run_checkpoint(ckpt_path, raw, args, device):
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    encoder, engine, head, anchors = load_checkpoint(ckpt_path, device)
    A = anchors.to(device)

    h0, v_p, v_h, labels = encode_samples(encoder, raw, device)

    results = {}

    # 1. Full (MLP + forces)
    print("Running Full (MLP + forces)...")
    h_full = run_full(engine, h0.to(device)).cpu()
    preds_full = get_predictions(h_full, head, v_p, v_h, device)
    acc_full, recall_full = accuracy_and_recall(preds_full, labels)
    results["full"] = {"acc": acc_full, "recall": recall_full}
    print(f"  Full:   acc={acc_full:.4f}  E={recall_full['E']:.3f}  N={recall_full['N']:.3f}  C={recall_full['C']:.3f}")

    # 2. Grad-V sweep
    best_beta, best_acc_gv = None, -1
    results["grad_v"] = {}

    for beta in args.betas:
        print(f"Running Grad-V (beta={beta})...")
        h_gv = run_grad_v(h0.to(device), A, beta=beta, alpha=args.alpha, steps=args.steps).cpu()
        preds_gv = get_predictions(h_gv, head, v_p, v_h, device)
        acc_gv, recall_gv = accuracy_and_recall(preds_gv, labels)
        agree = basin_agreement(preds_full, preds_gv)

        results["grad_v"][beta] = {
            "acc": acc_gv,
            "recall": recall_gv,
            "basin_agreement": agree,
        }
        print(f"  Grad-V β={beta}: acc={acc_gv:.4f}  N={recall_gv['N']:.3f}  agree={agree:.3f}")

        if acc_gv > best_acc_gv:
            best_acc_gv = acc_gv
            best_beta = beta

    results["best_beta"] = best_beta
    results["best_gradv_acc"] = best_acc_gv
    results["delta_vs_full"] = best_acc_gv - acc_full

    # 3. Anchor geometry
    geom = get_anchor_geometry(A)
    results["geometry"] = geom
    print(f"  Anchors: cos(E,C)={geom['cos(E,C)']:+.4f}  cos(E,N)={geom['cos(E,N)']:+.4f}  cos(C,N)={geom['cos(C,N)']:+.4f}")

    return results


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary(all_results):
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Checkpoint':<35} {'Full':>6} {'Best β':>6} {'Grad-V':>7} {'Δ':>7} {'Agree':>7}")
    print("-" * 72)

    best_betas, agreements, deltas = [], [], []
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

        print(f"{name:<35} {full_acc:>6.4f} {best_beta:>6} {best_acc:>7.4f} {delta:>+7.4f} {agree:>7.3f}")

    print(f"\n--- Reproducibility ---")
    print(f"Best β across seeds:  {best_betas}  {'✅ CONSISTENT' if len(set(best_betas))==1 else '⚠️  VARIES'}")
    print(f"Basin agreement:      mean={np.mean(agreements):.3f}  std={np.std(agreements):.4f}")
    print(f"Accuracy delta:       mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}")

    print(f"\n--- Anchor Geometry ---")
    for k, vals in geoms.items():
        print(f"  {k}: {[f'{v:+.4f}' for v in vals]}  mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}")

    print(f"\n--- Verdict ---")
    beta_consistent = len(set(best_betas)) == 1
    agree_high = np.mean(agreements) > 0.90
    delta_small = abs(np.mean(deltas)) < 0.005

    if beta_consistent and agree_high and delta_small:
        print("✅ LAW IS STRUCTURAL: same β optimal, >90% basin agreement, <0.5% delta across seeds.")
    elif agree_high and delta_small:
        print("🟡 LAW HOLDS but β varies — energy function robust, sharpness not pinned.")
    elif delta_small:
        print("🟡 ACCURACY HOLDS but basin agreement low — check if grad-V takes different paths to same answer.")
    else:
        print("⚠️  INCONSISTENT — law may be seed-specific. Inspect per-seed results.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-seed grad-V reproducibility test")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--snli-dev", required=True)
    parser.add_argument("--n-samples", type=int, default=9842)
    parser.add_argument("--betas", nargs="+", type=float, default=[1.0, 5.0, 20.0])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.n_samples} SNLI dev samples...")
    raw = load_snli_data(args.snli_dev, max_samples=args.n_samples)
    print(f"Loaded {len(raw)} samples.")

    all_results = {}
    for ckpt in args.checkpoints:
        all_results[ckpt] = run_checkpoint(ckpt, raw, args, device)

    print_summary(all_results)

    if args.output:
        def clean(obj):
            if isinstance(obj, dict):
                return {str(k): clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [clean(v) for v in obj]
            if isinstance(obj, (np.floating, np.integer, float)):
                return float(obj)
            return obj

        with open(args.output, "w") as f:
            json.dump(clean(all_results), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
