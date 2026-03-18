"""
Gradient Collapse Test
======================
Tests whether replacing the learned update δ_θ(h_t) with a pure
analytical gradient of an energy function V(h) matches accuracy.

Three collapse modes compared:
  1. Full:     h_{t+1} = h_t + δ_θ(h_t) - anchor_forces   (current system)
  2. No-delta: h_{t+1} = h_t             - anchor_forces   (remove learned MLP)
  3. Grad-V:   h_{t+1} = h_t - α * ∇V(h_t)                (pure gradient flow)

Energy function used in mode 3:
  V(h) = -logsumexp(β * cos(h, anchor_i))   (soft-min distance to anchors)
  ∇V(h) = -(softmax(β * alignments) @ anchors - h_n * sum)  (projected)

This tells you:
  - How much δ_θ contributes vs pure geometry (mode 1 vs 2)
  - Whether a clean energy function can match (mode 1 vs 3)
  - Speed of mode 3 vs mode 1 (no MLP forward pass needed)

Usage:
  python3 test_gradient_collapse.py \\
      --checkpoint ../../../pretrained/bert-joint/best_model.pt \\
      --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \\
      --n-samples  2000 \\
      --beta 5.0 \\
      --alpha 0.1
"""

import sys
import time
import argparse
import math
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

_here = Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from core import VectorCollapseEngine
from tasks.snli import BERTSNLIEncoder, SNLIHead
from train import load_snli_data


# ──────────────────────────────────────────────────────────────────────────────
# Three collapse modes
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collapse_full(engine, h0):
    """Standard Livnium: δ_θ + anchor forces."""
    return engine.collapse(h0)[0]


@torch.no_grad()
def collapse_no_delta(engine, h0):
    """Anchor forces only — no learned MLP residual."""
    h = h0.clone()
    if h.dim() == 1:
        h = h.unsqueeze(0)

    e_dir = F.normalize(engine.anchor_entail.detach(), dim=0)
    c_dir = F.normalize(engine.anchor_contra.detach(), dim=0)
    n_dir = F.normalize(engine.anchor_neutral.detach(), dim=0)

    for _ in range(engine.num_layers):
        h_n = F.normalize(h, dim=-1)
        a_e = (h_n * e_dir).sum(dim=-1)
        a_c = (h_n * c_dir).sum(dim=-1)
        a_n = (h_n * n_dir).sum(dim=-1)

        from core.physics_laws import divergence_from_alignment, boundary_proximity
        d_e = divergence_from_alignment(a_e)
        d_c = divergence_from_alignment(a_c)
        d_n = divergence_from_alignment(a_n)
        ec_boundary = boundary_proximity(a_e, a_c)

        e_vec = F.normalize(h - e_dir.unsqueeze(0), dim=-1)
        c_vec = F.normalize(h - c_dir.unsqueeze(0), dim=-1)
        n_vec = F.normalize(h - n_dir.unsqueeze(0), dim=-1)

        h = (h
             - engine.strength_entail * d_e.unsqueeze(-1) * e_vec
             - engine.strength_contra  * d_c.unsqueeze(-1) * c_vec
             - engine.strength_neutral * d_n.unsqueeze(-1) * n_vec
             - engine.strength_neutral_boost * ec_boundary.unsqueeze(-1) * n_vec)

        h_norm = h.norm(p=2, dim=-1, keepdim=True)
        h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)

    return h.squeeze(0) if h0.dim() == 1 else h


@torch.no_grad()
def collapse_grad_v(engine, h0, beta=5.0, alpha=0.1):
    """
    Pure gradient flow: h_{t+1} = h_t - alpha * ∇V(h_t)

    Energy: V(h) = -1/beta * logsumexp(beta * [cos(h,aE), cos(h,aC), cos(h,aN)])
    This is a smooth approximation to -max_i cos(h, anchor_i).
    ∇V pulls h toward the nearest anchor — a clean energy-based dynamics.

    No neural network forward pass — pure linear algebra.
    """
    h = h0.clone()
    if h.dim() == 1:
        h = h.unsqueeze(0)

    e_dir = F.normalize(engine.anchor_entail.detach(), dim=0)
    c_dir = F.normalize(engine.anchor_contra.detach(), dim=0)
    n_dir = F.normalize(engine.anchor_neutral.detach(), dim=0)
    anchors = torch.stack([e_dir, c_dir, n_dir], dim=0)  # [3, dim]

    for _ in range(engine.num_layers):
        h_n = F.normalize(h, dim=-1)                    # [B, dim]
        alignments = h_n @ anchors.T                    # [B, 3]

        # softmax weights: which anchor is h closest to?
        weights = torch.softmax(beta * alignments, dim=-1)  # [B, 3]

        # weighted target direction in normalized space
        target = weights @ anchors                       # [B, dim]

        # gradient of V w.r.t. h (projected onto unit sphere tangent plane):
        # ∇V = -(target - h_n * (h_n·target)) / ||h||
        # Simplified: just step h toward weighted anchor sum
        h_norm = h.norm(p=2, dim=-1, keepdim=True)
        grad_v = -(target - h_n * (h_n * target).sum(dim=-1, keepdim=True))

        h = h - alpha * h_norm * grad_v

        # norm control
        h_norm2 = h.norm(p=2, dim=-1, keepdim=True)
        h = torch.where(h_norm2 > 10.0, h * (10.0 / (h_norm2 + 1e-8)), h)

    return h.squeeze(0) if h0.dim() == 1 else h


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def accuracy_and_recall(preds, labels):
    n = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    recall = []
    for c in range(3):
        c_total = sum(1 for l in labels if l == c)
        c_correct = sum(1 for p, l in zip(preds, labels) if l == c and p == c)
        recall.append(c_correct / c_total if c_total > 0 else 0.0)
    return correct / n, recall


def time_fn(fn, n_runs=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--snli-dev',   type=str, required=True)
    parser.add_argument('--n-samples',  type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--beta',  type=float, default=5.0,
                        help='Temperature for softmax energy (higher = sharper)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Step size for gradient collapse')
    parser.add_argument('--beta-sweep', action='store_true',
                        help='Sweep beta values to find best gradient collapse')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    print(f"  dim={model_args.dim}, layers={model_args.num_layers}")

    # Load dev data and encode
    print(f"\nEncoding {args.n_samples} dev samples...")
    raw = load_snli_data(args.snli_dev, max_samples=args.n_samples)
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    all_h0, all_vp, all_vh, all_labels = [], [], [], []
    bs = args.batch_size
    with torch.no_grad():
        for b in tqdm(range(math.ceil(len(raw) / bs)), desc='Encoding'):
            batch = raw[b*bs:(b+1)*bs]
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

    all_h0 = torch.cat(all_h0).to(device)
    all_vp = torch.cat(all_vp).to(device)
    all_vh = torch.cat(all_vh).to(device)
    N = len(all_labels)
    print(f"  {N} samples ready")

    def run_mode(collapse_fn, desc):
        preds = []
        with torch.no_grad():
            for i in range(0, N, bs):
                h0_b = all_h0[i:i+bs]
                vp_b = all_vp[i:i+bs]
                vh_b = all_vh[i:i+bs]
                hf = collapse_fn(h0_b)
                logits = head(hf, vp_b, vh_b)
                preds.extend(logits.argmax(dim=-1).tolist())
        acc, recall = accuracy_and_recall(preds, all_labels)
        return acc, recall

    # Speed benchmark (batch=64, 20 runs)
    h0_bench = all_h0[:64]
    vp_bench = all_vp[:64]
    vh_bench = all_vh[:64]

    def bench_full():
        hf = collapse_full(engine, h0_bench)
        head(hf, vp_bench, vh_bench)

    def bench_grad(b=args.beta, a=args.alpha):
        hf = collapse_grad_v(engine, h0_bench, beta=b, alpha=a)
        head(hf, vp_bench, vh_bench)

    # ── Run all three modes ───────────────────────────────────────────────────
    print("\nRunning mode 1: Full (δ_θ + anchor forces)...")
    acc_full, rec_full = run_mode(lambda h: collapse_full(engine, h), "full")

    print("Running mode 2: No-delta (anchor forces only)...")
    acc_nodelta, rec_nodelta = run_mode(lambda h: collapse_no_delta(engine, h), "no-delta")

    if args.beta_sweep:
        betas  = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        alphas = [0.05, 0.1, 0.2, 0.5]
        print(f"\nRunning mode 3: Grad-V sweep (beta × alpha)...")
        print(f"\n  {'beta':>6}  {'alpha':>6}  {'acc':>6}  {'Δ vs full':>10}  "
              f"{'E-rec':>6}  {'N-rec':>6}  {'C-rec':>6}")
        print(f"  {'─'*60}")
        print(f"  {'full':>6}  {'─':>6}  {acc_full:.4f}  {'—':>10}  "
              f"{rec_full[0]:.4f}  {rec_full[2]:.4f}  {rec_full[1]:.4f}")
        print(f"  {'no-δ':>6}  {'─':>6}  {acc_nodelta:.4f}  "
              f"{acc_nodelta-acc_full:>+10.4f}  "
              f"{rec_nodelta[0]:.4f}  {rec_nodelta[2]:.4f}  {rec_nodelta[1]:.4f}")
        print(f"  {'─'*60}")
        best_acc, best_beta, best_alpha = 0, 0, 0
        for beta in betas:
            for alpha in alphas:
                acc_gv, rec_gv = run_mode(
                    lambda h, b=beta, a=alpha: collapse_grad_v(engine, h, beta=b, alpha=a),
                    f"grad-V b={beta} a={alpha}"
                )
                delta = acc_gv - acc_full
                marker = " ◄" if acc_gv > best_acc else ""
                if acc_gv > best_acc:
                    best_acc, best_beta, best_alpha = acc_gv, beta, alpha
                print(f"  {beta:>6.1f}  {alpha:>6.2f}  {acc_gv:.4f}  "
                      f"{delta:>+10.4f}  "
                      f"{rec_gv[0]:.4f}  {rec_gv[2]:.4f}  {rec_gv[1]:.4f}{marker}")
        print(f"\n  Best grad-V: beta={best_beta}, alpha={best_alpha}, acc={best_acc:.4f}")
        print(f"  Gap to full: {best_acc-acc_full:+.4f}")

    else:
        print(f"Running mode 3: Grad-V (beta={args.beta}, alpha={args.alpha})...")
        acc_gv, rec_gv = run_mode(
            lambda h: collapse_grad_v(engine, h, beta=args.beta, alpha=args.alpha),
            "grad-V"
        )

        # Speed
        full_ms, full_std  = time_fn(bench_full)
        grad_ms, grad_std  = time_fn(lambda: bench_grad(args.beta, args.alpha))
        speedup = full_ms / grad_ms

        # Report
        print(f"\n{'═'*64}")
        print(f"  GRADIENT COLLAPSE COMPARISON")
        print(f"{'═'*64}")
        print(f"\n  {'Mode':<26}  {'Acc':>6}  "
              f"{'E-rec':>6}  {'N-rec':>6}  {'C-rec':>6}  {'ms/b64':>8}")
        print(f"  {'─'*60}")
        print(f"  {'1. Full (δ_θ + forces)':<26}  {acc_full:.4f}  "
              f"{rec_full[0]:.4f}  {rec_full[2]:.4f}  {rec_full[1]:.4f}  "
              f"{full_ms:>8.1f}")
        print(f"  {'2. No-delta (forces only)':<26}  {acc_nodelta:.4f}  "
              f"{rec_nodelta[0]:.4f}  {rec_nodelta[2]:.4f}  {rec_nodelta[1]:.4f}  "
              f"{'n/a':>8}")
        print(f"  {'3. Grad-V (∇V, no MLP)':<26}  {acc_gv:.4f}  "
              f"{rec_gv[0]:.4f}  {rec_gv[2]:.4f}  {rec_gv[1]:.4f}  "
              f"{grad_ms:>8.1f}")
        print(f"  {'─'*60}")

        d_nodelta = acc_nodelta - acc_full
        d_gv      = acc_gv - acc_full
        print(f"  Mode 2 Δ (no δ_θ):   {d_nodelta:+.4f}  "
              f"→ this is what δ_θ contributes")
        print(f"  Mode 3 Δ (grad-V):   {d_gv:+.4f}  "
              f"→ this is the accuracy cost of pure gradient")
        print(f"  Grad-V speedup:      {speedup:.1f}x  "
              f"(collapse+head: {full_ms:.1f}ms → {grad_ms:.1f}ms)")

        print(f"\n  Interpretation:")
        if abs(d_nodelta) < 0.005:
            print(f"  • δ_θ contributes almost nothing — pure geometry drives the dynamics.")
        elif d_nodelta < -0.01:
            print(f"  • δ_θ contributes {-d_nodelta*100:.1f}% accuracy — the MLP is doing real work.")
        else:
            print(f"  • δ_θ contributes {abs(d_nodelta)*100:.1f}% accuracy (borderline).")

        if abs(d_gv) < 0.01:
            print(f"  • Grad-V matches full accuracy — the system IS a gradient flow.")
            print(f"    V(h) = -logsumexp(β·cos(h,anchors)) is the law.")
        elif d_gv > -0.03:
            print(f"  • Grad-V loses {-d_gv*100:.1f}% — close, but energy function needs tuning.")
            print(f"    Try --beta-sweep to find optimal β and α.")
        else:
            print(f"  • Grad-V loses {-d_gv*100:.1f}% — the dynamics are richer than ∇V.")
            print(f"    δ_θ is doing something a simple energy function can't replicate.")

        print(f"{'═'*64}\n")
        print(f"  Tip: run with --beta-sweep to scan β × α grid automatically")


if __name__ == '__main__':
    main()
