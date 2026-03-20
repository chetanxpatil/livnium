"""
5-Condition Manifold Contraction Ablation
==========================================
Tests whether contracting the representation to an anchor-aligned subspace
before running energy dynamics improves NLI accuracy.

Core hypothesis: task-aligned manifold contraction may simplify the effective
inference landscape by pruning nuisance dimensions before iterative dynamics.

Conditions
----------
  Baseline : Full model  — BERT + learned MLP dynamics
  A        : Grad-V only — replace MLP with ∇V(h), no projection
  B        : Anchor-proj — project h₀ onto anchor subspace, then head only
  C        : SVD-proj    — project h₀ onto top-k PCA directions, then head only
  D        : Anchor-proj → ∇V dynamics → head
  E        : SVD-proj    → ∇V dynamics → head
  F (sweep): λ-mixing    — h' = λ·P_anchors(h) + (1−λ)·h → ∇V → head

Usage
-----
  python3 ablation_experiment.py \\
      --checkpoint ../../../pretrained/bert-joint/best_model.pt \\
      --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \\
      --n-samples  -1          # -1 = full dev set (9842)
      --beta 2.0               # best from test_gradient_collapse sweep
      --alpha 0.05             # best from test_gradient_collapse sweep
      --svd-rank 16            # top-k principal directions of anchor covariance
      --lambda-sweep           # run condition F sweep over λ ∈ [0, 1]

Output
------
  ablation_results.json in the same directory as the checkpoint.
"""

import sys
import json
import argparse
import math
from pathlib import Path

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
# Projection operators
# ──────────────────────────────────────────────────────────────────────────────

def anchor_projection_matrix(engine):
    """
    Returns U @ U.T  (the orthogonal projector onto the anchor subspace).

    The anchor subspace is the row space of [A_E; A_C; A_N] ∈ ℝ^{3 × 768}.
    After SVD, U ∈ ℝ^{768 × r} spans that space (r ≤ 3).

    P(h) = (h @ U) @ U.T  projects h into and back out of the anchor plane.
    """
    e_dir = F.normalize(engine.anchor_entail.detach(), dim=0)
    c_dir = F.normalize(engine.anchor_contra.detach(), dim=0)
    n_dir = F.normalize(engine.anchor_neutral.detach(), dim=0)
    A = torch.stack([e_dir, c_dir, n_dir], dim=0)          # [3, 768]
    # full_matrices=False → U is [768, r], S is [r], Vh is [r, 768]
    U, _, _ = torch.linalg.svd(A.T, full_matrices=False)   # U: [768, r]
    return U                                                 # use as U @ U.T


def svd_projection_matrix(h0_all, rank):
    """
    Returns V_k ∈ ℝ^{768 × k}: the top-k principal directions of h₀.

    This is data-driven compression — no anchor geometry involved.
    P_k(h) = (h @ V_k) @ V_k.T  projects onto the dominant subspace of h₀.
    """
    h_centered = h0_all - h0_all.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(h_centered, full_matrices=False)
    return Vh[:rank].T                                       # [768, rank]


def project(h, U):
    """Orthogonal projection: h_proj = h @ U @ U.T"""
    coords = h @ U         # [B, r]
    return coords @ U.T    # [B, 768]


def mix(h, h_proj, lam):
    """λ-mixing: h' = λ·P(h) + (1−λ)·h"""
    return lam * h_proj + (1.0 - lam) * h


# ──────────────────────────────────────────────────────────────────────────────
# Collapse modes  (all return h_final of same shape as h0)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collapse_full(engine, h0):
    return engine.collapse(h0)[0]


@torch.no_grad()
def collapse_grad_v(engine, h0, beta=2.0, alpha=0.05):
    h = h0.clone()
    e_dir = F.normalize(engine.anchor_entail.detach(), dim=0)
    c_dir = F.normalize(engine.anchor_contra.detach(), dim=0)
    n_dir = F.normalize(engine.anchor_neutral.detach(), dim=0)
    anchors = torch.stack([e_dir, c_dir, n_dir], dim=0)   # [3, 768]

    for _ in range(engine.num_layers):
        h_n = F.normalize(h, dim=-1)
        weights = torch.softmax(beta * (h_n @ anchors.T), dim=-1)
        target  = weights @ anchors
        h_norm  = h.norm(p=2, dim=-1, keepdim=True)
        grad_v  = -(target - h_n * (h_n * target).sum(dim=-1, keepdim=True))
        h = h - alpha * h_norm * grad_v
        h_norm2 = h.norm(p=2, dim=-1, keepdim=True)
        h = torch.where(h_norm2 > 10.0, h * (10.0 / (h_norm2 + 1e-8)), h)
    return h


@torch.no_grad()
def collapse_proj_then_gradv(engine, h0, U, beta=2.0, alpha=0.05):
    """Condition D / E: project first, then run ∇V dynamics."""
    h = project(h0, U)
    return collapse_grad_v(engine, h, beta=beta, alpha=alpha)


@torch.no_grad()
def collapse_mixed_then_gradv(engine, h0, U, lam, beta=2.0, alpha=0.05):
    """Condition F: λ-mixing, then ∇V dynamics."""
    h_proj = project(h0, U)
    h_mixed = mix(h0, h_proj, lam)
    return collapse_grad_v(engine, h_mixed, beta=beta, alpha=alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(collapse_fn, head, all_h0, all_vp, all_vh, all_labels,
             device, batch_size=64):
    """Run collapse_fn → head → accuracy + per-class recall + basin agreement."""
    N = len(all_labels)
    preds, finals = [], []

    with torch.no_grad():
        for i in range(0, N, batch_size):
            h0_b = all_h0[i:i+batch_size]
            vp_b = all_vp[i:i+batch_size]
            vh_b = all_vh[i:i+batch_size]
            hf = collapse_fn(h0_b)
            logits = head(hf, vp_b, vh_b)
            batch_preds = logits.argmax(dim=-1).tolist()
            preds.extend(batch_preds)
            finals.append(hf.cpu())

    finals = torch.cat(finals, dim=0)

    n = len(all_labels)
    correct = sum(p == l for p, l in zip(preds, all_labels))
    acc = correct / n

    recall = {}
    class_names = ['entail', 'contra', 'neutral']
    for c, name in enumerate(class_names):
        total = sum(1 for l in all_labels if l == c)
        hit   = sum(1 for p, l in zip(preds, all_labels) if l == c and p == c)
        recall[name] = hit / total if total > 0 else 0.0

    return {
        'accuracy': round(acc, 6),
        'recall_entail':  round(recall['entail'],  4),
        'recall_contra':  round(recall['contra'],  4),
        'recall_neutral': round(recall['neutral'], 4),
        'preds': preds,
        'finals': finals,
    }


def basin_agreement(preds_a, preds_b):
    """Fraction of samples where two conditions reach same final prediction."""
    assert len(preds_a) == len(preds_b)
    agree = sum(a == b for a, b in zip(preds_a, preds_b))
    return agree / len(preds_a)


def steps_to_convergence(engine, h0, anchors, threshold=0.01, max_steps=20):
    """
    For a batch h0, find the median step at which ||h_{t+1} - h_t|| < threshold.
    Uses grad-V dynamics internally for a clean measurement.
    """
    h = h0.clone()
    conv_steps = torch.full((h.shape[0],), max_steps, dtype=torch.float32)

    beta = 2.0
    alpha = 0.05
    for step in range(max_steps):
        h_n = F.normalize(h, dim=-1)
        weights = torch.softmax(beta * (h_n @ anchors.T), dim=-1)
        target  = weights @ anchors
        h_norm  = h.norm(p=2, dim=-1, keepdim=True)
        grad_v  = -(target - h_n * (h_n * target).sum(dim=-1, keepdim=True))
        delta   = alpha * h_norm * grad_v
        move    = delta.norm(p=2, dim=-1)   # [B]

        just_converged = (move < threshold) & (conv_steps == max_steps)
        conv_steps[just_converged.cpu()] = step

        h = h - delta
        h_norm2 = h.norm(p=2, dim=-1, keepdim=True)
        h = torch.where(h_norm2 > 10.0, h * (10.0 / (h_norm2 + 1e-8)), h)

    return float(conv_steps.median().item())


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────────────────────────────────────

def print_row(label, res, baseline_acc=None, width=28):
    delta = f"{res['accuracy'] - baseline_acc:+.4f}" if baseline_acc else "  base"
    print(f"  {label:<{width}}  {res['accuracy']:.4f}  {delta:>7}  "
          f"{res['recall_entail']:.4f}  {res['recall_neutral']:.4f}  "
          f"{res['recall_contra']:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--snli-dev',   type=str, required=True)
    parser.add_argument('--n-samples',  type=int, default=-1,
                        help='-1 = full dev set')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--beta',  type=float, default=2.0,
                        help='Energy temperature (from test_gradient_collapse sweep)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Step size (from test_gradient_collapse sweep)')
    parser.add_argument('--svd-rank', type=int, default=16,
                        help='Rank for low-rank SVD projection (condition C/E)')
    parser.add_argument('--lambda-sweep', action='store_true',
                        help='Run condition F: sweep λ ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0}')
    parser.add_argument('--conditions', type=str, default='all',
                        help='Comma-separated: baseline,A,B,C,D,E or all')
    args = parser.parse_args()

    run_all = args.conditions == 'all'
    run_set = set(args.conditions.split(',')) if not run_all else set()

    def should_run(name):
        return run_all or name.lower() in run_set

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
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
    print(f"  beta={args.beta}, alpha={args.alpha}, svd_rank={args.svd_rank}")

    # ── Anchor geometry ────────────────────────────────────────────────────────
    e_dir = F.normalize(engine.anchor_entail.detach(), dim=0)
    c_dir = F.normalize(engine.anchor_contra.detach(), dim=0)
    n_dir = F.normalize(engine.anchor_neutral.detach(), dim=0)
    anchors = torch.stack([e_dir, c_dir, n_dir], dim=0)

    cos_ec = (e_dir * c_dir).sum().item()
    cos_en = (e_dir * n_dir).sum().item()
    cos_cn = (c_dir * n_dir).sum().item()
    print(f"\n  Anchor geometry:")
    print(f"    cos(E,C) = {cos_ec:.4f}  cos(E,N) = {cos_en:.4f}  cos(C,N) = {cos_cn:.4f}")
    ideal = abs(cos_ec) + abs(cos_en) + abs(cos_cn)
    print(f"    |total| = {ideal:.4f}  (lower → more orthogonal anchors)")

    # ── Load and encode data ───────────────────────────────────────────────────
    n_samples = None if args.n_samples == -1 else args.n_samples
    print(f"\nEncoding {'full dev set' if n_samples is None else str(n_samples)+' samples'}...")
    raw = load_snli_data(args.snli_dev, max_samples=n_samples)
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

    # ── Build projection matrices ──────────────────────────────────────────────
    print(f"\nBuilding projection matrices...")
    U_anchor = anchor_projection_matrix(engine).to(device)  # [768, r]
    print(f"  Anchor subspace rank: {U_anchor.shape[1]} "
          f"(expected 3, may be <3 if anchors are collinear)")

    U_svd = svd_projection_matrix(all_h0, rank=args.svd_rank).to(device)  # [768, k]
    print(f"  SVD projection rank: {U_svd.shape[1]}")

    # Variance explained by anchor subspace
    h_proj_anc = project(all_h0, U_anchor)
    var_explained_anchor = (h_proj_anc.var() / all_h0.var()).item()
    h_proj_svd = project(all_h0, U_svd)
    var_explained_svd = (h_proj_svd.var() / all_h0.var()).item()
    print(f"  Variance explained — anchor: {var_explained_anchor:.4f}, "
          f"SVD-{args.svd_rank}: {var_explained_svd:.4f}")

    # ── Helper to wrap collapse+eval ───────────────────────────────────────────
    def eval_condition(name, fn):
        print(f"  Running condition {name}...")
        return run_eval(fn, head, all_h0, all_vp, all_vh, all_labels,
                        device, batch_size=bs)

    results = {}

    # ── Baseline ───────────────────────────────────────────────────────────────
    if should_run('baseline'):
        results['baseline'] = eval_condition(
            'Baseline (Full)',
            lambda h: collapse_full(engine, h)
        )

    # ── Condition A: Grad-V only ───────────────────────────────────────────────
    if should_run('a'):
        results['A_grad_v'] = eval_condition(
            'A: Grad-V (∇V, no proj)',
            lambda h: collapse_grad_v(engine, h, beta=args.beta, alpha=args.alpha)
        )

    # ── Condition B: Anchor projection → head only (no dynamics) ──────────────
    if should_run('b'):
        results['B_anchor_proj_only'] = eval_condition(
            'B: Anchor-proj → head',
            lambda h: project(h, U_anchor)
        )

    # ── Condition C: SVD projection → head only (no dynamics) ─────────────────
    if should_run('c'):
        results['C_svd_proj_only'] = eval_condition(
            f'C: SVD-{args.svd_rank}-proj → head',
            lambda h: project(h, U_svd)
        )

    # ── Condition D: Anchor projection → ∇V dynamics ──────────────────────────
    if should_run('d'):
        results['D_anchor_then_gradv'] = eval_condition(
            'D: Anchor-proj → ∇V',
            lambda h: collapse_proj_then_gradv(engine, h, U_anchor,
                                                beta=args.beta, alpha=args.alpha)
        )

    # ── Condition E: SVD projection → ∇V dynamics ─────────────────────────────
    if should_run('e'):
        results['E_svd_then_gradv'] = eval_condition(
            f'E: SVD-{args.svd_rank}-proj → ∇V',
            lambda h: collapse_proj_then_gradv(engine, h, U_svd,
                                                beta=args.beta, alpha=args.alpha)
        )

    # ── Condition F: λ sweep ───────────────────────────────────────────────────
    if args.lambda_sweep:
        lambdas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        results['F_lambda_sweep'] = {}
        for lam in lambdas:
            key = f'lam_{lam:.2f}'
            results['F_lambda_sweep'][key] = eval_condition(
                f'F: λ={lam:.2f} anchor-mix → ∇V',
                lambda h, l=lam: collapse_mixed_then_gradv(
                    engine, h, U_anchor, lam=l,
                    beta=args.beta, alpha=args.alpha)
            )

    # ── Steps to convergence (conditions A, D, E if run) ──────────────────────
    print("\nMeasuring convergence speed (median steps on 512 samples)...")
    h_sample = all_h0[:512]

    def measure_conv(h0_init):
        return steps_to_convergence(engine, h0_init, anchors)

    if 'A_grad_v' in results:
        results['A_grad_v']['median_conv_steps'] = round(measure_conv(h_sample), 2)
    if 'D_anchor_then_gradv' in results:
        proj_sample = project(h_sample, U_anchor)
        results['D_anchor_then_gradv']['median_conv_steps'] = round(
            measure_conv(proj_sample), 2)
    if 'E_svd_then_gradv' in results:
        proj_sample = project(h_sample, U_svd)
        results['E_svd_then_gradv']['median_conv_steps'] = round(
            measure_conv(proj_sample), 2)

    # ── Basin agreement ────────────────────────────────────────────────────────
    if 'baseline' in results and 'A_grad_v' in results:
        agr = basin_agreement(results['baseline']['preds'],
                              results['A_grad_v']['preds'])
        results['basin_agreement_baseline_vs_A'] = round(agr, 4)
        print(f"\n  Basin agreement — Baseline vs A (Grad-V): {agr:.4f}")

    if 'A_grad_v' in results and 'D_anchor_then_gradv' in results:
        agr = basin_agreement(results['A_grad_v']['preds'],
                              results['D_anchor_then_gradv']['preds'])
        results['basin_agreement_A_vs_D'] = round(agr, 4)
        print(f"  Basin agreement — A vs D (anchor-proj+∇V): {agr:.4f}")

    # ── Print summary table ────────────────────────────────────────────────────
    baseline_acc = results.get('baseline', {}).get('accuracy')

    print(f"\n{'═'*80}")
    print(f"  5-CONDITION ABLATION RESULTS")
    print(f"  checkpoint: {Path(args.checkpoint).name}")
    print(f"  β={args.beta}, α={args.alpha}, SVD rank={args.svd_rank}, N={N}")
    print(f"{'═'*80}")
    print(f"  {'Condition':<30}  {'Acc':>6}  {'Δ':>7}  "
          f"{'E-rec':>6}  {'N-rec':>6}  {'C-rec':>6}")
    print(f"  {'─'*72}")

    order = ['baseline', 'A_grad_v', 'B_anchor_proj_only', 'C_svd_proj_only',
             'D_anchor_then_gradv', 'E_svd_then_gradv']
    labels = {
        'baseline':             'Baseline (Full MLP)',
        'A_grad_v':             'A: Grad-V only',
        'B_anchor_proj_only':   'B: Anchor-proj → head',
        'C_svd_proj_only':     f'C: SVD-{args.svd_rank} → head',
        'D_anchor_then_gradv':  'D: Anchor-proj → ∇V',
        'E_svd_then_gradv':    f'E: SVD-{args.svd_rank} → ∇V',
    }

    for key in order:
        if key in results:
            print_row(labels[key], results[key], baseline_acc)

    if args.lambda_sweep and 'F_lambda_sweep' in results:
        print(f"  {'─'*72}")
        for lam_key, res in results['F_lambda_sweep'].items():
            lam_val = float(lam_key.split('_')[1])
            print_row(f'F: λ={lam_val:.2f} anchor-mix → ∇V', res, baseline_acc)

    print(f"{'═'*80}")

    # ── Interpretation ─────────────────────────────────────────────────────────
    print(f"\n  INTERPRETATION")
    print(f"  {'─'*40}")

    if 'D_anchor_then_gradv' in results and baseline_acc:
        d_acc = results['D_anchor_then_gradv']['accuracy']
        delta = d_acc - baseline_acc
        if delta > 0.002:
            print(f"  ✓ D > Baseline (+{delta*100:.2f}pp): anchor-plane contraction")
            print(f"    helps dynamics. Nuisance dimensions are being pruned.")
        elif abs(delta) <= 0.002:
            print(f"  ∼ D ≈ Baseline ({delta:+.4f}): dynamics already implicitly")
            print(f"    exploit the anchor subspace. Projection is neutral.")
        else:
            print(f"  ✗ D < Baseline ({delta:+.4f}pp): anchor projection loses")
            print(f"    information needed by the dynamics (neutral may need")
            print(f"    extra-subspace dimensions).")

    if 'D_anchor_then_gradv' in results and 'A_grad_v' in results:
        d = results['D_anchor_then_gradv']['accuracy']
        a = results['A_grad_v']['accuracy']
        delta = d - a
        if delta > 0.001:
            print(f"  ✓ D > A (+{delta*100:.2f}pp): projection helps ∇V dynamics.")
            print(f"    Contraction IS doing something beyond plain grad-V.")
        elif abs(delta) <= 0.001:
            print(f"  ∼ D ≈ A: projection adds nothing to ∇V. The energy landscape")
            print(f"    already selects the right subspace implicitly.")
        else:
            print(f"  ✗ D < A: projection hurts ∇V dynamics. Don't project.")

    neutral_rec_base = results.get('baseline', {}).get('recall_neutral', None)
    neutral_rec_d    = results.get('D_anchor_then_gradv', {}).get('recall_neutral', None)
    if neutral_rec_base and neutral_rec_d:
        delta_n = neutral_rec_d - neutral_rec_base
        print(f"\n  Neutral recall: Baseline={neutral_rec_base:.4f}  D={neutral_rec_d:.4f}  "
              f"Δ={delta_n:+.4f}")
        if delta_n > 0.005:
            print(f"  ✓ Neutral recall improves under anchor projection — the neutral")
            print(f"    anchor captures structure that the MLP was suppressing.")
        else:
            print(f"  ∼ Neutral recall flat or worse — anchor plane may not fully")
            print(f"    span the neutral manifold (see λ sweep results).")

    print(f"\n{'═'*80}")

    # ── Save results ───────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint).parent
    out_path = ckpt_dir / 'ablation_results.json'

    # Remove tensor data before saving
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_results[k] = {
                kk: vv for kk, vv in v.items()
                if not isinstance(vv, (torch.Tensor, list)) or kk != 'finals'
            }
            # strip preds too (large) unless you want them
            save_results[k].pop('preds', None)
        else:
            save_results[k] = v

    # Add config
    save_results['_config'] = {
        'checkpoint': str(args.checkpoint),
        'n_samples': N,
        'beta': args.beta,
        'alpha': args.alpha,
        'svd_rank': args.svd_rank,
        'anchor_geometry': {
            'cos_EC': round(cos_ec, 4),
            'cos_EN': round(cos_en, 4),
            'cos_CN': round(cos_cn, 4),
        },
        'variance_explained': {
            'anchor_subspace': round(var_explained_anchor, 4),
            f'svd_rank_{args.svd_rank}': round(var_explained_svd, 4),
        }
    }

    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()
