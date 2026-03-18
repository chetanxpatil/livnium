"""
ACO Collapse Test — Ant Colony Phenomena for Livnium

Tests whether running K particles in parallel (explorers + exploiters)
during inference improves accuracy over a single deterministic collapse.

Idea:
  - Exploiter particles: low noise, greedy collapse toward dominant basin
  - Explorer particles:  high noise, wander further before settling
  - Vote: average alignment signals across all K particles → argmax

Metrics reported:
  - Overall accuracy:  single vs ACO
  - Per-class recall:  especially neutral (the weak spot)
  - Confidence delta:  entropy of vote distribution (lower = more certain)
  - Win/loss/tie breakdown per class

Usage:
  python3 test_aco_collapse.py \\
      --checkpoint ../../../pretrained/bert-joint/best_model.pt \\
      --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \\
      --n-samples  1000 \\
      --k-particles 16 \\
      --n-explorers 8 \\
      --explorer-noise 0.3
"""

import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Path setup
_here = Path(__file__).resolve().parent
_repo = _here.parent
_root = _here.parent.parent.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from core import VectorCollapseEngine, BasinField
from tasks.snli import SNLIEncoder, SNLIHead, LivniumNativeEncoder
from utils.vocab import Vocabulary
from train import load_snli_data


# ──────────────────────────────────────────────────────────────
# ACO collapse: K particles, mixed explorer/exploiter
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def aco_collapse(
    engine: VectorCollapseEngine,
    h0: torch.Tensor,          # [B, dim]
    k_particles: int = 16,
    n_explorers: int = 8,
    explorer_noise: float = 0.3,
    exploiter_noise: float = 0.02,
) -> torch.Tensor:
    """
    Run K particles from h0. Each particle collapses independently.
    Explorer particles get high noise; exploiters get tiny noise.

    Returns:
        h_final_avg: [B, dim] — mean collapsed state across all K particles.

    The caller passes this through the trained head (with v_p, v_h) so the
    full learned classification geometry is preserved. This is the correct
    ACO test: particles explore the attractor landscape, we classify from
    wherever the colony converged on average.
    """
    B, dim = h0.shape
    device = h0.device

    h_sum = torch.zeros(B, dim, device=device)

    for p in range(k_particles):
        is_explorer = p < n_explorers
        noise_scale = explorer_noise if is_explorer else exploiter_noise

        h = h0.clone()
        if noise_scale > 0.0:
            h = h + noise_scale * torch.randn_like(h)

        h_final, _ = engine.collapse(h)
        h_sum += h_final

    return h_sum / k_particles   # mean colony endpoint: [B, dim]


def logit_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax(logits). Lower = more certain."""
    probs = torch.softmax(logits, dim=-1)
    return -(probs * (probs + 1e-8).log()).sum(dim=-1)


# ──────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────

def accuracy_and_cm(preds: List[int], labels: List[int]):
    n = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    recall = []
    for c in range(3):
        row_sum = sum(cm[c])
        recall.append(cm[c][c] / row_sum if row_sum > 0 else 0.0)
    return correct / n, cm, recall


def entropy_stats(entropies: List[float]):
    arr = torch.tensor(entropies)
    return arr.mean().item(), arr.std().item()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ACO collapse phenomenon test')
    parser.add_argument('--checkpoint',    type=str, required=True)
    parser.add_argument('--snli-dev',      type=str, required=True)
    parser.add_argument('--n-samples',     type=int, default=1000,
                        help='Number of dev samples to test on')
    parser.add_argument('--batch-size',    type=int, default=64)
    # ACO params
    parser.add_argument('--k-particles',   type=int, default=16,
                        help='Total number of particles per sample')
    parser.add_argument('--n-explorers',   type=int, default=8,
                        help='Number of high-noise explorer particles')
    parser.add_argument('--explorer-noise',  type=float, default=0.3,
                        help='Noise std for explorer particles')
    parser.add_argument('--exploiter-noise', type=float, default=0.02,
                        help='Noise std for exploiter particles')
    # Sweep mode: test multiple noise levels automatically
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep explorer_noise from 0.05 to 0.8 and report table')
    # Stability characterization: flip rate for correct vs wrong predictions
    parser.add_argument('--stability-test', action='store_true',
                        help='Measure basin stability split by correct vs wrong predictions')
    parser.add_argument('--stability-trials', type=int, default=20,
                        help='Number of noise trials per sample for stability test')
    parser.add_argument('--stability-noise', type=float, default=0.3,
                        help='Noise std for stability perturbation trials')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load checkpoint ──────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = ckpt['args']
    vocab = ckpt.get('vocab', None)

    collapse_engine = VectorCollapseEngine(
        dim=model_args.dim,
        num_layers=model_args.num_layers,
        strength_entail=getattr(model_args, 'strength_entail', 0.1),
        strength_contra=getattr(model_args, 'strength_contra', 0.1),
        strength_neutral=getattr(model_args, 'strength_neutral', 0.05),
        strength_neutral_boost=getattr(model_args, 'strength_neutral_boost', 0.05),
    ).to(device)
    collapse_engine.load_state_dict(ckpt['collapse_engine'])
    collapse_engine.eval()

    # Build encoder (we only need h0 — the encoder's output)
    enc_type = getattr(model_args, 'encoder_type', 'bow')
    print(f"Encoder type: {enc_type}")

    if enc_type == 'livnium':
        encoder = LivniumNativeEncoder(
            vocab_size=len(vocab),
            dim=model_args.livnium_dim,
            n_layers=model_args.livnium_layers,
            n_head=model_args.livnium_nhead,
            ff_mult=getattr(model_args, 'livnium_ff_mult', 4),
            cross_encoder=getattr(model_args, 'livnium_cross_encoder', True),
            max_len=512,
        ).to(device)
    elif enc_type in ('bert', 'bert-joint', 'bert-cross'):
        from tasks.snli import BERTSNLIEncoder, CrossEncoderBERTSNLIEncoder
        if enc_type == 'bert-cross':
            encoder = CrossEncoderBERTSNLIEncoder(freeze=False).to(device)
        else:
            encoder = BERTSNLIEncoder(freeze=False).to(device)
    else:
        encoder = SNLIEncoder(vocab=vocab, dim=model_args.dim).to(device)

    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    head = SNLIHead(dim=model_args.dim).to(device)
    head.load_state_dict(ckpt['head'])
    head.eval()

    print(f"  collapse engine: {model_args.dim}d × {model_args.num_layers} layers")

    # ── Load dev data ────────────────────────────────────────
    print(f"\nLoading dev data (max {args.n_samples} samples)...")
    is_bert = getattr(encoder, 'is_bert', False)
    raw = load_snli_data(args.snli_dev, max_samples=args.n_samples)

    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    label_names = ['Entail', 'Contra', 'Neutral']  # matches 0=E,1=C,2=N

    # ── Collect h0 vectors and true labels ──────────────────
    print("Collecting h0 vectors...")
    all_h0, all_vp, all_vh, all_labels = [], [], [], []

    bs = args.batch_size
    n_batches = math.ceil(len(raw) / bs)

    with torch.no_grad():
        for b in tqdm(range(n_batches), desc='Encoding'):
            batch_samples = raw[b * bs : (b + 1) * bs]
            premises    = [s['premise']    for s in batch_samples]
            hypotheses  = [s['hypothesis'] for s in batch_samples]
            labels_b    = [label_map.get(s['gold_label'], 2) for s in batch_samples]

            if is_bert:
                # BERT encoders: takes raw string lists, returns (h0, v_p, v_h)
                h0, v_p, v_h = encoder.build_initial_state(
                    premises, hypotheses, device=device, add_noise=False
                )
            else:
                # Vocab-based encoders: encode to token IDs first
                max_len = 128
                prem_ids = torch.tensor(
                    [vocab.encode(p, max_len=max_len) for p in premises],
                    dtype=torch.long, device=device
                )
                hyp_ids = torch.tensor(
                    [vocab.encode(h, max_len=max_len) for h in hypotheses],
                    dtype=torch.long, device=device
                )
                h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)

            all_h0.append(h0.cpu())
            all_vp.append(v_p.cpu())
            all_vh.append(v_h.cpu())
            all_labels.extend(labels_b)

    all_h0 = torch.cat(all_h0).to(device)
    all_vp = torch.cat(all_vp).to(device)
    all_vh = torch.cat(all_vh).to(device)
    N = len(all_labels)
    print(f"  {N} samples encoded")

    # ── Baseline: single particle → head ────────────────────
    print("\nRunning baseline (single particle, trained head)...")
    baseline_preds, baseline_ent = [], []
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            h0_b = all_h0[i:i+args.batch_size]
            vp_b = all_vp[i:i+args.batch_size]
            vh_b = all_vh[i:i+args.batch_size]
            h_final, _ = collapse_engine.collapse(h0_b)
            logits = head(h_final, vp_b, vh_b)
            baseline_preds.extend(logits.argmax(dim=-1).tolist())
            baseline_ent.extend(logit_entropy(logits).tolist())

    base_acc, base_cm, base_recall = accuracy_and_cm(baseline_preds, all_labels)
    base_ent_mean, base_ent_std = entropy_stats(baseline_ent)

    # ── ACO sweep or single run ───────────────────────────────
    if args.sweep:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
        print(f"\n{'─'*72}")
        print(f"  ACO SWEEP  (K={args.k_particles}, "
              f"explorers={args.n_explorers}/{args.k_particles})")
        print(f"{'─'*72}")
        print(f"  {'noise':>6}  {'acc':>6}  {'Δacc':>6}  "
              f"{'E-rec':>6}  {'N-rec':>6}  {'C-rec':>6}  {'entropy':>8}")
        print(f"  {'baseline':>6}  {base_acc:.4f}  {'—':>6}  "
              f"{base_recall[0]:.4f}  {base_recall[2]:.4f}  "
              f"{base_recall[1]:.4f}  {base_ent_mean:.4f}")
        print(f"  {'─'*66}")

        for noise in noise_levels:
            aco_preds, aco_ent = [], []
            with torch.no_grad():
                for i in range(0, N, args.batch_size):
                    h0_b = all_h0[i:i+args.batch_size]
                    vp_b  = all_vp[i:i+args.batch_size]
                    vh_b  = all_vh[i:i+args.batch_size]
                    h_avg = aco_collapse(
                        collapse_engine, h0_b,
                        k_particles=args.k_particles,
                        n_explorers=args.n_explorers,
                        explorer_noise=noise,
                        exploiter_noise=0.0,
                    )
                    logits = head(h_avg, vp_b, vh_b)
                    aco_preds.extend(logits.argmax(dim=-1).tolist())
                    aco_ent.extend(logit_entropy(logits).tolist())

            aco_acc, aco_cm, aco_recall = accuracy_and_cm(aco_preds, all_labels)
            aco_ent_mean, _ = entropy_stats(aco_ent)
            delta = aco_acc - base_acc
            sign = '+' if delta >= 0 else ''
            print(f"  {noise:>6.2f}  {aco_acc:.4f}  {sign}{delta:.4f}  "
                  f"{aco_recall[0]:.4f}  {aco_recall[2]:.4f}  "
                  f"{aco_recall[1]:.4f}  {aco_ent_mean:.4f}")

        print(f"{'─'*72}")
        print("  Columns: noise=explorer noise std | N-rec=Neutral recall | "
              "entropy=avg vote entropy")

    else:
        # Single run with specified params
        print(f"\nRunning ACO (K={args.k_particles}, "
              f"explorers={args.n_explorers}, "
              f"explorer_noise={args.explorer_noise})...")
        aco_preds, aco_ent = [], []
        with torch.no_grad():
            for i in tqdm(range(0, N, args.batch_size), desc='ACO collapse'):
                h0_b = all_h0[i:i+args.batch_size]
                vp_b  = all_vp[i:i+args.batch_size]
                vh_b  = all_vh[i:i+args.batch_size]
                h_avg = aco_collapse(
                    collapse_engine, h0_b,
                    k_particles=args.k_particles,
                    n_explorers=args.n_explorers,
                    explorer_noise=args.explorer_noise,
                    exploiter_noise=args.exploiter_noise,
                )
                logits = head(h_avg, vp_b, vh_b)
                aco_preds.extend(logits.argmax(dim=-1).tolist())
                aco_ent.extend(logit_entropy(logits).tolist())

        aco_acc, aco_cm, aco_recall = accuracy_and_cm(aco_preds, all_labels)
        aco_ent_mean, aco_ent_std = entropy_stats(aco_ent)

        # ── Print results ──────────────────────────────────
        print(f"\n{'═'*60}")
        print(f"  ACO COLLAPSE PHENOMENON TEST")
        print(f"{'═'*60}")
        print(f"\n  Samples:      {N}")
        print(f"  K particles:  {args.k_particles}  "
              f"({args.n_explorers} explorers @ noise={args.explorer_noise}, "
              f"{args.k_particles - args.n_explorers} exploiters @ noise={args.exploiter_noise})")

        delta = aco_acc - base_acc
        sign = '+' if delta >= 0 else ''

        print(f"\n  {'Method':<22}  {'Acc':>6}  "
              f"{'E-recall':>8}  {'N-recall':>8}  {'C-recall':>8}  {'Entropy':>8}")
        print(f"  {'─'*70}")
        print(f"  {'Baseline (1 particle)':<22}  {base_acc:.4f}  "
              f"{base_recall[0]:>8.4f}  {base_recall[2]:>8.4f}  "
              f"{base_recall[1]:>8.4f}  {base_ent_mean:>8.4f}")
        print(f"  {'ACO (K particles)':<22}  {aco_acc:.4f}  "
              f"{aco_recall[0]:>8.4f}  {aco_recall[2]:>8.4f}  "
              f"{aco_recall[1]:>8.4f}  {aco_ent_mean:>8.4f}")
        print(f"  {'─'*70}")
        print(f"  {'Delta':<22}  {sign}{delta:.4f}  "
              f"{base_recall[0]-aco_recall[0]:>+8.4f}  "
              f"{base_recall[2]-aco_recall[2]:>+8.4f}  "
              f"{base_recall[1]-aco_recall[1]:>+8.4f}  "
              f"{base_ent_mean-aco_ent_mean:>+8.4f}")

        # Win/loss/tie
        wins = sum(1 for p, a, l in zip(baseline_preds, aco_preds, all_labels)
                   if a == l and p != l)
        losses = sum(1 for p, a, l in zip(baseline_preds, aco_preds, all_labels)
                     if p == l and a != l)
        ties_right = sum(1 for p, a, l in zip(baseline_preds, aco_preds, all_labels)
                         if p == l and a == l)
        ties_wrong = sum(1 for p, a, l in zip(baseline_preds, aco_preds, all_labels)
                         if p != l and a != l)

        print(f"\n  Win/loss breakdown:")
        print(f"    ACO fixed baseline wrong:  {wins:4d}  (+{wins/N*100:.1f}%)")
        print(f"    ACO broke baseline right:  {losses:4d}  (-{losses/N*100:.1f}%)")
        print(f"    Both right:                {ties_right:4d}")
        print(f"    Both wrong:                {ties_wrong:4d}")

        # Per-class win/loss
        print(f"\n  Per-class ACO vs Baseline wins (ACO right, baseline wrong):")
        for c, name in enumerate(label_names):
            c_wins = sum(1 for p, a, l in zip(baseline_preds, aco_preds, all_labels)
                         if l == c and a == l and p != l)
            c_losses = sum(1 for p, a, l in zip(baseline_preds, aco_preds, all_labels)
                           if l == c and p == l and a != l)
            print(f"    {name:<8}: wins={c_wins:3d}  losses={c_losses:3d}  "
                  f"net={c_wins-c_losses:+d}")

        print(f"\n  Verdict: ", end='')
        if delta > 0.002:
            print(f"✓ ACO gains {sign}{delta*100:.2f}% — phenomenon confirmed!")
        elif delta > -0.002:
            print(f"~ Neutral effect — ACO matches baseline (noise may need tuning)")
        else:
            print(f"✗ ACO hurts accuracy — explorer noise too high or K too small")

        print(f"\n  Tip: run with --sweep to scan noise levels automatically")
        print(f"{'═'*60}\n")

    # ── Stability characterization ────────────────────────────
    if args.stability_test:
        print(f"\n{'═'*60}")
        print(f"  BASIN STABILITY TEST")
        print(f"  noise={args.stability_noise}  trials={args.stability_trials}")
        print(f"{'═'*60}")
        print(f"  Question: are wrong predictions in equally stable basins")
        print(f"  as correct ones, or do they sit near basin boundaries?\n")

        # Get baseline predictions first (single deterministic collapse)
        base_preds_all = []
        base_ents_all  = []
        with torch.no_grad():
            for i in range(0, N, args.batch_size):
                h0_b = all_h0[i:i+args.batch_size]
                vp_b = all_vp[i:i+args.batch_size]
                vh_b = all_vh[i:i+args.batch_size]
                h_final, _ = collapse_engine.collapse(h0_b)
                logits = head(h_final, vp_b, vh_b)
                base_preds_all.extend(logits.argmax(dim=-1).tolist())
                base_ents_all.extend(logit_entropy(logits).tolist())

        # For each sample: run T noisy trials, count how often prediction flips
        # flip_rate[i] = fraction of trials where pred != base_pred[i]
        print(f"  Running {args.stability_trials} noise trials per sample...")
        flip_counts = [0] * N

        with torch.no_grad():
            for trial in tqdm(range(args.stability_trials), desc='Stability trials'):
                for i in range(0, N, args.batch_size):
                    h0_b = all_h0[i:i+args.batch_size]
                    vp_b = all_vp[i:i+args.batch_size]
                    vh_b = all_vh[i:i+args.batch_size]
                    h_noisy = h0_b + args.stability_noise * torch.randn_like(h0_b)
                    h_final, _ = collapse_engine.collapse(h_noisy)
                    logits = head(h_final, vp_b, vh_b)
                    trial_preds = logits.argmax(dim=-1).tolist()
                    for j, (tp, bp) in enumerate(zip(trial_preds,
                                                     base_preds_all[i:i+args.batch_size])):
                        if tp != bp:
                            flip_counts[i + j] += 1

        flip_rates = [fc / args.stability_trials for fc in flip_counts]

        # Split by correct vs wrong baseline predictions
        correct_flips  = [flip_rates[i] for i in range(N) if base_preds_all[i] == all_labels[i]]
        wrong_flips    = [flip_rates[i] for i in range(N) if base_preds_all[i] != all_labels[i]]
        correct_ents   = [base_ents_all[i] for i in range(N) if base_preds_all[i] == all_labels[i]]
        wrong_ents     = [base_ents_all[i] for i in range(N) if base_preds_all[i] != all_labels[i]]

        def mean_std(lst):
            t = torch.tensor(lst)
            return t.mean().item(), t.std().item()

        cf_mean, cf_std = mean_std(correct_flips)
        wf_mean, wf_std = mean_std(wrong_flips)
        ce_mean, _      = mean_std(correct_ents)
        we_mean, _      = mean_std(wrong_ents)

        print(f"\n  {'Group':<22}  {'n':>5}  {'flip_rate':>10}  {'entropy':>10}")
        print(f"  {'─'*52}")
        print(f"  {'Correct predictions':<22}  {len(correct_flips):>5}  "
              f"{cf_mean:>8.4f}±{cf_std:.3f}  {ce_mean:>10.4f}")
        print(f"  {'Wrong predictions':<22}  {len(wrong_flips):>5}  "
              f"{wf_mean:>8.4f}±{wf_std:.3f}  {we_mean:>10.4f}")
        print(f"  {'─'*52}")
        ratio = wf_mean / cf_mean if cf_mean > 1e-6 else float('inf')
        print(f"  Wrong/correct flip ratio: {ratio:.2f}x")

        # Per-class flip rates
        class_names = ['Entail (0)', 'Contra (1)', 'Neutral (2)']
        print(f"\n  Per-class flip rates (true label):")
        print(f"  {'Class':<14}  {'n':>5}  {'correct_flip':>13}  {'wrong_flip':>13}  {'ratio':>7}")
        print(f"  {'─'*58}")
        for c, cname in enumerate(class_names):
            c_correct = [flip_rates[i] for i in range(N)
                         if all_labels[i] == c and base_preds_all[i] == c]
            c_wrong   = [flip_rates[i] for i in range(N)
                         if all_labels[i] == c and base_preds_all[i] != c]
            if not c_correct:
                continue
            ccf_mean = sum(c_correct) / len(c_correct)
            cwf_mean = sum(c_wrong)   / len(c_wrong) if c_wrong else float('nan')
            r = cwf_mean / ccf_mean if ccf_mean > 1e-6 and c_wrong else float('nan')
            print(f"  {cname:<14}  {len(c_correct):>5}c {len(c_wrong):>4}w  "
                  f"{ccf_mean:>13.4f}  {cwf_mean:>13.4f}  {r:>7.2f}x")

        # Stability histogram (bucket flip_rate into 0%, 1-10%, 11-30%, >30%)
        def bucket(rates):
            b0  = sum(1 for r in rates if r == 0.0)
            b10 = sum(1 for r in rates if 0.0 < r <= 0.1)
            b30 = sum(1 for r in rates if 0.1 < r <= 0.3)
            bhi = sum(1 for r in rates if r > 0.3)
            return b0, b10, b30, bhi

        cb = bucket(correct_flips)
        wb = bucket(wrong_flips)
        print(f"\n  Flip rate distribution (fraction of trials that flipped):")
        print(f"  {'Group':<22}  {'0%':>6}  {'1-10%':>6}  {'11-30%':>7}  {'>30%':>6}")
        print(f"  {'─'*56}")
        print(f"  {'Correct':<22}  {cb[0]:>6}  {cb[1]:>6}  {cb[2]:>7}  {cb[3]:>6}")
        print(f"  {'Wrong':<22}  {wb[0]:>6}  {wb[1]:>6}  {wb[2]:>7}  {wb[3]:>6}")

        print(f"\n  Interpretation:")
        if ratio > 2.0:
            print(f"  ✓ Wrong predictions are {ratio:.1f}x more likely to flip under noise.")
            print(f"    This means errors sit near basin boundaries — the model")
            print(f"    is uncertain where it's wrong. Geometry is meaningful.")
        elif ratio > 1.3:
            print(f"  ~ Weak signal: wrong predictions flip {ratio:.1f}x more than correct.")
            print(f"    Some boundary sensitivity, but basins are not sharply separated.")
        else:
            print(f"  ✗ Wrong predictions are as stable as correct ones ({ratio:.1f}x).")
            print(f"    The model is confidently wrong — systematic geometric bias,")
            print(f"    not boundary ambiguity. Errors are in deep wrong basins.")
        print(f"{'═'*60}\n")


if __name__ == '__main__':
    main()
