"""
Tunnel Test — Trajectory Debugging for Neutral-Sponge Errors
=============================================================
For each neutral sample that the model gets wrong, this script traces the
full collapse trajectory step by step:

  h₀ → h₁ → h₂ → ... → h_L

At each step it records:
  • cos(h_t, A_E),  cos(h_t, A_C),  cos(h_t, A_N)   — basin alignments
  • V(h_t) = -logsumexp(β · alignments)              — energy at this step
  • dominant basin at this step                       — which label is winning

From this it classifies each failure as one of three types:
  TYPE-1  Bad h₀ (Law 1)    — wrong basin dominant from step 0 — bad initial state
  TYPE-2  Mid-diversion (Law 3) — correct basin leads, then loses during collapse
  TYPE-3  Boundary stall (Law 2) — stuck near basin boundary throughout; never commits

Summary statistics across all neutral errors tell you which law is breaking down most.

Usage:
  python3 tunnel_test.py \\
      --checkpoint ../../../pretrained/bert-joint/best_model.pt \\
      --snli-dev   ../../../data/snli/snli_1.0_dev.jsonl \\
      --n-samples  2000 \\
      --beta 1.0 \\
      --alpha 0.2 \\
      --show 5            # print full trajectory for this many error examples
      --class neutral     # focus on neutral errors (default); or: entail / contra / all
      --mode full         # collapse mode: full | grad-v
"""

import sys
import math
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

_here = Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from core import VectorCollapseEngine
from tasks.snli import BERTSNLIEncoder, SNLIHead
from train import load_snli_data


LABEL_NAMES = ['E', 'C', 'N']
LABEL_FULL  = ['entailment', 'contradiction', 'neutral']


# ──────────────────────────────────────────────────────────────────────────────
# Energy and gradient
# ──────────────────────────────────────────────────────────────────────────────

def energy(h_n: torch.Tensor, anchors: torch.Tensor, beta: float) -> torch.Tensor:
    """V(h) = -logsumexp(β · cos(h, anchors))  [B] or scalar."""
    alignments = h_n @ anchors.T          # [B, 3]
    return -torch.logsumexp(beta * alignments, dim=-1)  # [B]


def grad_v_step(h: torch.Tensor, anchors: torch.Tensor,
                beta: float, alpha: float) -> torch.Tensor:
    """
    One gradient step: h_{t+1} = h_t - alpha * ∇V(h_t)

    Each step creates a fresh leaf tensor (avoids graph accumulation).
    Returns new h — same shape as input.
    """
    h = h.detach()                                       # break any graph history
    h_n = F.normalize(h, dim=-1)                         # [B, dim]
    alignments = h_n @ anchors.T                         # [B, 3]
    weights = torch.softmax(beta * alignments, dim=-1)   # [B, 3]
    target  = weights @ anchors                          # [B, dim]  weighted anchor mix
    # projected gradient on unit sphere tangent plane
    grad = -(target - h_n * (h_n * target).sum(dim=-1, keepdim=True))
    h_norm = h.norm(p=2, dim=-1, keepdim=True)
    h_new  = h - alpha * h_norm * grad
    # norm control
    h_norm2 = h_new.norm(p=2, dim=-1, keepdim=True)
    h_new = torch.where(h_norm2 > 10.0,
                        h_new * (10.0 / (h_norm2 + 1e-8)), h_new)
    return h_new


# ──────────────────────────────────────────────────────────────────────────────
# Single-sample trajectory tracer
# ──────────────────────────────────────────────────────────────────────────────

def trace_trajectory(
    h0: torch.Tensor,       # [dim]  single sample
    engine: VectorCollapseEngine,
    mode: str,
    beta: float,
    alpha: float,
    device: torch.device,
) -> Dict:
    """
    Run collapse step-by-step, recording:
      cos_e[t], cos_c[t], cos_n[t], V[t], dominant[t]

    Returns dict with per-step lists plus summary info.
    """
    e_dir = F.normalize(engine.anchor_entail.detach(), dim=0)   # [dim]
    c_dir = F.normalize(engine.anchor_contra.detach(), dim=0)
    n_dir = F.normalize(engine.anchor_neutral.detach(), dim=0)
    anchors = torch.stack([e_dir, c_dir, n_dir], dim=0)         # [3, dim]

    h = h0.clone().unsqueeze(0)   # [1, dim]
    n_steps = engine.num_layers

    cos_e, cos_c, cos_n, energies, dominant = [], [], [], [], []

    @torch.no_grad()
    def record(h_: torch.Tensor):
        h_n = F.normalize(h_, dim=-1)               # [1, dim]
        a = (h_n @ anchors.T)[0]                    # [3]
        e_val = energy(h_n, anchors, beta)[0].item()
        cos_e.append(a[0].item())
        cos_c.append(a[1].item())
        cos_n.append(a[2].item())
        energies.append(e_val)
        dominant.append(int(a.argmax().item()))

    with torch.no_grad():
        record(h)
        for step in range(n_steps):
            if mode == 'grad-v':
                h = grad_v_step(h, anchors, beta, alpha)
            elif mode == 'full':
                # Use engine's update but step-by-step (replicate one layer)
                h_n = F.normalize(h, dim=-1)
                # learned residual
                delta = engine.update(h)
                # anchor forces
                a_e = (h_n * e_dir).sum(dim=-1)
                a_c = (h_n * c_dir).sum(dim=-1)
                a_n = (h_n * n_dir).sum(dim=-1)
                from core.physics_laws import divergence_from_alignment, boundary_proximity
                d_e = divergence_from_alignment(a_e)
                d_c = divergence_from_alignment(a_c)
                d_n = divergence_from_alignment(a_n)
                ec_b = boundary_proximity(a_e, a_c)
                e_vec = F.normalize(h - e_dir.unsqueeze(0), dim=-1)
                c_vec = F.normalize(h - c_dir.unsqueeze(0), dim=-1)
                n_vec = F.normalize(h - n_dir.unsqueeze(0), dim=-1)
                h = (h + delta
                     - engine.strength_entail * d_e.unsqueeze(-1) * e_vec
                     - engine.strength_contra  * d_c.unsqueeze(-1) * c_vec
                     - engine.strength_neutral * d_n.unsqueeze(-1) * n_vec
                     - engine.strength_neutral_boost * ec_b.unsqueeze(-1) * n_vec)
                h_norm = h.norm(p=2, dim=-1, keepdim=True)
                h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)
            record(h)

    return {
        'cos_e': cos_e, 'cos_c': cos_c, 'cos_n': cos_n,
        'energy': energies, 'dominant': dominant,
        'h_final': h[0],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Failure type classifier
# ──────────────────────────────────────────────────────────────────────────────

def classify_failure(traj: Dict, true_label: int, pred_label: int) -> Tuple[str, str]:
    """
    Classify why the trajectory ended up in the wrong basin.

    Returns (type_code, description)
    """
    dominant = traj['dominant']
    cos_true_at_0 = [traj['cos_e'], traj['cos_c'], traj['cos_n']][true_label][0]
    cos_pred_at_0 = [traj['cos_e'], traj['cos_c'], traj['cos_n']][pred_label][0]

    # Boundary stall: never clearly in any basin — low max cos throughout
    max_cos_per_step = [max(e, c, n) for e, c, n in
                        zip(traj['cos_e'], traj['cos_c'], traj['cos_n'])]
    mean_max_cos = sum(max_cos_per_step) / len(max_cos_per_step)

    if mean_max_cos < 0.15:
        return 'TYPE-3', 'boundary-stall (Law 2) — low commitment throughout'

    # Bad h₀: wrong basin already winning at step 0
    if dominant[0] == pred_label:
        return 'TYPE-1', f'bad-h₀ (Law 1) — wrong basin dominant from step 0'

    # Find when the path switches from correct to wrong
    switch_step = None
    for t, d in enumerate(dominant):
        if d == pred_label and t > 0:
            switch_step = t
            break

    if switch_step is not None:
        return 'TYPE-2', f'mid-diversion (Law 3) — correct at step 0, switched at step {switch_step}'

    # Dominant was right throughout but prediction still wrong (edge case: head geometry)
    return 'TYPE-?', 'unclear — dominant path correct but head disagreed'


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory printer
# ──────────────────────────────────────────────────────────────────────────────

def print_trajectory(
    traj: Dict,
    true_label: int,
    pred_label: int,
    premise: str,
    hypothesis: str,
    failure_type: str,
    failure_desc: str,
    sample_idx: int,
):
    label_chars = {0: 'E', 1: 'C', 2: 'N'}
    n_steps = len(traj['cos_e'])

    print(f"\n{'─'*72}")
    print(f"  Sample #{sample_idx}  |  True: {LABEL_FULL[true_label]}  "
          f"Pred: {LABEL_FULL[pred_label]}  |  {failure_type}")
    print(f"  {failure_desc}")
    print(f"  P: {premise[:80]}")
    print(f"  H: {hypothesis[:80]}")
    print(f"\n  Step  cos(E)   cos(C)   cos(N)   V(h)      Basin")
    print(f"  {'─'*54}")

    for t in range(n_steps):
        d = traj['dominant'][t]
        basin_str = label_chars[d]
        marker = ' ◄ TRUE' if d == true_label and t == n_steps - 1 else \
                 ' ✗ WRONG' if d == pred_label and t == n_steps - 1 else ''
        # highlight the dominant basin
        ce = f"{traj['cos_e'][t]:+.3f}"
        cc = f"{traj['cos_c'][t]:+.3f}"
        cn = f"{traj['cos_n'][t]:+.3f}"
        vals = [ce, cc, cn]
        vals[d] = f"[{vals[d]}]"
        print(f"  {t:>4}  {vals[0]:>8}  {vals[1]:>8}  {vals[2]:>8}  "
              f"{traj['energy'][t]:>8.4f}  {basin_str}{marker}")

    # Energy change summary
    e_drop = traj['energy'][-1] - traj['energy'][0]
    print(f"\n  Energy drop: {e_drop:+.4f}  "
          f"({'↓ descending' if e_drop < 0 else '↑ ascending — unusual'})")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--snli-dev',    type=str, required=True)
    parser.add_argument('--n-samples',   type=int, default=2000)
    parser.add_argument('--batch-size',  type=int, default=64)
    parser.add_argument('--beta',        type=float, default=1.0)
    parser.add_argument('--alpha',       type=float, default=0.2)
    parser.add_argument('--show',        type=int, default=5,
                        help='Number of error trajectories to print in full')
    parser.add_argument('--class',       dest='target_class',
                        choices=['neutral', 'entail', 'contra', 'all'],
                        default='neutral',
                        help='Which error class to inspect')
    parser.add_argument('--mode',        choices=['full', 'grad-v'],
                        default='full',
                        help='Collapse mode for trajectories')
    parser.add_argument('--correct-too', action='store_true',
                        help='Also show trajectories for correct predictions (for comparison)')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
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

    print(f"  dim={model_args.dim}, layers={engine.num_layers}, mode={args.mode}")

    # ── Load and encode dev data ───────────────────────────────────────────────
    print(f"\nEncoding {args.n_samples} dev samples...")
    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    raw = load_snli_data(args.snli_dev, max_samples=args.n_samples)

    all_h0, all_vp, all_vh, all_labels, all_texts = [], [], [], [], []
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
            all_texts.extend(list(zip(premises, hypotheses)))

    all_h0 = torch.cat(all_h0).to(device)
    all_vp = torch.cat(all_vp).to(device)
    all_vh = torch.cat(all_vh).to(device)
    N = len(all_labels)
    print(f"  {N} samples ready")

    # ── Get predictions for all samples ───────────────────────────────────────
    print(f"\nRunning predictions (mode={args.mode})...")
    from test_gradient_collapse import collapse_full, collapse_grad_v

    all_preds = []
    with torch.no_grad():
        for i in range(0, N, bs):
            h0_b = all_h0[i:i+bs]
            vp_b = all_vp[i:i+bs]
            vh_b = all_vh[i:i+bs]
            if args.mode == 'full':
                hf = collapse_full(engine, h0_b)
            else:
                hf = collapse_grad_v(engine, h0_b, beta=args.beta, alpha=args.alpha)
            logits = head(hf, vp_b, vh_b)
            all_preds.extend(logits.argmax(dim=-1).tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / N
    print(f"  Overall accuracy: {acc:.4f}")

    # ── Filter to target class ─────────────────────────────────────────────────
    target_map = {'neutral': 2, 'entail': 0, 'contra': 1}
    if args.target_class == 'all':
        error_indices = [i for i in range(N) if all_preds[i] != all_labels[i]]
        correct_indices = [i for i in range(N) if all_preds[i] == all_labels[i]]
    else:
        tc = target_map[args.target_class]
        error_indices = [i for i in range(N)
                         if all_labels[i] == tc and all_preds[i] != tc]
        correct_indices = [i for i in range(N)
                           if all_labels[i] == tc and all_preds[i] == tc]

    print(f"\n  Target class: {args.target_class.upper()}")
    print(f"  Errors:   {len(error_indices)}")
    print(f"  Correct:  {len(correct_indices)}")
    if len(error_indices) + len(correct_indices) > 0:
        class_acc = len(correct_indices) / (len(error_indices) + len(correct_indices))
        print(f"  Recall:   {class_acc:.4f}")

    # ── Trace trajectories for all errors ─────────────────────────────────────
    print(f"\nTracing trajectories for {len(error_indices)} errors...")
    failure_types = Counter()
    type_examples = defaultdict(list)      # type → list of (idx, traj, desc)
    energy_drops_error = []
    energy_drops_correct = []

    # Trace errors
    for idx in tqdm(error_indices, desc='Tracing errors'):
        h0_i = all_h0[idx]
        true_l = all_labels[idx]
        pred_l = all_preds[idx]
        traj = trace_trajectory(h0_i, engine, args.mode, args.beta, args.alpha, device)
        ft, fd = classify_failure(traj, true_l, pred_l)
        failure_types[ft] += 1
        type_examples[ft].append((idx, traj, fd, true_l, pred_l))
        energy_drops_error.append(traj['energy'][-1] - traj['energy'][0])

    # Optionally trace some correct ones for comparison
    sample_correct = correct_indices[:min(100, len(correct_indices))]
    for idx in tqdm(sample_correct, desc='Tracing correct (sample)'):
        h0_i = all_h0[idx]
        traj = trace_trajectory(h0_i, engine, args.mode, args.beta, args.alpha, device)
        energy_drops_correct.append(traj['energy'][-1] - traj['energy'][0])

    # ── Summary statistics ─────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  TUNNEL TEST — FAILURE ANALYSIS ({args.target_class.upper()} errors, mode={args.mode})")
    print(f"{'═'*72}")

    total_errors = len(error_indices)
    print(f"\n  Failure type breakdown ({total_errors} errors total):")
    print(f"\n  {'Type':<12}  {'Count':>6}  {'%':>6}  {'Description'}")
    print(f"  {'─'*64}")
    for ft in ['TYPE-1', 'TYPE-2', 'TYPE-3', 'TYPE-?']:
        count = failure_types[ft]
        if count == 0:
            continue
        pct = count / total_errors * 100
        desc = {
            'TYPE-1': 'bad h₀ (Law 1) — wrong basin from start',
            'TYPE-2': 'mid-diversion (Law 3) — correct then diverted',
            'TYPE-3': 'boundary stall (Law 2) — never committed',
            'TYPE-?': 'unclear (head geometry mismatch)',
        }[ft]
        print(f"  {ft:<12}  {count:>6}  {pct:>5.1f}%  {desc}")

    if energy_drops_error:
        avg_drop_err = sum(energy_drops_error) / len(energy_drops_error)
        avg_drop_cor = sum(energy_drops_correct) / len(energy_drops_correct) if energy_drops_correct else float('nan')
        print(f"\n  Energy dynamics:")
        print(f"    Error trajectories   — avg V drop: {avg_drop_err:+.4f}")
        print(f"    Correct trajectories — avg V drop: {avg_drop_cor:+.4f} (n={len(energy_drops_correct)})")
        if avg_drop_cor != float('nan') and avg_drop_err > avg_drop_cor:
            print(f"    ⚠ Errors LESS energetically committed ({avg_drop_err-avg_drop_cor:+.4f} shallower drop)")

    # Which labels do neutral errors get misclassified as?
    if args.target_class == 'neutral':
        wrong_as = Counter(all_preds[i] for i in error_indices)
        print(f"\n  Neutral errors classified as:")
        for label_idx, count in wrong_as.most_common():
            print(f"    → {LABEL_FULL[label_idx]}: {count} ({count/total_errors*100:.1f}%)")

    # ── Print example trajectories ─────────────────────────────────────────────
    n_shown = 0
    print(f"\n{'═'*72}")
    print(f"  EXAMPLE TRAJECTORIES (showing {args.show} per type)")
    print(f"{'═'*72}")

    for ft in ['TYPE-1', 'TYPE-2', 'TYPE-3', 'TYPE-?']:
        examples = type_examples[ft]
        if not examples:
            continue
        print(f"\n  ── {ft}: {failure_types[ft]} total ──")
        for idx, traj, fd, true_l, pred_l in examples[:args.show]:
            p, h = all_texts[idx]
            print_trajectory(traj, true_l, pred_l, p, h, ft, fd, idx)
            n_shown += 1

    # ── Correct examples for comparison ───────────────────────────────────────
    if args.correct_too and correct_indices:
        print(f"\n{'═'*72}")
        print(f"  CORRECT TRAJECTORIES (for comparison)")
        print(f"{'═'*72}")
        for idx in correct_indices[:args.show]:
            h0_i = all_h0[idx]
            true_l = all_labels[idx]
            traj = trace_trajectory(h0_i, engine, args.mode, args.beta, args.alpha, device)
            p, h = all_texts[idx]
            print(f"\n{'─'*72}")
            print(f"  Sample #{idx}  |  True: {LABEL_FULL[true_l]}  Pred: ✓ CORRECT")
            print(f"  P: {p[:80]}")
            print(f"  H: {h[:80]}")
            print(f"\n  Step  cos(E)   cos(C)   cos(N)   V(h)      Basin")
            print(f"  {'─'*54}")
            for t in range(len(traj['cos_e'])):
                d = traj['dominant'][t]
                marker = ' ✓' if t == len(traj['cos_e']) - 1 else ''
                ce = f"{traj['cos_e'][t]:+.3f}"
                cc = f"{traj['cos_c'][t]:+.3f}"
                cn = f"{traj['cos_n'][t]:+.3f}"
                vals = [ce, cc, cn]
                vals[d] = f"[{vals[d]}]"
                print(f"  {t:>4}  {vals[0]:>8}  {vals[1]:>8}  {vals[2]:>8}  "
                      f"{traj['energy'][t]:>8.4f}  {LABEL_NAMES[d]}{marker}")

    print(f"\n{'═'*72}")
    print(f"  INTERPRETATION GUIDE:")
    print(f"  TYPE-1 dominates → fix h₀ = v_h - v_p (Law 1 is the bottleneck)")
    print(f"  TYPE-2 dominates → fix collapse dynamics α, β, or step count (Law 3)")
    print(f"  TYPE-3 dominates → neutral basin geometry too narrow (Law 2)")
    print(f"  Mix of types     → multi-law interaction; look at per-sample energy curves")
    print(f"{'═'*72}\n")


if __name__ == '__main__':
    main()
