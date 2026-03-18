"""
train_livnium_joint.py — Joint Retraining under Grad-V Dynamics
================================================================
Retrains BERT encoder + anchors + SNLIHead together under the decisive
grad-V collapse (β=20, α=0.05). Forces the head to see the physically
collapsed h_final, not just raw BERT features.

This is the test of the core hypothesis:
  "Under low-β training, the head learned to compensate for weak dynamics.
   Under joint retraining at high β, does the head learn to trust h_final?"

If TYPE-? rate drops to near zero → head and dynamics are calibrated together.
If neutral recall breaks 80% → the neutral sponge is fixed by decisive geometry.

Key differences from original train.py:
  1. Forward pass uses differentiable grad-V (not VectorCollapseEngine MLP)
  2. Anchors are standalone nn.Parameters, not inside VectorCollapseEngine
  3. Loss = CE + λ_align * anchor_alignment_loss
     anchor_alignment_loss: maximize cos(h_final, A_y) for true label y
     this directly penalizes TYPE-? (head overrides correct collapse)
  4. Per-epoch diagnostic: TYPE-? rate, dynamics/head agreement, neutral recall

Warm start from an existing bert-joint checkpoint (anchors, head, BERT all
transferred). Cold start from scratch also supported (--no-pretrained).

Usage:
  python3 train_livnium_joint.py \\
      --checkpoint   ../../../pretrained/bert-joint/best_model.pt \\
      --snli-train   ../../../data/snli/snli_1.0_train.jsonl \\
      --snli-dev     ../../../data/snli/snli_1.0_dev.jsonl \\
      --output-dir   ../../../pretrained/livnium-joint-v3 \\
      --epochs 5 \\
      --beta 20.0 --alpha 0.05 --steps 6 \\
      --lambda-align 0.3 \\
      --batch-size 32 \\
      --lr 1e-3 --bert-lr 2e-5
"""

import sys
import math
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

_here = Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from tasks.snli import BERTSNLIEncoder, SNLIHead
from train import load_snli_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

LABEL_MAP  = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
LABEL_NAMES = ['E', 'C', 'N']


# ──────────────────────────────────────────────────────────────────────────────
# Core model: BERT + anchors + grad-V collapse + head
# ──────────────────────────────────────────────────────────────────────────────

class LiveniumJoint(nn.Module):
    """
    Joint model where dynamics and head are calibrated together.

    Architecture:
      h₀ = v_h − v_p                        (Law 1)
      V(h) = −logsumexp(β · cos(h, anchors)) (Law 2)
      h_{t+1} = h_t − α · ∇V(h_t)           (Law 3, differentiable)
      logits = head(h_final, v_p, v_h)

    The collapse is differentiable: gradients flow back through all L steps
    to the BERT encoder and to the anchor parameters.
    """

    def __init__(
        self,
        dim: int,
        beta: float = 20.0,
        alpha: float = 0.05,
        steps: int = 6,
        sphere_project: bool = False,
    ):
        super().__init__()
        self.dim   = dim
        self.beta  = beta
        self.alpha = alpha
        self.steps = steps
        self.sphere_project = sphere_project

        # Three class anchors: [E=0, C=1, N=2]
        self.anchors = nn.Parameter(torch.randn(3, dim) * 0.1)
        self.encoder = BERTSNLIEncoder(freeze=False)
        self.head    = SNLIHead(dim=dim)

    def collapse(self, h0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable grad-V collapse: L steps of gradient descent on V(h).

        h stays in the computation graph throughout — gradients flow back
        through all steps to h0 (and thus to the BERT encoder).

        Returns:
            h_final:    [B, dim]  final state after L steps
            alignments: [B, 3]   cosine alignments to each anchor at last step
        """
        h = h0                                               # [B, dim] — in graph
        anchors_n = F.normalize(self.anchors, dim=-1)        # [3, dim]

        for _ in range(self.steps):
            h_n = F.normalize(h, dim=-1)                     # [B, dim]
            alignments = h_n @ anchors_n.T                   # [B, 3]

            # Analytical gradient of V(h) projected onto unit-sphere tangent plane
            # ∇V = −(weighted_target − h_n · (h_n · weighted_target))
            weights = torch.softmax(self.beta * alignments, dim=-1)   # [B, 3]
            target  = weights @ anchors_n                              # [B, dim]
            grad    = -(target - h_n * (h_n * target).sum(dim=-1, keepdim=True))

            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = h - self.alpha * h_norm * grad

            if self.sphere_project:
                # Optional: project back to unit sphere (Riemannian GD on S^{d-1})
                h = F.normalize(h, dim=-1)
            else:
                # Soft norm control: prevent exploding norms
                h_norm2 = h.norm(p=2, dim=-1, keepdim=True)
                h = torch.where(h_norm2 > 10.0,
                                h * (10.0 / (h_norm2 + 1e-8)), h)

        # Final alignments at the last step (for diagnostic + alignment loss)
        h_n_final = F.normalize(h, dim=-1)
        alignments = h_n_final @ anchors_n.T                 # [B, 3]
        return h, alignments

    def forward(
        self,
        premises: List[str],
        hypotheses: List[str],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: text → logits.

        Returns:
            logits:     [B, 3]  classification logits
            h_final:    [B, dim] collapsed state
            alignments: [B, 3]  cosine alignments at final step
            v_p:        [B, dim] premise embedding
            v_h:        [B, dim] hypothesis embedding
        """
        h0, v_p, v_h = self.encoder.build_initial_state(
            premises, hypotheses, add_noise=False, device=device
        )
        h_final, alignments = self.collapse(h0)
        logits = self.head(h_final, v_p, v_h)
        return logits, h_final, alignments, v_p, v_h


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(
    logits:     torch.Tensor,   # [B, 3]
    h_final:    torch.Tensor,   # [B, dim]
    alignments: torch.Tensor,   # [B, 3] cosines at last step
    anchors:    torch.Tensor,   # [3, dim] (raw, not normalized)
    labels:     torch.Tensor,   # [B] int64
    lambda_align: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Joint loss = CrossEntropy + λ_align * AnchorAlignmentLoss

    AnchorAlignmentLoss:
      Maximizes cos(h_final, A_y) for the true label y.
      This directly trains the dynamics to land h_final in the correct basin,
      penalizing the TYPE-? case (dynamics correct but head overrides).
    """
    # 1. Standard classification loss
    loss_ce = F.cross_entropy(logits, labels)

    # 2. Anchor alignment loss: maximize cosine to correct anchor
    #    Uses the pre-computed alignments at the last step [B, 3]
    #    alignments[i, labels[i]] = cos(h_final_i, A_{y_i})
    correct_alignments = alignments.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
    loss_align = -correct_alignments.mean()   # maximize → minimize negative

    total = loss_ce + lambda_align * loss_align

    return total, {
        'ce':    loss_ce.item(),
        'align': loss_align.item(),
        'total': total.item(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Metrics + diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    preds:      List[int],
    labels:     List[int],
    basin_doms: List[int],  # argmax of alignments at final step
) -> Dict:
    """Accuracy, per-class recall, TYPE-? rate (head disagrees with dynamics)."""
    n = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))

    recall = []
    for c in range(3):
        total = sum(1 for l in labels if l == c)
        right = sum(1 for p, l in zip(preds, labels) if l == c and p == c)
        recall.append(right / total if total > 0 else 0.0)

    # TYPE-? rate: dominant basin = true label, but head predicted wrong
    type_q = sum(
        1 for p, l, d in zip(preds, labels, basin_doms)
        if d == l and p != l
    )
    # Dynamics-head agreement: fraction where argmax(alignments) == head pred
    agree = sum(1 for p, d in zip(preds, basin_doms) if p == d)

    return {
        'acc':    correct / n,
        'e_rec':  recall[0],
        'c_rec':  recall[1],
        'n_rec':  recall[2],
        'type_q_rate':   type_q / n,
        'dyn_head_agree': agree / n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Warm start: transfer weights from bert-joint checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def warm_start(model: LiveniumJoint, ckpt_path: str, device: torch.device):
    """
    Transfer weights from a bert-joint checkpoint into LiveniumJoint.

    Transfers:
      - BERT encoder (full fine-tuned weights)
      - Anchors: A_E, A_C, A_N from VectorCollapseEngine
      - SNLIHead (full)
    """
    log.info(f"Warm-starting from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # BERT encoder
    model.encoder.load_state_dict(ckpt['encoder'])
    log.info("  ✓ BERT encoder transferred")

    # Head
    model.head.load_state_dict(ckpt['head'])
    log.info("  ✓ SNLIHead transferred")

    # Anchors: from VectorCollapseEngine
    ce_state = ckpt['collapse_engine']
    with torch.no_grad():
        model.anchors[0].copy_(ce_state['anchor_entail'])   # E
        model.anchors[1].copy_(ce_state['anchor_contra'])   # C
        model.anchors[2].copy_(ce_state['anchor_neutral'])  # N
    log.info("  ✓ Anchors transferred (E, C, N from VectorCollapseEngine)")

    return ckpt.get('args', None)


# ──────────────────────────────────────────────────────────────────────────────
# Save checkpoint (compatible with eval.py via VectorCollapseEngine reconstruction)
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: LiveniumJoint,
    args,
    epoch: int,
    metrics: Dict,
    path: Path,
):
    """
    Save in a format that eval.py can load.
    Reconstructs a VectorCollapseEngine state dict from trained anchors
    so the checkpoint is interoperable with existing eval/infer scripts.
    """
    # Reconstruct collapse_engine state dict with trained anchors + default MLP
    # (MLP weights don't matter — grad-V replaces them — but the format requires them)
    from core import VectorCollapseEngine
    dummy_engine = VectorCollapseEngine(dim=model.dim, num_layers=model.steps)
    with torch.no_grad():
        dummy_engine.anchor_entail.copy_(model.anchors[0])
        dummy_engine.anchor_contra.copy_(model.anchors[1])
        dummy_engine.anchor_neutral.copy_(model.anchors[2])

    torch.save({
        'encoder':        model.encoder.state_dict(),
        'collapse_engine': dummy_engine.state_dict(),
        'head':           model.head.state_dict(),
        'joint_anchors':  model.anchors.detach().cpu(),  # standalone for grad-V
        'args':           args,
        'epoch':          epoch,
        'metrics':        metrics,
        'beta':           model.beta,
        'alpha':          model.alpha,
        'steps':          model.steps,
    }, path)


# ──────────────────────────────────────────────────────────────────────────────
# Eval loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: LiveniumJoint,
    data: list,
    batch_size: int,
    device: torch.device,
) -> Dict:
    model.eval()
    preds, labels, basin_doms = [], [], []

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        premises   = [s['premise']    for s in batch]
        hypotheses = [s['hypothesis'] for s in batch]
        labels_b   = [LABEL_MAP.get(s['gold_label'], 2) for s in batch]

        logits, h_final, alignments, v_p, v_h = model(premises, hypotheses, device)
        preds.extend(logits.argmax(dim=-1).tolist())
        labels.extend(labels_b)
        basin_doms.extend(alignments.argmax(dim=-1).tolist())

    return compute_metrics(preds, labels, basin_doms)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Seed ───────────────────────────────────────────────────────────────────
    if args.seed is not None:
        import random, numpy as np
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        log.info(f"Seed: {args.seed}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else
                              'cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Build model ────────────────────────────────────────────────────────────
    dim = 768   # BERT-base hidden size

    model = LiveniumJoint(
        dim=dim,
        beta=args.beta,
        alpha=args.alpha,
        steps=args.steps,
        sphere_project=args.sphere_project,
    ).to(device)

    original_args = None
    if args.checkpoint and not args.no_pretrained:
        original_args = warm_start(model, args.checkpoint, device)
    else:
        log.info("Cold start — random initialization")

    # ── Optimizer: different LRs for BERT vs head/anchors ──────────────────────
    bert_params  = list(model.encoder.parameters())
    other_params = list(model.head.parameters()) + [model.anchors]

    optimizer = AdamW([
        {'params': bert_params,  'lr': args.bert_lr,  'weight_decay': 1e-2},
        {'params': other_params, 'lr': args.lr,        'weight_decay': 1e-4},
    ])

    # ── Data ───────────────────────────────────────────────────────────────────
    log.info(f"\nLoading data...")
    # Load more than needed when stratifying, so we have enough per class
    load_n = (args.max_train * 4) if (args.stratify and args.max_train) else args.max_train
    train_data = load_snli_data(args.snli_train, max_samples=load_n)
    dev_data   = load_snli_data(args.snli_dev,   max_samples=args.max_dev)

    # Filter out gold_label='-'
    train_data = [s for s in train_data if s['gold_label'] in LABEL_MAP]
    dev_data   = [s for s in dev_data   if s['gold_label'] in LABEL_MAP]

    # Stratified sampling: equal examples per class
    if args.stratify and args.max_train:
        import random
        per_class = args.max_train // 3
        buckets = {0: [], 1: [], 2: []}
        for s in train_data:
            buckets[LABEL_MAP[s['gold_label']]].append(s)
        train_data = []
        for label_idx, bucket in buckets.items():
            random.shuffle(bucket)
            train_data.extend(bucket[:per_class])
        random.shuffle(train_data)
        log.info(f"  Stratified: {per_class}/class × 3 = {len(train_data)} samples")

    log.info(f"  Train: {len(train_data):,}  Dev: {len(dev_data):,}")

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * math.ceil(len(train_data) / args.batch_size))

    # ── Output dir ─────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save args for later eval
    with open(out_dir / 'train_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── Initial eval ───────────────────────────────────────────────────────────
    log.info(f"\nInitial eval (before training)...")
    m = evaluate(model, dev_data, args.batch_size, device)
    log.info(f"  acc={m['acc']:.4f}  N={m['n_rec']:.4f}  "
             f"agree={m['dyn_head_agree']:.4f}  TYPE-?={m['type_q_rate']:.4f}")

    best_acc = m['acc']
    best_epoch = 0
    history = []

    # ── Training epochs ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()

        # Shuffle
        import random
        random.shuffle(train_data)

        epoch_loss = 0.0
        epoch_ce   = 0.0
        epoch_align= 0.0
        n_batches  = 0
        t0 = time.time()

        bar = tqdm(range(0, len(train_data), args.batch_size),
                   desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for i in bar:
            batch = train_data[i:i+args.batch_size]
            premises   = [s['premise']    for s in batch]
            hypotheses = [s['hypothesis'] for s in batch]
            labels     = torch.tensor(
                [LABEL_MAP.get(s['gold_label'], 2) for s in batch],
                device=device, dtype=torch.long
            )

            optimizer.zero_grad()

            logits, h_final, alignments, v_p, v_h = model(premises, hypotheses, device)

            loss, loss_parts = compute_loss(
                logits, h_final, alignments,
                model.anchors, labels,
                lambda_align=args.lambda_align,
            )

            loss.backward()

            # Gradient clipping (important for BERT fine-tuning stability)
            nn.utils.clip_grad_norm_(bert_params, 1.0)
            nn.utils.clip_grad_norm_(other_params, 5.0)

            optimizer.step()
            scheduler.step()

            epoch_loss  += loss_parts['total']
            epoch_ce    += loss_parts['ce']
            epoch_align += loss_parts['align']
            n_batches   += 1

            bar.set_postfix({
                'ce':    f"{loss_parts['ce']:.3f}",
                'align': f"{loss_parts['align']:.3f}",
            })

        dt = time.time() - t0

        # ── Dev eval ──────────────────────────────────────────────────────────
        m = evaluate(model, dev_data, args.batch_size, device)

        log.info(
            f"\nEpoch {epoch:2d}/{args.epochs}  "
            f"loss={epoch_loss/n_batches:.4f} "
            f"(ce={epoch_ce/n_batches:.3f} align={epoch_align/n_batches:.3f})  "
            f"time={dt:.0f}s"
        )
        log.info(
            f"  Dev:  acc={m['acc']:.4f}  "
            f"E={m['e_rec']:.4f}  C={m['c_rec']:.4f}  N={m['n_rec']:.4f}"
        )
        log.info(
            f"  Diag: dyn/head-agree={m['dyn_head_agree']:.4f}  "
            f"TYPE-?={m['type_q_rate']:.4f}"
        )

        # Anchor geometry
        with torch.no_grad():
            an = F.normalize(model.anchors, dim=-1)
            ec = (an[0] * an[1]).sum().item()
            en = (an[0] * an[2]).sum().item()
            cn = (an[1] * an[2]).sum().item()
        log.info(f"  Anchors: cos(E,C)={ec:.3f}  cos(E,N)={en:.3f}  cos(C,N)={cn:.3f}")

        # What labels are TYPE-? errors?
        if m['type_q_rate'] > 0.01:
            log.info(f"  Note: TYPE-? still {m['type_q_rate']*100:.1f}% — head not fully trusting dynamics")
        else:
            log.info(f"  ✓ TYPE-? near zero — head and dynamics synchronized")

        history.append({'epoch': epoch, **m,
                        'loss': epoch_loss/n_batches,
                        'loss_ce': epoch_ce/n_batches,
                        'loss_align': epoch_align/n_batches})

        # Save best
        if m['acc'] >= best_acc:
            best_acc = m['acc']
            best_epoch = epoch
            save_checkpoint(model, original_args or args, epoch, m,
                           out_dir / 'best_model.pt')
            log.info(f"  ✓ Saved best (acc={best_acc:.4f})")

        # Save latest
        save_checkpoint(model, original_args or args, epoch, m,
                       out_dir / f'epoch_{epoch:02d}.pt')

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info(f"\n{'═'*64}")
    log.info(f"  TRAINING COMPLETE")
    log.info(f"{'═'*64}")
    log.info(f"  Best accuracy: {best_acc:.4f}  (epoch {best_epoch})")

    # Show full history
    log.info(f"\n  Epoch  Acc    E-rec  C-rec  N-rec  agree  TYPE-?")
    log.info(f"  {'─'*56}")
    for h in history:
        log.info(
            f"  {h['epoch']:>5}  {h['acc']:.4f}  "
            f"{h['e_rec']:.4f}  {h['c_rec']:.4f}  {h['n_rec']:.4f}  "
            f"{h['dyn_head_agree']:.4f}  {h['type_q_rate']:.4f}"
        )

    # Save history
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    log.info(f"\n  Output: {out_dir}")
    log.info(f"  Checkpoint: {out_dir / 'best_model.pt'}")

    # ── Hypothesis test ────────────────────────────────────────────────────────
    final = history[-1]
    log.info(f"\n{'═'*64}")
    log.info(f"  HYPOTHESIS TEST RESULTS")
    log.info(f"{'═'*64}")
    if final['type_q_rate'] < 0.02:
        log.info(f"  ✓ TYPE-? vanished ({final['type_q_rate']*100:.1f}%)")
        log.info(f"    → Head and dynamics are synchronized")
        log.info(f"    → Geometry-native classification confirmed")
    else:
        log.info(f"  ✗ TYPE-? persists ({final['type_q_rate']*100:.1f}%)")
        log.info(f"    → Head is still partially ignoring dynamics")

    if final['n_rec'] > 0.80:
        log.info(f"  ✓ Neutral recall {final['n_rec']*100:.1f}% > 80%")
        log.info(f"    → Neutral sponge fixed by decisive gradient-flow geometry")
    else:
        log.info(f"  → Neutral recall {final['n_rec']*100:.1f}% (target: >80%)")

    log.info(f"{'═'*64}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Joint Livnium retraining under grad-V dynamics'
    )

    # Data
    parser.add_argument('--snli-train',  required=True)
    parser.add_argument('--snli-dev',    required=True)
    parser.add_argument('--max-train',   type=int, default=None,
                        help='Cap training samples (default: all ~549K)')
    parser.add_argument('--max-dev',     type=int, default=None,
                        help='Cap dev samples (default: all ~9.8K)')
    parser.add_argument('--stratify',    action='store_true',
                        help='Balance training set: equal samples per class (use with --max-train)')

    # Warm start
    parser.add_argument('--checkpoint',    type=str, default=None,
                        help='Path to bert-joint checkpoint for warm start')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Cold start from scratch (ignore --checkpoint)')

    # Output
    parser.add_argument('--output-dir', required=True)

    # Physics
    parser.add_argument('--beta',  type=float, default=20.0,
                        help='Energy sharpness (β=1 flat, β=20 decisive)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Gradient step size')
    parser.add_argument('--steps', type=int, default=6,
                        help='Number of collapse steps L')
    parser.add_argument('--sphere-project', action='store_true',
                        help='Project h back to unit sphere after each step')

    # Loss
    parser.add_argument('--lambda-align', type=float, default=0.3,
                        help='Weight for anchor alignment loss (0 = CE only)')

    # Device
    parser.add_argument('--device',     type=str,   default=None,
                        help='Force device: cpu, mps, cuda (default: auto-detect)')

    # Training
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3,
                        help='Learning rate for head + anchors')
    parser.add_argument('--bert-lr',    type=float, default=2e-5,
                        help='Learning rate for BERT encoder')
    parser.add_argument('--seed',       type=int,   default=None,
                        help='Random seed for reproducibility (sets torch, numpy, random)')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
