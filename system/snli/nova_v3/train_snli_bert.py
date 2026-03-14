"""
Livnium SNLI Training — BERT Encoder (triple_crown_slow_BERT)
=============================================================
Pipeline:
  bert-base-uncased
      └─► BERTSNLIEncoder  (mean-pool premise/hypothesis → project 768→256)
              └─► h0 = v_p - v_h  (difference vector)
                      └─► VectorCollapseEngine  (settles into E/N/C basin)
                              └─► SNLIHead  (logits)

NEW: encoder-type=bert replaces quantum. Everything else identical to
     train_snli_vector.py — do NOT edit that file.

Run:
  python train_snli_bert.py \\
      --snli-train data/snli/snli_1.0_train.jsonl \\
      --snli-dev   data/snli/snli_1.0_dev.jsonl \\
      --output-dir runs/triple_crown_slow_BERT \\
      --dim 256 --num-layers 6 --epochs 10 \\
      --batch-size 16 --lr 2e-5 \\
      --lambda-traj 0.1 --lambda-fn 0.15 --lambda-rep 0.1 --margin-rep 0.3 \\
      --neutral-weight 1.5 --label-smoothing 0.08 \\
      --adaptive-metric
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from core import VectorCollapseEngine
from tasks.snli import SNLIHead
from text.bert_encoder import BERTSNLIEncoder


# ── Dataset ───────────────────────────────────────────────────────────────────
class SNLIDataset(Dataset):
    LABEL_MAP = {"entailment": 0, "contradiction": 1, "neutral": 2}

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "premise":    s["premise"],
            "hypothesis": s["hypothesis"],
            "label":      torch.tensor(self.LABEL_MAP[s["gold_label"]], dtype=torch.long),
            "gold_label": s["gold_label"],
        }


def load_snli_data(path: Path, max_samples: Optional[int] = None) -> List[Dict]:
    samples, seen = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            lbl = d.get("gold_label", "").strip()
            if lbl not in ("entailment", "contradiction", "neutral"):
                continue
            prem = d.get("sentence1", "").strip()
            hyp  = d.get("sentence2", "").strip()
            if not prem or not hyp:
                continue
            key = (prem, hyp)
            if key in seen and seen[key] != lbl:
                continue
            seen[key] = lbl
            samples.append({"premise": prem, "hypothesis": hyp, "gold_label": lbl})
            if max_samples and len(samples) >= max_samples:
                break
    return samples


# ── Train epoch ───────────────────────────────────────────────────────────────
def train_epoch(collapse, encoder, head, loader, optimizer, criterion, device,
                lambda_traj=0.0, lambda_fn=0.0, lambda_rep=0.0, margin_rep=0.3,
                scheduler=None):
    collapse.train(); encoder.train(); head.train()
    total_loss = correct = total = 0

    for batch in tqdm(loader, desc="Training"):
        labels = batch["label"].to(device)
        h0, v_p, v_h = encoder.build_initial_state(
            list(batch["premise"]), list(batch["hypothesis"]), device=device
        )
        h_final, _ = collapse.collapse(h0)
        logits = head(h_final, v_p, v_h)
        loss = criterion(logits, labels)

        # Trajectory loss
        if lambda_traj > 0:
            anchors = torch.stack([
                F.normalize(collapse.anchor_entail, dim=0),
                F.normalize(collapse.anchor_contra, dim=0),
                F.normalize(collapse.anchor_neutral, dim=0),
            ], dim=0)
            correct_anchors = anchors[labels]
            delta = h_final - h0
            L_traj = 1.0 - F.cosine_similarity(delta, correct_anchors.detach(), dim=-1)
            loss = loss + lambda_traj * L_traj.mean()

        # False-neutral penalty
        if lambda_fn > 0:
            probs = F.softmax(logits, dim=-1)
            p_neutral = probs[:, 2]
            non_neutral = (labels != 2).float()
            loss = loss + lambda_fn * (non_neutral * p_neutral).mean()

        # Basin repulsion
        if lambda_rep > 0:
            anchors_rep = torch.stack([
                F.normalize(collapse.anchor_entail, dim=0),
                F.normalize(collapse.anchor_contra, dim=0),
                F.normalize(collapse.anchor_neutral, dim=0),
            ], dim=0)
            h_norm = F.normalize(h_final, dim=-1)
            sims = h_norm @ anchors_rep.T
            sim_correct = sims.gather(1, labels.unsqueeze(1))
            wrong_mask = torch.ones(labels.size(0), 3, device=device)
            wrong_mask.scatter_(1, labels.unsqueeze(1), 0.0)
            violations = F.relu(margin_rep + sims - sim_correct) * wrong_mask
            loss = loss + lambda_rep * (violations.sum(1) / 2.0).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(collapse.parameters()) + list(encoder.parameters()) + list(head.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        pred = logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(collapse, encoder, head, loader, device):
    collapse.eval(); encoder.eval(); head.eval()
    correct = total = 0
    all_pred, all_lbl = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            labels = batch["label"].to(device)
            h0, v_p, v_h = encoder.build_initial_state(
                list(batch["premise"]), list(batch["hypothesis"]), device=device
            )
            h_final, _ = collapse.collapse(h0)
            logits = head(h_final, v_p, v_h)
            pred = logits.argmax(-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            all_pred.extend(pred.cpu().tolist())
            all_lbl.extend(labels.cpu().tolist())

    confusion = np.zeros((3, 3), dtype=int)
    for p, l in zip(all_pred, all_lbl):
        confusion[l, p] += 1

    return correct / total, confusion


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Livnium + BERT encoder training")

    # Data
    parser.add_argument("--snli-train",   required=True)
    parser.add_argument("--snli-dev",     default=None)
    parser.add_argument("--max-samples",  type=int, default=None)

    # Model
    parser.add_argument("--dim",          type=int,   default=256)
    parser.add_argument("--num-layers",   type=int,   default=6)
    parser.add_argument("--bert-model",   default="bert-base-uncased")
    parser.add_argument("--bert-pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--freeze-bert",  action="store_true",
                        help="Freeze BERT weights — only train collapse+head")

    # Training
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-len",      type=int,   default=128)

    # Physics
    parser.add_argument("--strength-entail",        type=float, default=0.1)
    parser.add_argument("--strength-contra",        type=float, default=0.1)
    parser.add_argument("--strength-neutral",       type=float, default=0.08)
    parser.add_argument("--strength-neutral-boost", type=float, default=0.05)
    parser.add_argument("--adaptive-metric",        action="store_true")
    parser.add_argument("--barrier",                type=float, default=0.38)

    # Losses
    parser.add_argument("--lambda-traj",     type=float, default=0.0)
    parser.add_argument("--lambda-fn",       type=float, default=0.0)
    parser.add_argument("--lambda-rep",      type=float, default=0.0)
    parser.add_argument("--margin-rep",      type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--neutral-weight",  type=float, default=1.0)

    # Output
    parser.add_argument("--output-dir",  required=True)

    args = parser.parse_args()

    # Inject barrier
    import core.physics_laws as _physics
    _physics.BARRIER = args.barrier
    print(f"Livnium BARRIER = {args.barrier}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading SNLI data...")
    train_samples = load_snli_data(Path(args.snli_train), args.max_samples)
    print(f"  Train: {len(train_samples):,}")
    train_loader = DataLoader(
        SNLIDataset(train_samples),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda x: {k: [s[k] for s in x] if k not in ("label",)
                               else torch.stack([s[k] for s in x]) for k in x[0]},
    )

    dev_loader = None
    if args.snli_dev:
        dev_samples = load_snli_data(Path(args.snli_dev))
        print(f"  Dev:   {len(dev_samples):,}")
        dev_loader = DataLoader(
            SNLIDataset(dev_samples),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda x: {k: [s[k] for s in x] if k not in ("label",)
                                   else torch.stack([s[k] for s in x]) for k in x[0]},
        )

    # Models
    print(f"Loading {args.bert_model}...")
    encoder = BERTSNLIEncoder(
        out_dim=args.dim,
        model_name=args.bert_model,
        freeze_bert=args.freeze_bert,
        pooling=args.bert_pooling,
    ).to(device)

    collapse = VectorCollapseEngine(
        dim=args.dim,
        num_layers=args.num_layers,
        strength_entail=args.strength_entail,
        strength_contra=args.strength_contra,
        strength_neutral=args.strength_neutral,
        strength_neutral_boost=args.strength_neutral_boost,
        adaptive_metric=args.adaptive_metric,
    ).to(device)

    head = SNLIHead(dim=args.dim).to(device)

    # Optimizer — lower LR for BERT, higher for collapse+head
    bert_params  = list(encoder.bert.parameters())
    other_params = list(encoder.proj.parameters()) + list(collapse.parameters()) + list(head.parameters())
    optimizer = optim.AdamW([
        {"params": bert_params,  "lr": args.lr},
        {"params": other_params, "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)

    # LR schedule: linear warmup → linear decay
    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    class_weights = torch.tensor([1.0, 1.0, args.neutral_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Training  (triple_crown_slow_BERT)")
    print("=" * 70)

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(
            collapse, encoder, head, train_loader,
            optimizer, criterion, device,
            lambda_traj=args.lambda_traj,
            lambda_fn=args.lambda_fn,
            lambda_rep=args.lambda_rep,
            margin_rep=args.margin_rep,
            scheduler=scheduler,
        )
        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")

        if dev_loader:
            dev_acc, confusion = evaluate(collapse, encoder, head, dev_loader, device)
            print(f"Dev Acc: {dev_acc:.4f}")
            e_rec = confusion[0,0] / max(confusion[0].sum(), 1)
            c_rec = confusion[1,1] / max(confusion[1].sum(), 1)
            n_rec = confusion[2,2] / max(confusion[2].sum(), 1)
            print(f"Per-class → E:{e_rec*100:.1f}%  C:{c_rec*100:.1f}%  N:{n_rec*100:.1f}%")

            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save({
                    "collapse_engine": collapse.state_dict(),
                    "encoder":         encoder.state_dict(),
                    "head":            head.state_dict(),
                    "args":            args,
                    "vocab":           None,
                    "encoder_type":    "bert",
                    "best_acc":        best_acc,
                }, output_dir / "best_model.pt")
                print(f"✓ Saved best model ({best_acc:.4f})")

    print(f"\nBest Dev Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
