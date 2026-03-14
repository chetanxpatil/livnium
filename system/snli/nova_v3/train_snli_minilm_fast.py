"""
train_snli_minilm_fast.py — Livnium collapse on frozen MiniLM embeddings.
=========================================================================
Pipeline:
    [cached MiniLM v_p, v_h]  ← loaded from disk, no BERT in the loop
         └─► Linear(384→256)  ← tiny projection layer
                 └─► h0 = proj(v_p) - proj(v_h)
                         └─► VectorCollapseEngine  (the "shake")
                                 └─► SNLIHead  (logits)

The "shake" idea:
  MiniLM is already trained on NLI — its embedding space has geometric
  structure near E/N/C. The collapse physics amplifies those clusters into
  hard attractor basins.  Only ~2M params train (projection + collapse + head).
  Converges in 2-3 epochs. Per-epoch time: ~3-5 min on CPU.

Step 1 — cache embeddings (ONE time, ~5-10 min):
    python system/snli/nova_v3/cache_minilm.py \\
        --snli-train data/snli/snli_1.0_train.jsonl \\
        --snli-dev   data/snli/snli_1.0_dev.jsonl \\
        --out-dir    minilm-cache

Step 2 — fast training (~5-10 min per epoch):
    python system/snli/nova_v3/train_snli_minilm_fast.py \\
        --minilm-cache minilm-cache \\
        --output-dir   runs/triple_crown_MiniLM_fast \\
        --dim 256 --num-layers 6 --epochs 5 \\
        --batch-size 256 --lr 1e-3 \\
        --lambda-traj 0.1 --lambda-fn 0.15 --lambda-rep 0.1 --margin-rep 0.3 \\
        --neutral-weight 1.5 --label-smoothing 0.08 \\
        --strength-neutral 0.08 --adaptive-metric
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# ── path setup ─────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from core import VectorCollapseEngine
from tasks.snli import SNLIHead


# ── Dataset (reads from cache) ─────────────────────────────────────────────────
class CachedMiniLMDataset(Dataset):
    """Serves pre-computed MiniLM v_p, v_h tensors + labels from a .pt cache file."""

    def __init__(self, cache_path: Path):
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.v_p    = data["v_p"].float()      # (N, hidden_dim)
        self.v_h    = data["v_h"].float()      # (N, hidden_dim)
        self.labels = data["labels"].long()    # (N,)
        self.hidden_dim = self.v_p.shape[1]
        print(f"  Loaded {len(self.labels):,} samples  dim={self.hidden_dim}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.v_p[idx], self.v_h[idx], self.labels[idx]


# ── Projection (MiniLM dim → out_dim) ─────────────────────────────────────────
class MiniLMProjection(nn.Module):
    """Linear(hidden_dim → out_dim) + LayerNorm. Shared across v_p and v_h."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ── Training epoch ─────────────────────────────────────────────────────────────
def train_epoch(
    projector, collapse, head, loader, optimizer, criterion, device,
    lambda_traj=0.0, lambda_fn=0.0, lambda_rep=0.0, margin_rep=0.3,
    scheduler=None,
):
    projector.train(); collapse.train(); head.train()
    total_loss = correct = total = 0

    for v_p_raw, v_h_raw, labels in tqdm(loader, desc="Training"):
        v_p_raw = v_p_raw.to(device)
        v_h_raw = v_h_raw.to(device)
        labels  = labels.to(device)

        # Project into collapse space
        v_p    = projector(v_p_raw)
        v_h    = projector(v_h_raw)
        h0     = v_p - v_h                         # difference vector

        # Collapse (the "shake")
        h_final, _ = collapse.collapse(h0)

        logits = head(h_final, v_p, v_h)
        loss   = criterion(logits, labels)

        # Trajectory loss — push δ toward correct anchor direction
        if lambda_traj > 0:
            anchors = torch.stack([
                F.normalize(collapse.anchor_entail,  dim=0),
                F.normalize(collapse.anchor_contra,  dim=0),
                F.normalize(collapse.anchor_neutral, dim=0),
            ], dim=0)
            correct_anchors = anchors[labels]
            delta  = h_final - h0
            L_traj = 1.0 - F.cosine_similarity(delta, correct_anchors.detach(), dim=-1)
            loss   = loss + lambda_traj * L_traj.mean()

        # False-neutral penalty
        if lambda_fn > 0:
            probs      = F.softmax(logits, dim=-1)
            p_neutral  = probs[:, 2]
            non_neutral = (labels != 2).float()
            loss = loss + lambda_fn * (non_neutral * p_neutral).mean()

        # Basin repulsion
        if lambda_rep > 0:
            anchors_rep = torch.stack([
                F.normalize(collapse.anchor_entail,  dim=0),
                F.normalize(collapse.anchor_contra,  dim=0),
                F.normalize(collapse.anchor_neutral, dim=0),
            ], dim=0)
            h_norm       = F.normalize(h_final, dim=-1)
            sims         = h_norm @ anchors_rep.T
            sim_correct  = sims.gather(1, labels.unsqueeze(1))
            wrong_mask   = torch.ones(labels.size(0), 3, device=device)
            wrong_mask.scatter_(1, labels.unsqueeze(1), 0.0)
            violations   = F.relu(margin_rep + sims - sim_correct) * wrong_mask
            loss = loss + lambda_rep * (violations.sum(1) / 2.0).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(projector.parameters()) +
            list(collapse.parameters()) +
            list(head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        pred     = logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total   += labels.size(0)

    return total_loss / len(loader), correct / total


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(projector, collapse, head, loader, device):
    projector.eval(); collapse.eval(); head.eval()
    correct = total = 0
    all_pred, all_lbl = [], []

    with torch.no_grad():
        for v_p_raw, v_h_raw, labels in tqdm(loader, desc="Evaluating"):
            v_p_raw = v_p_raw.to(device)
            v_h_raw = v_h_raw.to(device)
            labels  = labels.to(device)

            v_p    = projector(v_p_raw)
            v_h    = projector(v_h_raw)
            h0     = v_p - v_h
            h_final, _ = collapse.collapse(h0)
            logits = head(h_final, v_p, v_h)

            pred     = logits.argmax(-1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
            all_pred.extend(pred.cpu().tolist())
            all_lbl.extend(labels.cpu().tolist())

    confusion = np.zeros((3, 3), dtype=int)
    for p, l in zip(all_pred, all_lbl):
        confusion[l, p] += 1

    return correct / total, confusion


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Livnium collapse on cached MiniLM embeddings (fast)"
    )
    # Cache
    parser.add_argument("--minilm-cache", required=True,
                        help="Directory with train.pt and dev.pt from cache_minilm.py")
    # Model
    parser.add_argument("--dim",          type=int,   default=256)
    parser.add_argument("--num-layers",   type=int,   default=6)
    # Training
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
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
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    # Inject BARRIER
    import core.physics_laws as _physics
    _physics.BARRIER = args.barrier
    print(f"Livnium BARRIER = {args.barrier}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load cached embeddings ─────────────────────────────────────────────────
    cache_dir = Path(args.minilm_cache)
    print("\nLoading cached MiniLM embeddings...")
    train_set = CachedMiniLMDataset(cache_dir / "train.pt")
    hidden_dim = train_set.hidden_dim

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    dev_loader = None
    dev_cache = cache_dir / "dev.pt"
    if dev_cache.exists():
        dev_set    = CachedMiniLMDataset(dev_cache)
        dev_loader = DataLoader(
            dev_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

    # ── Models ────────────────────────────────────────────────────────────────
    projector = MiniLMProjection(hidden_dim, args.dim).to(device)
    collapse  = VectorCollapseEngine(
        dim=args.dim,
        num_layers=args.num_layers,
        strength_entail=args.strength_entail,
        strength_contra=args.strength_contra,
        strength_neutral=args.strength_neutral,
        strength_neutral_boost=args.strength_neutral_boost,
        adaptive_metric=args.adaptive_metric,
    ).to(device)
    head = SNLIHead(dim=args.dim).to(device)

    total_params = (
        sum(p.numel() for p in projector.parameters()) +
        sum(p.numel() for p in collapse.parameters()) +
        sum(p.numel() for p in head.parameters())
    )
    print(f"Trainable params: {total_params:,}  (projection + collapse + head only)")
    print(f"MiniLM hidden dim: {hidden_dim} → projected to {args.dim}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    all_params = (
        list(projector.parameters()) +
        list(collapse.parameters()) +
        list(head.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss ──────────────────────────────────────────────────────────────────
    class_weights = torch.tensor([1.0, 1.0, args.neutral_weight], device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Training  (triple_crown_MiniLM_fast — cached embeddings)")
    print("=" * 70)

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(
            projector, collapse, head, train_loader,
            optimizer, criterion, device,
            lambda_traj=args.lambda_traj,
            lambda_fn=args.lambda_fn,
            lambda_rep=args.lambda_rep,
            margin_rep=args.margin_rep,
            scheduler=scheduler,
        )
        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")

        if dev_loader:
            dev_acc, confusion = evaluate(projector, collapse, head, dev_loader, device)
            print(f"Dev Acc: {dev_acc:.4f}")
            e_rec = confusion[0, 0] / max(confusion[0].sum(), 1)
            c_rec = confusion[1, 1] / max(confusion[1].sum(), 1)
            n_rec = confusion[2, 2] / max(confusion[2].sum(), 1)
            print(f"Per-class → E:{e_rec*100:.1f}%  C:{c_rec*100:.1f}%  N:{n_rec*100:.1f}%")

            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save({
                    "projector":       projector.state_dict(),
                    "collapse_engine": collapse.state_dict(),
                    "head":            head.state_dict(),
                    "args":            args,
                    "encoder_type":    "minilm_cached",
                    "hidden_dim":      hidden_dim,
                    "best_acc":        best_acc,
                }, output_dir / "best_model.pt")
                print(f"✓ Saved best model ({best_acc:.4f})")

    print(f"\nBest Dev Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
