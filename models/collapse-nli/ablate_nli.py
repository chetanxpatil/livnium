#!/usr/bin/env python3
"""
ablate_nli.py — Ablation study for the supervised NLI collapse model.

Loads the best model (default: model_nli_v1/nli_epoch20.pt), freezes the trained
embeddings, and compares the full collapse model against:
1. Same embeddings + a trained linear head (no collapse engine)
2. Same embeddings + a trained 2-layer MLP head (no collapse engine)
3. Collapse engine with randomized anchors
4. Frozen random embeddings + trained collapse engine

This answers: does the collapse engine add value, or do the supervised
embeddings alone explain the performance?
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from eval_nli import load_checkpoint, resolve_device
from torch.utils.data import DataLoader, TensorDataset
from train_collapse_embeddings import (
    NLIDataset,
    _anchor_matrix,
    _meanpool,
    nli_collate,
    read_nli_jsonl,
)


class LinearProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@torch.no_grad()
def evaluate_model(model, engine, vocab, loader, device) -> float:
    correct, seen = 0, 0
    pad_idx = vocab.pad_idx
    for prem, hyp, gold in loader:
        prem, hyp, gold = prem.to(device), hyp.to(device), gold.to(device)
        u = _meanpool(model, prem, pad_idx)
        v = _meanpool(model, hyp, pad_idx)
        pair = u - v
        if engine is not None:
            pair, _ = engine(pair)
        pair_n = F.normalize(pair, dim=-1)
        if engine is not None:
            anchors = _anchor_matrix(engine)
            logits = pair_n @ anchors.t()
        else:
            # If no engine, just use raw pair vector as representation (unscaled)
            logits = pair_n
        pred = logits.argmax(dim=-1)
        correct += int((pred == gold).sum().item())
        seen += gold.size(0)
    return 100.0 * correct / max(seen, 1)


@torch.no_grad()
def extract_features(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute the u - v difference vectors to train probes quickly."""
    features = []
    labels = []
    pad_idx = model.pad_idx
    for prem, hyp, gold in loader:
        prem, hyp, gold = prem.to(device), hyp.to(device), gold.to(device)
        u = _meanpool(model, prem, pad_idx)
        v = _meanpool(model, hyp, pad_idx)
        pair = u - v
        features.append(pair.cpu())
        labels.append(gold.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def train_probe(
    probe,
    X_train,
    y_train,
    X_dev,
    y_dev,
    X_test,
    y_test,
    device,
    epochs=10,
    batch_size=512,
    lr=1e-3,
) -> tuple[float, float]:
    """Train a classification head on top of pre-computed features."""
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_dev_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(epochs):
        probe.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = probe(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()

        # Evaluate on dev
        probe.eval()
        with torch.no_grad():
            dev_logits = probe(X_dev.to(device))
            dev_acc = (dev_logits.argmax(dim=-1) == y_dev.to(device)).float().mean().item()

            test_logits = probe(X_test.to(device))
            test_acc = (test_logits.argmax(dim=-1) == y_test.to(device)).float().mean().item()

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                best_test_acc = test_acc

    return best_dev_acc * 100.0, best_test_acc * 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Run NLI collapse ablation studies.")
    ap.add_argument(
        "--ckpt", type=str, default="model_nli_v1/nli_epoch20.pt", help="Checkpoint to ablate."
    )
    ap.add_argument("--train-data", type=str, required=True, help="SNLI train .jsonl")
    ap.add_argument("--dev-data", type=str, required=True, help="SNLI dev .jsonl")
    ap.add_argument("--test-data", type=str, required=True, help="SNLI test .jsonl")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--probe-epochs", type=int, default=15, help="Epochs to train linear/MLP probes."
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"🚀 Device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_examples = read_nli_jsonl(args.train_data)
    dev_examples = read_nli_jsonl(args.dev_data)
    test_examples = read_nli_jsonl(args.test_data)
    print(
        f"Loaded: Train={len(train_examples)}, Dev={len(dev_examples)}, Test={len(test_examples)}"
    )

    # 2. Load Checkpoint
    print(f"Loading checkpoint {args.ckpt}...")
    model, engine, vocab = load_checkpoint(args.ckpt, device)
    dim = model.dim

    # Datasets and Loaders
    train_dataset = NLIDataset(train_examples, vocab)
    dev_dataset = NLIDataset(dev_examples, vocab)
    test_dataset = NLIDataset(test_examples, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: nli_collate(b, vocab.pad_idx),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: nli_collate(b, vocab.pad_idx),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: nli_collate(b, vocab.pad_idx),
    )

    print("\n--- Running Ablations ---")
    results = {}

    # Ablation A: Full model
    print("Evaluating Full Collapse model...")
    full_dev_acc = evaluate_model(model, engine, vocab, dev_loader, device)
    full_test_acc = evaluate_model(model, engine, vocab, test_loader, device)
    results["full_collapse"] = (full_dev_acc, full_test_acc)
    print(f"  -> Dev: {full_dev_acc:.2f}%, Test: {full_test_acc:.2f}%")

    # Extract features for Probes
    print("\nPre-computing embedding features for probe training...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_dev, y_dev = extract_features(model, dev_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)

    # Ablation B: Embeddings + Linear Head
    print("Training same embeddings + Linear Head...")
    linear_probe = LinearProbe(dim)
    lin_dev, lin_test = train_probe(
        linear_probe,
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        y_test,
        device,
        epochs=args.probe_epochs,
        batch_size=args.batch_size,
    )
    results["embeddings_linear"] = (lin_dev, lin_test)
    print(f"  -> Dev: {lin_dev:.2f}%, Test: {lin_test:.2f}%")

    # Ablation C: Embeddings + MLP Head
    print("Training same embeddings + MLP Head...")
    mlp_probe = MLPProbe(dim)
    mlp_dev, mlp_test = train_probe(
        mlp_probe,
        X_train,
        y_train,
        X_dev,
        y_dev,
        X_test,
        y_test,
        device,
        epochs=args.probe_epochs,
        batch_size=args.batch_size,
    )
    results["embeddings_mlp"] = (mlp_dev, mlp_test)
    print(f"  -> Dev: {mlp_dev:.2f}%, Test: {mlp_test:.2f}%")

    # Ablation D: Random-Anchor collapse
    print("Evaluating Collapse with randomized anchors...")
    orig_entail = engine.anchor_entail.data.clone()
    orig_contra = engine.anchor_contra.data.clone()
    orig_neutral = engine.anchor_neutral.data.clone()

    # Initialize random unit anchors
    engine.anchor_entail.data.copy_(F.normalize(torch.randn_like(orig_entail), dim=0))
    engine.anchor_contra.data.copy_(F.normalize(torch.randn_like(orig_contra), dim=0))
    engine.anchor_neutral.data.copy_(F.normalize(torch.randn_like(orig_neutral), dim=0))

    rand_anc_dev = evaluate_model(model, engine, vocab, dev_loader, device)
    rand_anc_test = evaluate_model(model, engine, vocab, test_loader, device)
    results["random_anchor_collapse"] = (rand_anc_dev, rand_anc_test)
    print(f"  -> Dev: {rand_anc_dev:.2f}%, Test: {rand_anc_test:.2f}%")

    # Restore anchors
    engine.anchor_entail.data.copy_(orig_entail)
    engine.anchor_contra.data.copy_(orig_contra)
    engine.anchor_neutral.data.copy_(orig_neutral)

    # Ablation E: Frozen random embeddings + Collapse engine
    print("Evaluating Collapse engine with frozen random embeddings...")
    orig_emb = model.emb.weight.data.clone()
    nn.init.normal_(model.emb.weight, mean=0.0, std=0.05)

    rand_emb_dev = evaluate_model(model, engine, vocab, dev_loader, device)
    rand_emb_test = evaluate_model(model, engine, vocab, test_loader, device)
    results["random_embeddings_collapse"] = (rand_emb_dev, rand_emb_test)
    print(f"  -> Dev: {rand_emb_dev:.2f}%, Test: {rand_emb_test:.2f}%")

    # Restore embeddings
    model.emb.weight.data.copy_(orig_emb)

    # Print final summary table
    print("\n" + "=" * 55)
    print(f"{'model':<28}{'dev_acc':>12}{'test_acc':>12}")
    print("-" * 55)
    for model_name, (dev_acc, test_acc) in results.items():
        print(f"{model_name:<28}{dev_acc:>12.2f}{test_acc:>12.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
