#!/usr/bin/env python3
"""
eval_nli.py — held-out evaluation for the label-supervised NLI checkpoints.

Loads one or more `nli_epoch*.pt` checkpoints produced by
`train_collapse_embeddings.py --task nli`, runs them on a SNLI/MultiNLI .jsonl
(dev or test), and reports accuracy + a confusion matrix.

It reuses the trainer's OWN forward path (`_meanpool`, `_anchor_matrix`, the
`pair = u - v` warp, the anchor dot-product) so evaluation is identical to
training — only with no_grad and argmax instead of a backward pass.

Run (from inside collapse_retrain/):

    python3 eval_nli.py \
        --ckpt-dir model_nli_v1 \
        --data ../data/snli_1.0_test.jsonl \
        --device auto

Or a single checkpoint:

    python3 eval_nli.py --ckpt model_nli_v1/nli_epoch20.pt --data ../data/snli_1.0_test.jsonl

Reference bars on SNLI (from results/RESULTS.md): majority 34.3, full
bag-of-words 59.4, hypothesis-only artifact 61.5, GloVe-avg 60.7. A test number
only "means" something once it clears bag-of-words, and ideally the hyp-only bar.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse the exact training-time pieces so inference cannot drift from training.
from train_collapse_embeddings import (
    NLI_LABEL_TO_IDX,
    CollapseEmbeddingModel,
    NLIDataset,
    Vocab,
    _anchor_matrix,
    _meanpool,
    nli_collate,
    read_nli_jsonl,
)
from vector_collapse import VectorCollapseEngine

IDX_TO_LABEL = {v: k for k, v in NLI_LABEL_TO_IDX.items()}  # 0=entailment 1=neutral 2=contradiction
LABELS = [IDX_TO_LABEL[i] for i in range(3)]


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def vocab_from_checkpoint(ckpt: dict) -> Vocab:
    """Rebuild a Vocab from the {idx2word, pad_idx, unk_idx} blob in a checkpoint."""
    v = Vocab()
    v.idx2word = list(ckpt["vocab"]["idx2word"])
    v.word2idx = {w: i for i, w in enumerate(v.idx2word)}
    return v


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    vocab = vocab_from_checkpoint(ckpt)
    dim = ckpt["dim"]

    model = CollapseEmbeddingModel(vocab_size=len(vocab), dim=dim, pad_idx=vocab.pad_idx)
    with torch.no_grad():
        model.emb.weight.copy_(ckpt["embeddings"])
    model.to(device).eval()

    cfg = ckpt["collapse_config"]
    engine = VectorCollapseEngine(
        dim=dim,
        num_layers=cfg["num_layers"],
        strength_entail=cfg["strength_entail"],
        strength_contra=cfg["strength_contra"],
        strength_neutral=cfg["strength_neutral"],
    )
    engine.load_state_dict(ckpt["collapse_engine"])
    engine.to(device).eval()
    return model, engine, vocab


@torch.no_grad()
def evaluate(model, engine, vocab, loader, device) -> tuple[float, list[list[int]]]:
    correct, seen = 0, 0
    confusion = [[0, 0, 0] for _ in range(3)]  # rows = true, cols = pred
    pad_idx = vocab.pad_idx
    for prem, hyp, gold in loader:
        prem, hyp, gold = prem.to(device), hyp.to(device), gold.to(device)
        u = _meanpool(model, prem, pad_idx)
        v = _meanpool(model, hyp, pad_idx)
        pair = u - v
        pair, _ = engine(pair)
        pair_n = F.normalize(pair, dim=-1)
        anchors = _anchor_matrix(engine)  # [E, N, C]
        logits = pair_n @ anchors.t()  # temperature is irrelevant to argmax
        pred = logits.argmax(dim=-1)
        correct += int((pred == gold).sum().item())
        seen += gold.size(0)
        for t, p in zip(gold.tolist(), pred.tolist()):
            confusion[t][p] += 1
    return 100.0 * correct / max(seen, 1), confusion


def print_confusion(confusion: list[list[int]]) -> None:
    head = "true\\pred " + "".join(f"{name:>14}" for name in LABELS)
    print(head)
    for i, row in enumerate(confusion):
        print(f"{LABELS[i]:>9} " + "".join(f"{c:>14}" for c in row))


def epoch_key(path: str) -> int:
    m = re.search(r"epoch(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate NLI collapse checkpoints on held-out data.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=str, help="A single nli_epoch*.pt checkpoint.")
    g.add_argument("--ckpt-dir", type=str, help="Directory of nli_epoch*.pt checkpoints.")
    ap.add_argument("--data", type=str, required=True, help="SNLI/MultiNLI .jsonl (dev or test).")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-lines", type=int, default=0, help="0 = use all examples.")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"🚀 Device: {device}")

    examples = read_nli_jsonl(args.data, max_lines=args.max_lines)
    print(f"[eval] loaded {len(examples)} labeled examples from {args.data}")
    if not examples:
        raise SystemExit("No usable examples (check the path / gold labels).")

    if args.ckpt:
        ckpts = [args.ckpt]
    else:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "nli_epoch*.pt")), key=epoch_key)
    if not ckpts:
        raise SystemExit("No checkpoints found.")

    # Build the dataset once with the first checkpoint's vocab; vocab is identical
    # across epochs of one run, so this is safe and avoids re-tokenizing each time.
    _, _, vocab0 = load_checkpoint(ckpts[0], device)
    dataset = NLIDataset(examples, vocab0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: nli_collate(b, vocab0.pad_idx),
    )

    print(f"\n{'checkpoint':<28}{'test acc %':>12}")
    print("-" * 40)
    results = []
    for path in ckpts:
        model, engine, vocab = load_checkpoint(path, device)
        acc, confusion = evaluate(model, engine, vocab, loader, device)
        results.append((path, acc, confusion))
        print(f"{os.path.basename(path):<28}{acc:>12.2f}")

    best = max(results, key=lambda r: r[1])
    print("-" * 40)
    print(f"BEST: {os.path.basename(best[0])} @ {best[1]:.2f}%")
    print("\nConfusion matrix for the best checkpoint:")
    print_confusion(best[2])
    print(
        "\nReference (SNLI): majority 34.3 | bag-of-words 59.4 | " "hyp-only 61.5 | GloVe-avg 60.7"
    )


if __name__ == "__main__":
    main()
