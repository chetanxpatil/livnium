#!/usr/bin/env python3
"""
predict.py — classify a single (premise, hypothesis) pair with a trained
collapse-NLI checkpoint.

Reuses the exact eval/training forward (`_meanpool`, `_anchor_matrix`, the
`pair = u - v` warp) so a prediction here matches what the model was trained to do.

Example (from inside collapse_retrain/):

    python3 predict.py \
        --ckpt model_nli_v1/nli_epoch20.pt \
        --premise "A man is playing a guitar on stage." \
        --hypothesis "A person is performing music."

Prints the predicted label (entailment / neutral / contradiction) and the
softmax confidence over all three.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from eval_nli import LABELS, load_checkpoint, resolve_device
from train_collapse_embeddings import _anchor_matrix, _meanpool


@torch.no_grad()
def predict(model, engine, vocab, premise: str, hypothesis: str, device) -> tuple[str, dict]:
    p = torch.tensor([vocab.encode_line(premise.lower()) or [vocab.unk_idx]], device=device)
    h = torch.tensor([vocab.encode_line(hypothesis.lower()) or [vocab.unk_idx]], device=device)
    u = _meanpool(model, p, vocab.pad_idx)
    v = _meanpool(model, h, vocab.pad_idx)
    pair = u - v
    pair, _ = engine(pair)
    pair_n = F.normalize(pair, dim=-1)
    anchors = _anchor_matrix(engine)  # [E, N, C]
    probs = F.softmax(pair_n @ anchors.t(), dim=-1)[0]
    idx = int(probs.argmax())
    return LABELS[idx], {LABELS[i]: float(probs[i]) for i in range(3)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify a premise/hypothesis pair (NLI).")
    ap.add_argument("--ckpt", type=str, default="model_nli_v1/nli_epoch20.pt")
    ap.add_argument("--premise", type=str, required=True)
    ap.add_argument("--hypothesis", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, engine, vocab = load_checkpoint(args.ckpt, device)
    label, probs = predict(model, engine, vocab, args.premise, args.hypothesis, device)

    print(f"\nPremise   : {args.premise}")
    print(f"Hypothesis: {args.hypothesis}")
    print(f"\nPrediction: {label.upper()}")
    for name in LABELS:
        print(f"  {name:<14} {probs[name] * 100:6.2f}%")


if __name__ == "__main__":
    main()
