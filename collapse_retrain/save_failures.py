#!/usr/bin/env python3
"""
save_failures.py — Find and log all incorrectly classified NLI test examples.

Loads the optimal checkpoint, runs it over the SNLI test set, gathers all
failed predictions, and writes them to collapse_retrain/failed_examples.json.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from eval_nli import LABELS, load_checkpoint, resolve_device
from train_collapse_embeddings import (
    NLIDataset,
    _anchor_matrix,
    _meanpool,
    nli_collate,
    read_nli_jsonl,
)


def main() -> None:
    device = resolve_device("auto")
    ckpt_path = "model_nli_v1/nli_epoch20.pt"
    data_path = "/Users/chetanpatil/Desktop/test/data/snli_1.0_test.jsonl"

    print(f"🚀 Device: {device}")
    print(f"Loading checkpoint {ckpt_path}...")
    model, engine, vocab = load_checkpoint(ckpt_path, device)

    print(f"Loading test dataset from {data_path}...")
    examples = read_nli_jsonl(data_path)
    print(f"Loaded {len(examples)} test examples.")

    dataset = NLIDataset(examples, vocab)
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        collate_fn=lambda b: nli_collate(b, vocab.pad_idx),
    )

    failures = []
    pad_idx = vocab.pad_idx
    example_idx = 0

    print("Running evaluation and extracting failure cases...")
    with torch.no_grad():
        for prem, hyp, gold in loader:
            prem, hyp, gold = prem.to(device), hyp.to(device), gold.to(device)
            u = _meanpool(model, prem, pad_idx)
            v = _meanpool(model, hyp, pad_idx)
            pair = u - v
            pair, _ = engine(pair)
            pair_n = F.normalize(pair, dim=-1)
            anchors = _anchor_matrix(engine)
            logits = pair_n @ anchors.t()
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            for pred, label, prob in zip(preds.tolist(), gold.tolist(), probs.tolist()):
                raw_prem, raw_hyp, _ = examples[example_idx]
                if pred != label:
                    failures.append(
                        {
                            "index": example_idx,
                            "premise": raw_prem,
                            "hypothesis": raw_hyp,
                            "gold_label": LABELS[label],
                            "predicted_label": LABELS[pred],
                            "confidence": {
                                LABELS[0]: float(prob[0]),
                                LABELS[1]: float(prob[1]),
                                LABELS[2]: float(prob[2]),
                            },
                        }
                    )
                example_idx += 1

    output_path = "failed_examples.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    print(f"🎉 Successfully logged {len(failures)} failures out of {len(examples)} total examples.")
    print(f"Failure file saved to: {output_path}")


if __name__ == "__main__":
    main()
