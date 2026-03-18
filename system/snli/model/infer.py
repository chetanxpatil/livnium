"""
Livnium Inference
=================
Run NLI inference on premise/hypothesis pairs using a trained checkpoint.

Usage:
    # Single pair
    python3 infer.py \
        --checkpoint ../../../pretrained/bert-joint/best_model.pt \
        --premise "A man is playing guitar on stage." \
        --hypothesis "Someone is performing music."

    # Interactive mode (keep typing pairs)
    python3 infer.py \
        --checkpoint ../../../pretrained/bert-joint/best_model.pt \
        --interactive

    # Batch from file (one JSON per line: {"premise": ..., "hypothesis": ...})
    python3 infer.py \
        --checkpoint ../../../pretrained/bert-joint/best_model.pt \
        --file pairs.jsonl
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from core import VectorCollapseEngine
from tasks.snli import BERTSNLIEncoder, CrossEncoderBERTSNLIEncoder, SNLIHead

LABEL_MAP  = {0: "ENTAILMENT", 1: "CONTRADICTION", 2: "NEUTRAL"}
LABEL_EMOJI = {0: "✅", 1: "❌", 2: "🤔"}


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", None)

    dim          = getattr(saved_args, "dim", 768) if saved_args else 768
    num_layers   = getattr(saved_args, "num_layers", 6) if saved_args else 6
    bert_model   = getattr(saved_args, "bert_model", "bert-base-uncased") if saved_args else "bert-base-uncased"
    encoder_type = getattr(saved_args, "encoder_type", "bert") if saved_args else "bert"

    if encoder_type == "crossbert":
        encoder = CrossEncoderBERTSNLIEncoder(model_name=bert_model, freeze=True).to(device)
        print(f"  Encoder: CrossEncoderBERT (joint premise+hypothesis)")
    else:
        encoder = BERTSNLIEncoder(model_name=bert_model, freeze=True).to(device)
        print(f"  Encoder: BERTSNLIEncoder (bi-encoder)")
    collapse_engine = VectorCollapseEngine(dim=dim, num_layers=num_layers).to(device)
    head = SNLIHead(dim=dim).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    collapse_engine.load_state_dict(ckpt["collapse_engine"])
    head.load_state_dict(ckpt["head"])

    encoder.eval()
    collapse_engine.eval()
    head.eval()

    print(f"  Model loaded  (dim={dim}, layers={num_layers}, bert={bert_model})")
    return encoder, collapse_engine, head


def predict(premises, hypotheses, encoder, collapse_engine, head, device):
    """
    Run inference on a batch of premise/hypothesis pairs.

    Returns:
        labels      : List[str]   e.g. ["ENTAILMENT", ...]
        confidences : List[float] e.g. [0.925, ...]
        probs_all   : List[dict]  e.g. [{"ENTAILMENT": 0.92, "CONTRADICTION": 0.03, "NEUTRAL": 0.05}, ...]
    """
    with torch.no_grad():
        h0, v_p, v_h = encoder.build_initial_state(
            premises, hypotheses, add_noise=False, device=device
        )
        h_final, _ = collapse_engine.collapse(h0)
        logits = head(h_final, v_p, v_h)
        probs  = F.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)

    labels      = [LABEL_MAP[p.item()] for p in preds]
    confidences = [probs[i, preds[i]].item() for i in range(len(preds))]
    probs_all   = [
        {LABEL_MAP[j]: probs[i, j].item() for j in range(3)}
        for i in range(len(preds))
    ]
    return labels, confidences, probs_all


def print_result(premise, hypothesis, label, confidence, probs_all):
    emoji = LABEL_EMOJI[list(LABEL_MAP.values()).index(label)]
    print(f"\n  Premise    : {premise}")
    print(f"  Hypothesis : {hypothesis}")
    print(f"  Prediction : {emoji}  {label}  ({confidence*100:.1f}% confidence)")
    print(f"  All probs  : "
          f"E={probs_all['ENTAILMENT']*100:.1f}%  "
          f"C={probs_all['CONTRADICTION']*100:.1f}%  "
          f"N={probs_all['NEUTRAL']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Livnium NLI Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--premise", type=str, default=None)
    parser.add_argument("--hypothesis", type=str, default=None)
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode — keep entering pairs until Ctrl+C")
    parser.add_argument("--file", type=str, default=None,
                        help="JSONL file with {premise, hypothesis} per line")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cpu / mps / cuda). Auto-detects if not set.")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nLivnium Inference  |  device={device}")
    print(f"Loading {args.checkpoint} ...")
    encoder, collapse_engine, head = load_model(args.checkpoint, device)

    # ── Single pair ───────────────────────────────────────────────────────────
    if args.premise and args.hypothesis:
        labels, confs, probs_all = predict(
            [args.premise], [args.hypothesis],
            encoder, collapse_engine, head, device
        )
        print_result(args.premise, args.hypothesis, labels[0], confs[0], probs_all[0])
        print()

    # ── Batch file ────────────────────────────────────────────────────────────
    elif args.file:
        pairs = []
        with open(args.file) as f:
            for line in f:
                d = json.loads(line.strip())
                pairs.append((d["premise"], d["hypothesis"]))

        premises    = [p for p, _ in pairs]
        hypotheses  = [h for _, h in pairs]
        labels, confs, probs_all = predict(
            premises, hypotheses, encoder, collapse_engine, head, device
        )
        print(f"\nResults for {len(pairs)} pairs:")
        for i, (p, h) in enumerate(pairs):
            print_result(p, h, labels[i], confs[i], probs_all[i])
        print()

    # ── Interactive ───────────────────────────────────────────────────────────
    elif args.interactive:
        print("\nInteractive mode — enter premise/hypothesis pairs (Ctrl+C to quit)\n")
        while True:
            try:
                premise    = input("  Premise    > ").strip()
                hypothesis = input("  Hypothesis > ").strip()
                if not premise or not hypothesis:
                    continue
                labels, confs, probs_all = predict(
                    [premise], [hypothesis],
                    encoder, collapse_engine, head, device
                )
                print_result(premise, hypothesis, labels[0], confs[0], probs_all[0])
                print()
            except KeyboardInterrupt:
                print("\nBye!")
                break

    else:
        print("\nProvide --premise + --hypothesis, --interactive, or --file")
        parser.print_help()


if __name__ == "__main__":
    main()
