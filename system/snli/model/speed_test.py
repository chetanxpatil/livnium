"""
Livnium Speed Test
==================
Measures inference latency and throughput of the full pipeline:
    BERT encoder → h0 = v_h - v_p → CollapseEngine → SNLIHead → prediction

Reports:
  - Warmup runs (ignored)
  - Single-example latency (ms)
  - Batch throughput (examples/sec)
  - BERT-only time vs full Livnium pipeline
  - CPU vs MPS comparison (if both available)

Usage:
    python3 speed_test.py \
        --checkpoint ../../../pretrained/bert-joint/best_model.pt \
        --bert-model bert-base-uncased \
        --batch-sizes 1 8 32 64 \
        --n-runs 50
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

_here = Path(__file__).resolve().parent
_repo = _here.parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_repo))

from core import VectorCollapseEngine
from tasks.snli import BERTSNLIEncoder, SNLIHead

# ── Sample sentences for benchmarking ─────────────────────────────────────────
PREMISES = [
    "A man is playing a guitar on stage.",
    "Two children are running in a park.",
    "A woman is reading a book by the window.",
    "The dog is sleeping on the couch.",
    "A group of people are eating at a restaurant.",
    "The cat is sitting on the roof.",
    "A cyclist is riding down a mountain trail.",
    "Several students are studying in the library.",
]
HYPOTHESES = [
    "Someone is performing music.",
    "The kids are outside.",
    "A person is reading.",
    "The dog is resting.",
    "People are having a meal.",
    "An animal is on the roof.",
    "A person is cycling outdoors.",
    "People are in the library.",
]

LABEL_MAP = {0: "ENTAILMENT", 1: "CONTRADICTION", 2: "NEUTRAL"}


def build_batch(premises, hypotheses, size):
    """Repeat sentences to fill requested batch size."""
    n = len(premises)
    p = [premises[i % n] for i in range(size)]
    h = [hypotheses[i % n] for i in range(size)]
    return p, h


def time_fn(fn, n_runs, warmup=5):
    """Run fn n_runs times, skip first warmup, return (mean_ms, std_ms, all_ms)."""
    times = []
    for i in range(n_runs + warmup):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), times


def run_speed_test(checkpoint_path, bert_model, batch_sizes, n_runs, devices):
    print("\n" + "=" * 65)
    print(" Livnium Speed Test")
    print("=" * 65)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", None)

    dim = 768  # BERT base default
    num_layers = 6
    if saved_args is not None:
        dim = getattr(saved_args, "dim", 768)
        num_layers = getattr(saved_args, "num_layers", 6)

    for device_name in devices:
        device = torch.device(device_name)
        print(f"\n{'─'*65}")
        print(f" Device: {str(device).upper()}")
        print(f"{'─'*65}")

        # Build models
        encoder = BERTSNLIEncoder(model_name=bert_model, freeze=True).to(device)
        collapse_engine = VectorCollapseEngine(dim=dim, num_layers=num_layers).to(device)
        head = SNLIHead(dim=dim).to(device)

        # Load weights
        encoder.load_state_dict(ckpt["encoder"])
        collapse_engine.load_state_dict(ckpt["collapse_engine"])
        head.load_state_dict(ckpt["head"])

        encoder.eval()
        collapse_engine.eval()
        head.eval()

        print(f"\n{'Batch':>6}  {'Latency(ms)':>12}  {'Std(ms)':>8}  {'Examples/s':>12}  {'ms/example':>11}")
        print(f"{'─'*6}  {'─'*12}  {'─'*8}  {'─'*12}  {'─'*11}")

        for batch_size in batch_sizes:
            premises, hypotheses = build_batch(PREMISES, HYPOTHESES, batch_size)

            def full_pipeline():
                with torch.no_grad():
                    h0, v_p, v_h = encoder.build_initial_state(
                        premises, hypotheses, add_noise=False, device=device
                    )
                    h_final, _ = collapse_engine.collapse(h0)
                    logits = head(h_final, v_p, v_h)
                    return logits.argmax(dim=-1)

            mean_ms, std_ms, _ = time_fn(full_pipeline, n_runs=n_runs, warmup=5)
            per_example_ms = mean_ms / batch_size
            examples_per_sec = 1000.0 / per_example_ms

            print(f"{batch_size:>6}  {mean_ms:>12.1f}  {std_ms:>8.1f}  {examples_per_sec:>12.1f}  {per_example_ms:>11.2f}")

        # ── BERT-only vs full pipeline breakdown (batch=32) ────────────────
        print(f"\n── Pipeline breakdown at batch=32 (device={device_name}) ──")
        B = min(32, max(batch_sizes))
        premises_b, hypotheses_b = build_batch(PREMISES, HYPOTHESES, B)

        def bert_only():
            with torch.no_grad():
                encoder.build_initial_state(
                    premises_b, hypotheses_b, add_noise=False, device=device
                )

        def collapse_only(h0_cache):
            with torch.no_grad():
                h_final, _ = collapse_engine.collapse(h0_cache)
                return h_final

        # Pre-compute h0 for collapse timing
        with torch.no_grad():
            h0_cache, v_p_cache, v_h_cache = encoder.build_initial_state(
                premises_b, hypotheses_b, add_noise=False, device=device
            )

        bert_ms, _, _ = time_fn(bert_only, n_runs=n_runs, warmup=5)
        col_ms, _, _ = time_fn(lambda: collapse_only(h0_cache), n_runs=n_runs, warmup=5)

        total_ms = bert_ms + col_ms
        print(f"  BERT encoder :  {bert_ms:6.1f} ms  ({bert_ms/total_ms*100:.0f}%)")
        print(f"  Collapse+Head:  {col_ms:6.1f} ms  ({col_ms/total_ms*100:.0f}%)")
        print(f"  Total pipeline: {total_ms:6.1f} ms  →  {B*1000/total_ms:.0f} examples/sec")

        # ── Sample predictions ─────────────────────────────────────────────
        print(f"\n── Sample predictions ──")
        sample_p = PREMISES[:4]
        sample_h = HYPOTHESES[:4]
        with torch.no_grad():
            h0, v_p, v_h = encoder.build_initial_state(
                sample_p, sample_h, add_noise=False, device=device
            )
            h_final, _ = collapse_engine.collapse(h0)
            logits = head(h_final, v_p, v_h)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

        for i in range(len(sample_p)):
            pred_label = LABEL_MAP[preds[i].item()]
            conf = probs[i, preds[i]].item() * 100
            print(f"  P: {sample_p[i][:45]:<45}")
            print(f"  H: {sample_h[i][:45]:<45}")
            print(f"  → {pred_label}  ({conf:.1f}% confidence)")
            print()

        del encoder, collapse_engine, head

    print("=" * 65)
    print(" Speed test complete.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Livnium inference speed test")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--n-runs", type=int, default=30,
                        help="Number of timed runs per batch size (after warmup)")
    parser.add_argument("--devices", type=str, nargs="+", default=None,
                        help="Devices to test (e.g. cpu mps). Auto-detects if not set.")
    args = parser.parse_args()

    # Auto-detect devices
    if args.devices is None:
        devices = ["cpu"]
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
    else:
        devices = args.devices

    print(f"Testing on devices: {devices}")
    run_speed_test(
        checkpoint_path=args.checkpoint,
        bert_model=args.bert_model,
        batch_sizes=args.batch_sizes,
        n_runs=args.n_runs,
        devices=devices,
    )


if __name__ == "__main__":
    main()
