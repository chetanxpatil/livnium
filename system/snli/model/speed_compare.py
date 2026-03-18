"""
Livnium Speed Comparison
========================
Head-to-head inference speed: BERT-joint vs Livnium-native encoder.

Reports latency and throughput at multiple batch sizes so you can see
exactly how much faster the 772K-param native encoder is vs 110M-param BERT.

Usage:
    python3 speed_compare.py \
        --bert-checkpoint   ../../../pretrained/bert-joint/best_model.pt \
        --native-checkpoint ../../../pretrained/livnium-native/best_model.pt \
        --batch-sizes 1 8 32 64 \
        --n-runs 30
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
from tasks.snli import BERTSNLIEncoder, LivniumNativeEncoder, SNLIHead
from utils.vocab import Vocabulary

# ── Sample sentences ──────────────────────────────────────────────────────────
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
LABEL_MAP = {0: "ENTAIL", 1: "CONTRA", 2: "NEUTRAL"}


def build_batch(size):
    n = len(PREMISES)
    return [PREMISES[i % n] for i in range(size)], \
           [HYPOTHESES[i % n] for i in range(size)]


def time_fn(fn, n_runs=30, warmup=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


def load_bert_pipeline(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt['args']
    encoder = BERTSNLIEncoder(freeze=True).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    collapse = VectorCollapseEngine(
        dim=args.dim, num_layers=args.num_layers
    ).to(device)
    collapse.load_state_dict(ckpt['collapse_engine'])
    collapse.eval()
    head = SNLIHead(dim=args.dim).to(device)
    head.load_state_dict(ckpt['head'])
    head.eval()
    param_count = sum(p.numel() for p in encoder.parameters())
    return encoder, collapse, head, param_count, args


def load_native_pipeline(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt['args']
    vocab = ckpt['vocab']
    # Read max_seq_len from the saved pos_enc shape — avoids size mismatch
    saved_max_seq_len = ckpt['encoder']['pos_enc'].shape[1]
    encoder = LivniumNativeEncoder(
        vocab_size=len(vocab),
        dim=args.livnium_dim,
        num_layers=args.livnium_layers,
        nhead=args.livnium_nhead,
        ff_mult=getattr(args, 'livnium_ff_mult', 4),
        use_cross_encoder=getattr(args, 'livnium_cross_encoder', True),
        max_seq_len=saved_max_seq_len,
    ).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    collapse = VectorCollapseEngine(
        dim=args.livnium_dim, num_layers=args.num_layers
    ).to(device)
    collapse.load_state_dict(ckpt['collapse_engine'])
    collapse.eval()
    head = SNLIHead(dim=args.livnium_dim).to(device)
    head.load_state_dict(ckpt['head'])
    head.eval()
    param_count = sum(p.numel() for p in encoder.parameters())
    return encoder, collapse, head, vocab, param_count, args


def run_comparison(bert_ckpt, native_ckpt, batch_sizes, n_runs, device):
    print(f"\n{'═'*68}")
    print(f"  LIVNIUM SPEED COMPARISON  —  device: {str(device).upper()}")
    print(f"{'═'*68}")

    # Load both pipelines
    print(f"\nLoading BERT-joint pipeline...")
    bert_enc, bert_col, bert_head, bert_params, bert_args = \
        load_bert_pipeline(bert_ckpt, device)
    print(f"  encoder params: {bert_params:,}  (dim={bert_args.dim})")

    print(f"Loading Livnium-native pipeline...")
    nat_enc, nat_col, nat_head, nat_vocab, nat_params, nat_args = \
        load_native_pipeline(native_ckpt, device)
    print(f"  encoder params: {nat_params:,}  (dim={nat_args.livnium_dim})")
    print(f"  speedup just on params: {bert_params/nat_params:.0f}x smaller encoder")

    print(f"\n{'─'*68}")
    print(f"  {'Batch':>5}  {'BERT (ms)':>10}  {'Native (ms)':>12}  "
          f"{'Speedup':>8}  {'BERT ex/s':>10}  {'Native ex/s':>12}")
    print(f"  {'─'*63}")

    speedups = []

    for bs in batch_sizes:
        premises, hypotheses = build_batch(bs)

        # Tokenize once for native (vocab-based)
        max_len = 128
        prem_ids = torch.tensor(
            [nat_vocab.encode(p, max_len=max_len) for p in premises],
            dtype=torch.long, device=device
        )
        hyp_ids = torch.tensor(
            [nat_vocab.encode(h, max_len=max_len) for h in hypotheses],
            dtype=torch.long, device=device
        )

        def bert_pipeline():
            with torch.no_grad():
                h0, vp, vh = bert_enc.build_initial_state(
                    premises, hypotheses, add_noise=False, device=device
                )
                hf, _ = bert_col.collapse(h0)
                return bert_head(hf, vp, vh).argmax(dim=-1)

        def native_pipeline():
            with torch.no_grad():
                h0, vp, vh = nat_enc.build_initial_state(prem_ids, hyp_ids)
                hf, _ = nat_col.collapse(h0)
                return nat_head(hf, vp, vh).argmax(dim=-1)

        bert_ms, bert_std = time_fn(bert_pipeline, n_runs=n_runs)
        nat_ms,  nat_std  = time_fn(native_pipeline, n_runs=n_runs)

        speedup = bert_ms / nat_ms
        speedups.append(speedup)

        bert_exs = bs * 1000 / bert_ms
        nat_exs  = bs * 1000 / nat_ms

        print(f"  {bs:>5}  {bert_ms:>9.1f}±{bert_std:.1f}  "
              f"{nat_ms:>10.1f}±{nat_std:.1f}  "
              f"{speedup:>7.1f}x  {bert_exs:>10.0f}  {nat_exs:>12.0f}")

    print(f"  {'─'*63}")
    print(f"  Average speedup: {np.mean(speedups):.1f}x  "
          f"(range: {min(speedups):.1f}x – {max(speedups):.1f}x)")

    # ── Pipeline breakdown ────────────────────────────────────────────────────
    bs = min(32, max(batch_sizes))
    premises_b, hypotheses_b = build_batch(bs)
    prem_ids_b = torch.tensor(
        [nat_vocab.encode(p, max_len=128) for p in premises_b],
        dtype=torch.long, device=device
    )
    hyp_ids_b = torch.tensor(
        [nat_vocab.encode(h, max_len=128) for h in hypotheses_b],
        dtype=torch.long, device=device
    )

    print(f"\n  Pipeline breakdown at batch={bs}:")
    print(f"  {'─'*50}")

    # BERT breakdown
    def bert_enc_only():
        with torch.no_grad():
            bert_enc.build_initial_state(
                premises_b, hypotheses_b, add_noise=False, device=device
            )

    with torch.no_grad():
        h0_bert, vp_b, vh_b = bert_enc.build_initial_state(
            premises_b, hypotheses_b, add_noise=False, device=device
        )

    def bert_col_only():
        with torch.no_grad():
            hf, _ = bert_col.collapse(h0_bert)
            bert_head(hf, vp_b, vh_b)

    bert_enc_ms, _ = time_fn(bert_enc_only, n_runs=n_runs)
    bert_col_ms, _ = time_fn(bert_col_only, n_runs=n_runs)
    bert_total = bert_enc_ms + bert_col_ms

    print(f"  BERT-joint:")
    print(f"    encoder (BERT):   {bert_enc_ms:6.1f} ms  "
          f"({bert_enc_ms/bert_total*100:.0f}%)")
    print(f"    collapse + head:  {bert_col_ms:6.1f} ms  "
          f"({bert_col_ms/bert_total*100:.0f}%)")
    print(f"    total:            {bert_total:6.1f} ms")

    # Native breakdown
    def native_enc_only():
        with torch.no_grad():
            nat_enc.build_initial_state(prem_ids_b, hyp_ids_b)

    with torch.no_grad():
        h0_nat, vp_n, vh_n = nat_enc.build_initial_state(prem_ids_b, hyp_ids_b)

    def native_col_only():
        with torch.no_grad():
            hf, _ = nat_col.collapse(h0_nat)
            nat_head(hf, vp_n, vh_n)

    nat_enc_ms, _ = time_fn(native_enc_only, n_runs=n_runs)
    nat_col_ms, _ = time_fn(native_col_only, n_runs=n_runs)
    nat_total = nat_enc_ms + nat_col_ms

    print(f"\n  Livnium-native:")
    print(f"    encoder (native): {nat_enc_ms:6.1f} ms  "
          f"({nat_enc_ms/nat_total*100:.0f}%)")
    print(f"    collapse + head:  {nat_col_ms:6.1f} ms  "
          f"({nat_col_ms/nat_total*100:.0f}%)")
    print(f"    total:            {nat_total:6.1f} ms")

    print(f"\n  Encoder-only speedup: {bert_enc_ms/nat_enc_ms:.1f}x faster")
    print(f"  Full pipeline speedup: {bert_total/nat_total:.1f}x faster")

    # ── Accuracy reminder ─────────────────────────────────────────────────────
    print(f"\n  Accuracy reference (SNLI dev):")
    print(f"    BERT-joint:      82.06%  (110M encoder params)")
    print(f"    Livnium-native:  ~80.3%  (best epoch so far, {nat_params:,} params)")
    print(f"    Param ratio:     {bert_params/nat_params:.0f}x fewer params in native encoder")

    print(f"\n{'═'*68}\n")


def main():
    parser = argparse.ArgumentParser(description='Livnium speed comparison')
    parser.add_argument('--bert-checkpoint',   type=str, required=True)
    parser.add_argument('--native-checkpoint', type=str, required=True)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 32, 64])
    parser.add_argument('--n-runs',      type=int, default=30)
    parser.add_argument('--device',      type=str, default=None,
                        help='Force device (cpu/mps/cuda). Auto-detects if omitted.')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    run_comparison(
        bert_ckpt=args.bert_checkpoint,
        native_ckpt=args.native_checkpoint,
        batch_sizes=args.batch_sizes,
        n_runs=args.n_runs,
        device=device,
    )


if __name__ == '__main__':
    main()
