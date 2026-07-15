"""
chat_bench_batch.py — find the CPU vs GPU (MPS) crossover for the premise generator.

Single-reply latency (batch=1) is launch-bound, so CPU wins. But if you want many
variations at once, batching amortizes the GPU's fixed per-op launch cost. This
sweeps batch size on each device and prints where the GPU overtakes the CPU.

For each batch size B it generates B premises at once (same prompt repeated) and
reports total batch latency, per-sequence latency, and total throughput.

Usage:
    python3 chat_bench_batch.py
    python3 chat_bench_batch.py --ckpt model/premise_from_hyp_align_53.pt
    python3 chat_bench_batch.py --batches 1,2,4,8,16,32,64,128,256,512
"""

import argparse
import os
import time
import statistics
import torch

from sentence_typer import encode_batch, MAXLEN
from premise_from_hyp import PremiseBrain

PROMPT = "a girl is standing in the doorway"


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    stoi, unk, eos = ck["stoi"], ck["unk"], ck["eos"]
    dim, n_words = ck["config"]["dim"], ck["config"]["n_words"]
    align = ck["config"].get("align", False)
    label_every = ck["config"].get("label_every", False)
    m = PremiseBrain(n_words, dim, 0, eos, warm=None, align=align,
                     label_every=label_every).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, stoi, unk, eos


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_device(ckpt, device, batches, label, runs, warmup):
    m, stoi, unk, eos = load_model(ckpt, device)
    one = encode_batch([PROMPT], stoi, unk, eos).to(device)   # (1, Lh)

    def run_batch(B):
        hyp = one.repeat(B, 1)
        y = torch.full((B,), label, dtype=torch.long, device=device)
        with torch.no_grad():
            gen, _ = m.generate(hyp, y, MAXLEN, unk=unk)
        sync(device)
        # count non-pad/eos tokens across the whole batch
        ntok = 0
        for row in gen.tolist():
            for t in row:
                if t == eos or t == 0:
                    break
                ntok += 1
        return max(ntok, B)

    results = {}
    for B in batches:
        for _ in range(warmup):
            run_batch(B)
        times, toks = [], 0
        for _ in range(runs):
            t0 = time.perf_counter()
            ntok = run_batch(B)
            times.append((time.perf_counter() - t0) * 1000.0)
            toks = ntok
        med = statistics.median(times)
        results[B] = {"batch_ms": med, "per_seq_ms": med / B,
                      "tok_s": toks / (med / 1000.0)}
    return results


def main():
    ap = argparse.ArgumentParser()
    default_ckpt = os.path.join(os.path.dirname(__file__), "model", "premise_from_hyp_align_53.pt")
    ap.add_argument("--ckpt", default=default_ckpt)
    ap.add_argument("--label", default=1, type=int)
    ap.add_argument("--batches", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--runs", default=20, type=int)
    ap.add_argument("--warmup", default=3, type=int)
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    elif torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    all_res = {}
    for dev in devices:
        all_res[dev.type] = bench_device(args.ckpt, dev, batches, args.label,
                                          args.runs, args.warmup)

    # table
    print(f"\ncheckpoint: {args.ckpt}   prompt='{PROMPT}'   runs={args.runs}\n")
    head = f"{'batch':>6} | " + " | ".join(
        f"{d:>26}" for d in all_res)
    print(head); print("-" * len(head))
    for B in batches:
        cells = []
        for d in all_res:
            r = all_res[d][B]
            cells.append(f"{r['batch_ms']:7.1f}ms  {r['per_seq_ms']:6.2f}/seq  {r['tok_s']:7.0f}t/s")
        print(f"{B:>6} | " + " | ".join(cells))

    # crossover (per-sequence latency: where gpu < cpu)
    if len(all_res) == 2:
        gpu = "mps" if "mps" in all_res else "cuda"
        cross = next((B for B in batches
                      if all_res[gpu][B]["per_seq_ms"] < all_res["cpu"][B]["per_seq_ms"]), None)
        print()
        if cross:
            print(f"crossover: {gpu.upper()} beats CPU on per-reply latency at batch >= {cross}")
        else:
            print(f"no crossover in tested range — CPU faster per reply through batch {batches[-1]}")


if __name__ == "__main__":
    main()
