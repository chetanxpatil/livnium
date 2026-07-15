"""
noun_bench.py — speed of the pure-collapse noun model.

Three numbers, the way chat/BENCHMARKS.md reports them:
  1. LOOKUP    — a trained word's vector is just a table row (the O(1) case).
  2. ENCODE    — collapse a context window into a state (the O(L) READ path,
                 the real work: L attraction steps per window).
  3. NEIGHBORS — one probe = cosine of a vector against all noun wells.

Measured on CPU and MPS across batch sizes, warm-up excluded.

Usage:
    python3 noun_bench.py
    python3 noun_bench.py --ckpt model/noun_collapse_pure.pt --window 10
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F


def collapse_encode(ctx_ids, A, start, strength):
    """The pure READ path, lifted straight from noun_collapse_pure.encode."""
    mask = ctx_ids != 0
    h = start.expand(ctx_ids.size(0), -1).contiguous()
    s = strength
    for i in range(ctx_ids.size(1)):
        t = A[ctx_ids[:, i]]
        m = mask[:, i].float().unsqueeze(-1)
        align = (F.normalize(h, dim=-1) * t).sum(-1)
        away = F.normalize(h - t, dim=-1)
        h = h + m * (-s * (1.0 - align).unsqueeze(-1) * away)
        n = h.norm(dim=-1, keepdim=True)
        h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
    return h


def bench_device(dev, A, start, strength, noun_ids, V, window, batches, iters):
    A = A.to(dev); start = start.to(dev); noun_ids = noun_ids.to(dev)
    AN = A[noun_ids]

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()
        elif dev.type == "cuda":
            torch.cuda.synchronize()

    print(f"\n=== device: {dev} ===")
    print(f"{'batch':>7} {'ms/encode':>11} {'windows/s':>12} {'words/s':>12}")
    for B in batches:
        ctx = torch.randint(1, V, (B, window), device=dev)
        for _ in range(3):                         # warm-up
            collapse_encode(ctx, A, start, strength)
        sync()
        t0 = time.time()
        for _ in range(iters):
            collapse_encode(ctx, A, start, strength)
        sync()
        dt = (time.time() - t0) / iters
        wps = B / dt
        print(f"{B:>7} {dt*1e3/B:>11.4f} {wps:>12,.0f} {wps*window:>12,.0f}")

    # neighbor query (one probe against all nouns)
    v = F.normalize(A[noun_ids[0]], dim=-1)
    for _ in range(5):
        (AN @ v).argmax()
    sync()
    t0 = time.time()
    for _ in range(200):
        (AN @ v).topk(8)
    sync()
    print(f"neighbor query (vs {len(noun_ids):,} nouns): "
          f"{(time.time()-t0)/200*1e3:.3f} ms")


def main():
    ap = argparse.ArgumentParser()
    default_ckpt = os.path.join(os.path.dirname(__file__), "model", "noun_collapse_pure.pt")
    ap.add_argument("--ckpt", default=default_ckpt)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    A = F.normalize(ck["wells"], dim=-1)
    start = ck["start"]
    strength = float(ck["strength"])
    noun_ids = torch.tensor(ck["noun_ids"])
    V = A.size(0)
    print(f"model: {V:,} wells x {A.size(1)}d   {len(noun_ids):,} nouns   "
          f"strength {strength:.3f}   window {args.window}")

    devs = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devs.append(torch.device("mps"))
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    batches = [1, 8, 64, 256, 1024]
    for d in devs:
        bench_device(d, A, start, strength, noun_ids, V, args.window,
                     batches, args.iters)


if __name__ == "__main__":
    main()
