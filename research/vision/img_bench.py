"""
img_bench.py — speed test for the pure image-collapse model.

Mirrors noun_bench.py: encode throughput (images/s) across batch sizes on
CPU and MPS, single-image latency, and nearest-noun query time. Pure
inference — same collapse math as img_collapse_pure.py, no grad, no cache.

Usage (from repo root):
    python3 research/vision/img_bench.py                     # CPU + MPS if available
    python3 research/vision/img_bench.py --device cpu        # one device only
    python3 research/vision/img_bench.py --batches 1 64 1024
    python3 research/vision/img_bench.py --iters 20 --warmup 3
"""

import argparse
import time

import torch
import torch.nn.functional as F

from vision_paths import model_path

OUT = model_path("img_collapse_pure.pt")


def collapse_span(h, wells_raw, vals, s):
    """One attraction step per pixel. h (B,D), wells (K,D), vals (B,K)."""
    A = F.normalize(wells_raw, dim=-1)
    for i in range(A.size(0)):
        t = A[i]
        v = vals[:, i:i + 1]
        align = (F.normalize(h, dim=-1) * t).sum(-1, keepdim=True)
        away = F.normalize(h - t, dim=-1)
        h = h - v * s * (1.0 - align) * away
        n = h.norm(dim=-1, keepdim=True)
        h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
    return h


def encode(pix_wells, start, s, imgs_u8, S):
    """imgs_u8 (B,S,S) uint8 -> final state (B,D). Full raster trajectory."""
    vals = imgs_u8.float().view(imgs_u8.size(0), -1) / 255.0
    h = start.expand(vals.size(0), -1).contiguous()
    return collapse_span(h, pix_wells, vals, s)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_device(dev, ck, batches, iters, warmup):
    device = torch.device(dev)
    S = ck["config"]["size"]
    pix = ck["pix_wells"].to(device)
    noun = F.normalize(ck["noun_wells"].to(device), dim=-1)
    start = ck["start"].to(device)
    s = torch.sigmoid(ck["log_strength"].to(device))
    N = noun.size(0)

    print(f"\n=== device: {dev}  ({S}x{S} = {S*S} pixel steps/image, "
          f"{N} nouns, dim {pix.size(1)}) ===")
    print(f"{'batch':>6} {'images/s':>12} {'ms/image':>10} {'ms/batch':>10}")

    with torch.no_grad():
        for B in batches:
            imgs = torch.randint(0, 256, (B, S, S), dtype=torch.uint8, device=device)
            for _ in range(warmup):
                encode(pix, start, s, imgs, S)
            sync(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                encode(pix, start, s, imgs, S)
            sync(device)
            dt = (time.perf_counter() - t0) / iters
            ips = B / dt
            print(f"{B:>6} {ips:>12,.0f} {dt/B*1e3:>10.3f} {dt*1e3:>10.2f}")

        # single-image end-to-end latency (encode + nearest-noun readout)
        one = torch.randint(0, 256, (1, S, S), dtype=torch.uint8, device=device)
        for _ in range(warmup):
            h = F.normalize(encode(pix, start, s, one, S), dim=-1)
            (h @ noun.t()).topk(8)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            h = F.normalize(encode(pix, start, s, one, S), dim=-1)
            (h @ noun.t()).topk(8)
        sync(device)
        print(f"\n  1-image encode+readout: {(time.perf_counter()-t0)/iters*1e3:.3f} ms")

        # nearest-noun query only (state already computed)
        h = F.normalize(encode(pix, start, s, one, S), dim=-1)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters * 20):
            (h @ noun.t()).topk(8)
        sync(device)
        q = (time.perf_counter() - t0) / (iters * 20)
        print(f"  nearest-noun query vs {N} wells: {q*1e3:.3f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=OUT)
    ap.add_argument("--device", default="both", choices=["both", "cpu", "mps", "cuda"])
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 16, 64, 256, 1024])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu")

    devs = []
    if args.device in ("both", "cpu"):
        devs.append("cpu")
    if args.device in ("both", "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    if args.device in ("both", "cuda") and torch.cuda.is_available():
        devs.append("cuda")
    if args.device not in ("both",) and args.device not in devs:
        devs = [args.device]  # honor explicit request even if backend check is odd

    for d in devs:
        bench_device(d, ck, args.batches, args.iters, args.warmup)


if __name__ == "__main__":
    main()
