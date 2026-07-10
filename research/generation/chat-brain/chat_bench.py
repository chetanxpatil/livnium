"""
chat_bench.py — measure the REAL per-reply chat latency of the premise generator.

Loads your actual checkpoint and runs the same generate() path chat_premise.py
uses, on a set of held prompts, and reports:
    - per-reply latency (ms): mean / median / p90
    - tokens generated per reply
    - tokens/sec (decode throughput)
on whatever device is available (mps on your Mac).

Usage:
    python3 chat_bench.py
    python3 chat_bench.py --ckpt model/premise_from_hyp_align_53.pt --device mps
    python3 chat_bench.py --device cpu        # force CPU to compare
    python3 chat_bench.py --runs 50
"""

import argparse
import time
import statistics
import torch

from sentence_typer import encode_batch, MAXLEN
from premise_from_hyp import PremiseBrain

PROMPTS = [
    "a girl is standing in the doorway",
    "two men are playing football on a field",
    "the dog ran across the park chasing a ball",
    "a woman in a red dress is dancing",
    "people are waiting at the train station",
    "a child is eating ice cream in the sun",
    "the chef is cooking in a busy kitchen",
    "a man plays guitar on the street corner",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/premise_from_hyp_align_53.pt")
    ap.add_argument("--label", default=1, type=int, help="0 entail / 1 neutral / 2 contra")
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (auto if unset)")
    ap.add_argument("--runs", default=40, type=int, help="timed replies")
    ap.add_argument("--warmup", default=5, type=int)
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    stoi, unk, eos = ck["stoi"], ck["unk"], ck["eos"]
    dim, n_words = ck["config"]["dim"], ck["config"]["n_words"]
    align = ck["config"].get("align", False)
    label_every = ck["config"].get("label_every", False)

    model = PremiseBrain(n_words, dim, 0, eos, warm=None, align=align,
                         label_every=label_every).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    def one_reply(text):
        hyp = encode_batch([text], stoi, unk, eos).to(device)
        y = torch.full((1,), args.label, dtype=torch.long, device=device)
        with torch.no_grad():
            gen, _ = model.generate(hyp, y, MAXLEN, unk=unk)
        sync()
        ntok = 0
        for t in gen[0].tolist():
            if t == eos or t == 0:
                break
            ntok += 1
        return max(ntok, 1)

    # warmup (first MPS calls compile kernels — exclude them)
    for i in range(args.warmup):
        one_reply(PROMPTS[i % len(PROMPTS)])

    lat_ms, toks = [], []
    for i in range(args.runs):
        text = PROMPTS[i % len(PROMPTS)]
        t0 = time.perf_counter()
        ntok = one_reply(text)
        dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt); toks.append(ntok)

    lat_ms.sort()
    mean = statistics.mean(lat_ms)
    median = statistics.median(lat_ms)
    p90 = lat_ms[int(0.9 * (len(lat_ms) - 1))]
    avg_tok = statistics.mean(toks)
    tps = (sum(toks) / (sum(lat_ms) / 1000.0))

    print(f"\ncheckpoint : {args.ckpt}")
    print(f"device     : {device}   (align={align})")
    print(f"params     : {n_params/1e6:.2f}M")
    print(f"runs       : {args.runs} timed  (+{args.warmup} warmup)")
    print("-" * 44)
    print(f"latency/reply  mean {mean:7.1f} ms   median {median:7.1f} ms   p90 {p90:7.1f} ms")
    print(f"tokens/reply   avg  {avg_tok:7.1f}")
    print(f"throughput     {tps:7.1f} tokens/sec")
    print()


if __name__ == "__main__":
    main()
