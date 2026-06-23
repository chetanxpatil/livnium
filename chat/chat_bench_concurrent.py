"""
chat_bench_concurrent.py — how much can ONE CPU push with many parallel replies?

Pure single-reply (batch=1) concurrency: N independent workers, one per core, each
pinned to a single thread, all generating batch=1 premises at once. This is the
"serve many users on CPU, no GPU" regime — keep every reply fast, scale by cores.

Clean timing: each worker loads the model and warms up FIRST, then all workers hit
a barrier and start the timed loop together. Model-load and process-spawn overhead
are excluded — only the steady-state generation loop is measured.

Usage:
    python3 chat_bench_concurrent.py
    python3 chat_bench_concurrent.py --workers 1,2,4,8,10
    python3 chat_bench_concurrent.py --iters 400
"""

import argparse
import os
import time
import statistics
import multiprocessing as mp

PROMPT = "a girl is standing in the doorway"


def worker(wid, ckpt, label, iters, warmup, barrier, q):
    import torch
    torch.set_num_threads(1)                     # pin: one thread per worker
    from sentence_typer import encode_batch, MAXLEN
    from premise_from_hyp import PremiseBrain

    dev = torch.device("cpu")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    stoi, unk, eos = ck["stoi"], ck["unk"], ck["eos"]
    dim, n_words = ck["config"]["dim"], ck["config"]["n_words"]
    align = ck["config"].get("align", False)
    label_every = ck["config"].get("label_every", False)
    m = PremiseBrain(n_words, dim, 0, eos, warm=None, align=align,
                     label_every=label_every).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()

    hyp = encode_batch([PROMPT], stoi, unk, eos).to(dev)
    y = torch.full((1,), label, dtype=torch.long, device=dev)

    def one():
        with torch.no_grad():
            gen, _ = m.generate(hyp, y, MAXLEN, unk=unk)
        n = 0
        for t in gen[0].tolist():
            if t == eos or t == 0:
                break
            n += 1
        return max(n, 1)

    for _ in range(warmup):           # untimed
        one()

    barrier.wait()                    # everyone starts the timed loop together
    lat, toks = [], 0
    t0 = time.perf_counter()
    for _ in range(iters):
        s = time.perf_counter()
        toks += one()
        lat.append((time.perf_counter() - s) * 1000.0)
    loop_wall = time.perf_counter() - t0
    q.put((iters, toks, loop_wall, lat))


def run(ckpt, label, W, iters, warmup):
    barrier = mp.Barrier(W)
    q = mp.Queue()
    procs = [mp.Process(target=worker,
                        args=(i, ckpt, label, iters, warmup, barrier, q))
             for i in range(W)]
    for p in procs:
        p.start()
    res = [q.get() for _ in procs]    # collect before join (queue buffering)
    for p in procs:
        p.join()

    total_replies = sum(r[0] for r in res)
    total_toks = sum(r[1] for r in res)
    window = max(r[2] for r in res)   # concurrent window = slowest loop
    all_lat = [x for r in res for x in r[3]]
    return total_replies / window, total_toks / window, statistics.median(all_lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/premise_from_hyp_align_53.pt")
    ap.add_argument("--label", default=1, type=int)
    ap.add_argument("--iters", default=300, type=int, help="replies per worker (timed)")
    ap.add_argument("--warmup", default=10, type=int)
    ncpu = os.cpu_count() or 8
    ap.add_argument("--workers", default=",".join(
        str(w) for w in sorted(set([1, 2, 4, max(4, ncpu // 2), ncpu]))))
    args = ap.parse_args()

    worker_counts = [int(w) for w in args.workers.split(",")]
    print(f"\ncheckpoint: {args.ckpt}   logical cores: {ncpu}   iters/worker: {args.iters}")
    print("(load + warmup excluded from timing)\n")
    head = f"{'workers':>7} | {'replies/s':>10} | {'tokens/s':>10} | {'median ms/reply':>16} | {'speedup':>8}"
    print(head); print("-" * len(head))

    base_rps = None
    for W in worker_counts:
        rps, tps, med = run(args.ckpt, args.label, W, args.iters, args.warmup)
        if base_rps is None:
            base_rps = rps
        print(f"{W:>7} | {rps:>10.0f} | {tps:>10.0f} | {med:>16.2f} | {rps/base_rps:>7.2f}x")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
