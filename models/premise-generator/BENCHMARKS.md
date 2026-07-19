# Benchmarks — CPU vs GPU for the chat (premise) model

All numbers measured on an Apple-silicon MacBook Pro (10 logical cores), 6M-param
model, `model/premise_from_hyp_align_53.pt`, ~9-token replies. Reproduce with the
scripts in this folder. Numbers are measured, not assumed.

## TL;DR

| goal | best strategy | result |
|---|---|---|
| Lowest latency (1 reply) | **CPU, batch 1** | ~5–6 ms / reply |
| Max throughput, on-device | **GPU (MPS), batched** | ~107k tok/s @ batch 256 |
| Max throughput on CPU | **single process, batched** | ~24k tok/s @ batch 256 |
| Many users at once on CPU | spawning processes **does not help** | ~2x ceiling (memory-bound) |

Two rules fall out of this:
1. **For one reply, use the CPU** — at batch 1 the decode is launch-bound, so the
   GPU's per-op overhead loses to plain CPU.
2. **For volume, batch — don't spawn.** The model is memory-bandwidth bound, so a
   single batched pass (weights loaded once, reused across the batch) beats many
   independent processes (each re-streaming the weights from RAM).

## 1. Batch sweep — latency & throughput (`chat_bench_batch.py`)

| batch | CPU ms/reply | CPU tok/s | GPU ms/reply | GPU tok/s |
|---:|---:|---:|---:|---:|
| 1 | **5.89** | 1,699 | 11.66 | 858 |
| 2 | 6.00 | 1,667 | 6.12 | 1,634 |
| 4 | 3.15 | 3,176 | 3.09 | 3,236 |
| 8 | 1.74 | 5,754 | 1.23 | 8,117 |
| 16 | 0.90 | 11,114 | 0.65 | 15,373 |
| 32 | 0.60 | 16,534 | 0.35 | 28,452 |
| 64 | 0.54 | 18,559 | 0.19 | 54,045 |
| 128 | 0.41 | 24,386 | 0.12 | 82,687 |
| 256 | 0.41 | 24,485 | 0.09 | **106,673** |

**Crossover: the GPU overtakes the CPU on per-reply latency at batch ≥ 4.** Below
that, the CPU wins because the work is launch-bound (per-op kernel launch cost
dominates the tiny math); above it, the GPU's parallelism amortizes that fixed
cost and runs away — ~4.4x the CPU's throughput at batch 256.

## 2. Parallel single-replies on CPU (`chat_bench_concurrent.py`)

N independent batch-1 workers, one per core, load + warmup excluded from timing:

| workers | replies/s | tok/s | median ms/reply | speedup |
|---:|---:|---:|---:|---:|
| 1 | 162 | 1,615 | 6.22 | 1.00x |
| 2 | 218 | 2,185 | 9.07 | 1.35x |
| 4 | 233 | 2,331 | 16.13 | 1.44x |
| 5 | 274 | 2,741 | 17.49 | 1.70x |
| 10 | 317 | 3,174 | 30.63 | 1.96x |

This **does not scale** — ~2x at 10 workers, and per-reply latency degrades 6 → 31
ms. The reason: each reply streams the 20k×256 embedding table (~20 MB) from RAM,
so the workload is **memory-bandwidth bound, not compute bound**. Ten processes
each pull their own copy of that traffic; the memory controller saturates at ~2
cores' worth and the rest just queue. More cores ≠ more throughput when the
bottleneck is RAM. This is exactly why batching (one process, weights reused) beats
multiprocessing on CPU here (~24k vs ~3k tok/s).

## How to reproduce

```bash
cd models/premise-generator
python3 chat_bench.py                 # single-reply latency, auto device
python3 chat_bench_batch.py           # batch sweep, CPU vs MPS (table + crossover)
python3 chat_bench_concurrent.py      # parallel batch-1 workers on CPU
```
