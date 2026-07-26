# Measured Results

- Run date: 2026-07-26
- PyTorch: 2.9.1
- Training: 500 steps per seed, paths of 1–3 observer operations
- Evaluation: 4,096 examples per condition, including paths of 4–6 operations
- Address space: 27 macro × 27 micro = 729 payloads
- Random exact-match chance: 1/729 = 0.1372%

## Three independent seeds

| Measurement | Seed 7 | Seed 13 | Seed 29 | Mean |
|---|---:|---:|---:|---:|
| Unseen pile, seen path lengths | 100% | 100% | 100% | 100% |
| Unseen pile, longer paths | 100% | 100% | 100% | 100% |
| Wrong-pile control | 0.000% | 0.000% | 0.000% | 0.000% |
| Deranged-instruction control | 0.195% | 0.195% | 0.122% | 0.171% |
| Reversed path, all examples | 18.701% | 18.628% | 19.189% | 18.840% |
| Endpoints changed by reversal | 81.299% | 81.372% | 80.811% | 81.160% |
| Reversed path, changed endpoints only | 0.000% | 0.000% | 0.000% | 0.000% |
| No-flow control | 0.342% | 0.317% | 0.220% | 0.293% |
| Random-router control | 0.171% | 0.293% | 0.366% | 0.277% |
| Target intervention followed | 100% | 100% | 100% | 100% |
| Non-target intervention stable | 100% | 100% | 100% | 100% |
| Learned all eight token meanings | 100% | 100% | 100% | 100% |
| Exact inverse round trip | 100% | 100% | 100% | 100% |

Hard retrieval accuracy was 100% separately at every path length from 1 through
6 for every seed. Neither the evaluation payload values nor their arrangement
were used for training.

## Verdict

**PASS for the narrow prototype claim:**

> A trainable neural action head learned the hidden meanings of eight instruction
> tokens and composed their reversible observer operations over a persistent
> two-level Livnium pile. The fixed read interface returned genuinely unseen
> payload values perfectly, and the learned action dictionary generalized from
> paths of length 1–3 to paths of length 4–6.

The wrong-pile, random-router, no-flow, and deranged-instruction controls rule out
simple answer memorization in this task. Reversing action order changed 81.16% of
endpoints; accuracy on those changed paths was 0%, confirming that the learned
execution is order-sensitive. Paired interventions show that the fixed reader
returns the payload at the addressed cell and ignores changes elsewhere.

## Claim boundary

The current result is a read-only router over synthetic payload tokens. The
architecture forces answers through a fixed pile-read operation; the learned
part is the instruction-to-operation dictionary, not a decision to use memory or
the read algorithm itself. It does not establish natural-language understanding,
content-based addressing, writable/self-organizing memory, superiority to flat
memory, or reversibility of the neural controller. Soft mixtures are used only
during training; exact reversibility belongs to the hard observer trace and pile
substrate. The transitive observer shifts are a new reversible memory-interface
extension; they move only the read address and are not part of Livnium Core's
canonical 24 rigid rotations.

Raw measurements:

- `results/metrics.json`
- `results/metrics_seed13.json`
- `results/metrics_seed29.json`
