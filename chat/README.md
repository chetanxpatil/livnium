# Chat demo — the premise generator (attention-free, on-device)

A tiny (5.98M-param) NLI model trained **only on SNLI**, no pretrained
embeddings, no attention. You type a hypothesis; it types back a premise under a
fixed label. Same `generate()` path, made interactive.

```bash
cd chat
# talk to it
python3 chat_premise.py            # defaults to model/premise_from_hyp_align_53.pt, label=neutral
python3 chat_premise.py --label entail

# benchmark latency / throughput on your machine
python3 chat_bench.py              # auto device (MPS on Mac)
python3 chat_bench.py --device cpu
```

Example:

```
you > is the girl standing
ai  > a girl in a pink shirt standing in a doorway.   [neutral]
```

## Measured numbers

- **Generative-classifier accuracy:** ~53% (gold-label match of the generated premise).
- **Speed (Apple-silicon MacBook, `chat_bench.py`, 40 replies):**

  | device | median / reply | throughput |
  |---|---:|---:|
  | MPS (GPU) | 13.1 ms | 591 tok/s |
  | **CPU** | **5.3 ms** | **1,630 tok/s** |

  CPU wins — the ~9-step decode is launch-bound, not compute-bound, so the GPU's
  per-op overhead costs more than its compute saves. Runs best with no accelerator.

## Honest notes

- Not a general chatbot and has no awareness. Trained only on SNLI captions, it
  can only produce SNLI-shaped sentences. Fluent grammar emerges fast because
  grammar is local/regular — that is not understanding.
- The classifier sibling (CollapseNLI) reaches 66.1% mean-pool / **72.7%** with
  cross-sentence alignment on SNLI dev; the same-footing baseline (SNLI-only, no
  embeddings) is 78.2%. The remaining gap is a *mechanism* limit (cross-sentence
  word interaction), not a training-time one. See `SNLI_BASELINES.md`.
- The speed comes from being tiny. Scaling parameters up makes it compute-bound
  and slow, and needs a GPU — you can't keep "5 ms on CPU" at billions of params.

## Files

`chat_premise.py` (interactive) · `chat_bench.py` (benchmark) ·
`premise_from_hyp.py` (model) · `sentence_typer.py`, `char_collapse_pure.py`
(encoders) · `model/premise_from_hyp_align_53.pt` (weights) ·
`SNLI_BASELINES.md` (the honest leaderboard comparison).
