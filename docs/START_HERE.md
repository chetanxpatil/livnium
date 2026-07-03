# Start Here

*Five sections. Read them in order. ~10 minutes.*

---

## 1. What this is, and why

A chat model built on **one mechanism** — vector collapse — instead of a transformer. No self-attention blocks, no pretrained embeddings, no external knowledge. Every layer of the system (reading characters, reading words, reading a conversation, writing a reply) uses the same one-line update rule, and the whole thing trains from scratch on a laptop.

The purpose: find out how far a single geometric primitive can go as a language engine, and measure it honestly. The repo keeps its failures on record next to its wins (`docs/FINDINGS.md`) — every claim below has a number behind it.

---

## 2. What to read, step by step

1. This file, top to bottom.
2. Root `README.md` — the project's origin (a conserved cube geometry) and the honest benchmark history.
3. `chat/README.md` — the current work: the chat-brain ladder + the SNLI premise generator, with results and known-stale items.
4. `docs/COLLAPSE_VISUALIZATION.md` — *see* the attractor basins: flow fields and grid warping of the trained engine.
5. Code, in this order: `chat/premise_from_hyp.py` (smallest complete generator) → `chat/chat_typer.py` → `chat/chat_reply.py` (the full model).

---

## 3. Run it in two minutes

```bash
git clone https://github.com/chetanxpatil/livnium.git && cd livnium
python3 -m pytest -q            # proven core: 67 tests, no dependencies
pip3 install torch              # the chat models need torch
cd chat && python3 chat_premise.py
```

```
you > is the girl standing
ai  > a girl in a pink shirt standing in a doorway.   [neutral]
```

That's a 5.98M-parameter model, trained only on SNLI, answering in ~5 ms on CPU. Checkpoint ships in `chat/model/` — no training needed to try it.

To train the chat model yourself (two-stage: general fluency on DailyDialog, then personal voice):

```bash
python3 prep_dailydialog.py
python3 chat_reply.py --data data/dd_context.tsv \
    --extra-vocab data/chat_context.tsv --ckpt model/chat_reply_general.pt
python3 chat_reply.py --data data/chat_context.tsv \
    --resume model/chat_reply_general.pt --pos-anneal 0 --lr 5e-4
python3 chat_reply.py --chat
```

---

## 4. The mechanism: vector collapse

One update rule runs everything. A hidden state `h` is pulled toward a target vector `W` — a "well," one learned vector per word:

```
h ← h − strength · (1 − cos(h, W)) · norm(h − W)
```

- `(1 − cos(h, W))` — how misaligned the state is. Already at the well → factor ~0, nothing moves. Far away → strong pull.
- `norm(h − W)` — the unit direction away from the well; subtracting it moves `h` toward it.
- `strength` — one learned scalar. Norm clipped at 10 so nothing explodes.

Each well is a **point attractor**. Collapsing through a sentence word-by-word traces a **trajectory**, and that trajectory is order-sensitive: reordering the words gives mean-pool cosine 1.000 but collapse cosine 0.072. Mean-pooling can't tell "dog bites man" from "man bites dog"; collapse can. Each step is O(1) — no attention matrix — which is where the CPU speed comes from.

This sits in **dynamical systems, not physics**: the trained step admits an empirical Lyapunov candidate `V(h) = 1 − cos(h, target)` that was monotone non-increasing on 100% of 12,000 measured steps, with predominantly contracting Jacobians (mean singular value ≈ 0.89). Precise claims and caveats: `chat/LYAPUNOV_TEST.md`.

**How the models use it.** The neural parts only *pick* the next well; collapse *executes* the move.

- *Premise generator* (`premise_from_hyp.py`): inverts NLI — given (hypothesis + label), it must **type the premise back** word by word. Thought vector `z = think([meanpool(hyp) ; label])`, then per step: a small MLP builds a query from `[h ; z]`, cosine against all wells picks the word, `h` collapses onto it. Scoring the true premise under all 3 labels gives a free generative classifier. No self-attention; the best checkpoint adds one cross-attention step for word alignment.
- *Chat model* (`chat_reply.py`): READ — collapse through the conversation, keeping every intermediate state (a trajectory, not a pooled vector). THINK — `z = linear(final state)`. WRITE — per word: attend over the trajectory *plus its own already-typed words* (growing memory), pick the nearest well, collapse. Out-of-vocab context words get wells minted from their spelling (`char_fingerprint.py`), so the reader never sees `<unk>` — but the writer can only say trained words.

---

## 5. The numbers, with baselines

| model | task | score | baseline it must beat |
|---|---|---|---|
| CollapseNLI + alignment | SNLI classification | **74.7% dev / 74.4% test** | hypothesis-only artifact 61.5%; GloVe avg 60.7%; same-footing BiLSTM 78.2% (gap = mechanism, not training) |
| Supervised collapse (earlier) | SNLI | 68.9% | ablation: collapse layers add **+4.86%** over a linear head (68.9 vs 64.1) |
| Premise generator | generative NLI | ~53% gold-label match | chance 33% |
| Word typer (20k vocab) | type sentences back | 98.0% per-word; **100.0%** OOV-free exact | — |
| Speed (premise model) | CPU, per reply | **~5 ms**, 1,630 tok/s | GPU is *slower* (13 ms) — decode is launch-bound |

What it is **not**: not a general chatbot, no understanding, coherent only in its training domains (SNLI captions; everyday dialogue). Grammar emerges fast because grammar is local — that's fluency, not comprehension. The cube geometry alone carries no meaning (38%, ~chance) and ANLI sits at chance — both documented, kept, and explained in `results/RESULTS.md` and `docs/LIMITS.md`.

Full leaderboard and kill-tests: `chat/SNLI_BASELINES.md` · claim-to-checkpoint map: `chat/CLAIMS_CHECKPOINT_MAP.md`
