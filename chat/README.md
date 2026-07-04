# chat/ — the chat-brain + the SNLI premise demo

Two projects live in this folder:

1. **The chat-brain** (2026-07-02, active): a personal model ladder — char →
   word → sentence → context → reasoning — trained entirely on one person's
   ChatGPT export, every rung the same vector-collapse engine.
2. **The SNLI premise generator** (the original on-device demo, unchanged,
   documented in the second half of this file).

---

# Wikipedia noun embeddings — pure collapse learns word meaning

`noun_collapse_pure.py` trains word embeddings with **nothing but the collapse
engine** — one 256-d well per word, a start state, and two scalars (strength,
temp). No MLP, no attention, no output layer. The task is CBOW-style fill-in-
the-blank: collapse a state through a noun's ±5-word ordered context and make
the final state point at the missing noun (sampled-softmax CE over nouns).

Trained on ~5M lines of English Wikipedia (~7.5% of the corpus), **94.75M noun
occurrences**, one streaming pass, ~3.2 h on an M-series MacBook (MPS).

## What it learned (probe)

```
cat     -> tabby dog pet felis mouse stray feline
physics -> chemistry mathematics astronomy quantum mechanics astrophysics
war     -> vietnam outbreak world cold ii boer veteran
india   -> gujarat pakistan nepal sikkim delhi bombay punjab bengal
```

Synonyms, hypernyms, sibling terms and geographic manifolds — emerged from
prediction pressure alone, nothing told it what a noun means.

## Quality (SimLex-999, the honest yardstick)

| model | data | SimLex-999 ρ (nouns) |
|---|---|---|
| **pure collapse** | 7.5% of Wikipedia, noun-only | **0.362** (662/666 pairs) |
| word2vec / GloVe (published) | full Wikipedia+Gigaword, billions of tokens | ~0.37–0.44 |
| PPMI+SVD (reference) | full corpus | ~0.38 |

Within the word2vec/GloVe band on a fraction of the data, with no neural
network. Reproduce: `python3 embed_eval.py --model model/noun_collapse_pure.pt`.

## Geometry — a curved ~20-d manifold, not a flat subspace

`intrinsic_dim.py` measures how many dimensions the wells *actually* use. The
answer depends on scale, and the gap is the interesting part:

| measure | value | reading |
|---|---:|---|
| TwoNN intrinsic dim (local) | **~20** | around any noun, its neighbors live on a ~20-d patch — in the normal word-embedding range (word2vec/GloVe ≈ 10–30) |
| participation ratio (global) | ~135 | variance spreads across ~135 effective axes of 256 |
| PCA components for 90% variance | 195 | nearly full linear rank |

Local ~20 but global ~135 means the manifold is **not** a flat 20-d subspace
(that would give local == global) — it's a ~20-d sheet **heavily curved and
folded** through ~135 of the 256 stored axes, the way a 2-d Swiss roll needs 3.
That gap *is* the nonlinear warping the collapse dynamics carve (the same
warping the flow-field plots in `docs/COLLAPSE_VISUALIZATION.md` show) — a linear
method like PPMI+SVD would give a flat subspace with local == global. Measured
on the unit-normalized wells, so the dimension is of the spherical manifold.

Reproduce: `python3 intrinsic_dim.py`.

## Speed (`noun_bench.py`, M-series MacBook; **training was running during the
measurement, so these are pessimistic**)

| batch | CPU words/s | MPS words/s |
|---:|---:|---:|
| 1 | 43,863 | 5,174 |
| 64 | 831,754 | 215,429 |
| 1024 | 1,464,201 | **2,311,023** |

- Embed one 10-word context: **0.23 ms on CPU** (real-time, no GPU needed).
- Bulk throughput: **2.3M words/s** on MPS at batch 1024.
- Nearest-noun query vs 23,758 wells: **0.48 ms** (CPU).

Crossover matches the SNLI demo below: CPU wins single items (launch-bound),
GPU runs away in bulk (~batch 256+). The 10-step sequential collapse walk
amortizes to ~4 µs/window under batch parallelism.

## Use it from Hugging Face

The model ships as a ready-to-push HF repo in `hf_noun_upload/` (standalone
loader, model card, config). To publish it, copy the checkpoint in and upload —
full steps in [`hf_noun_upload/UPLOAD.md`](hf_noun_upload/UPLOAD.md):

```bash
cp model/noun_collapse_pure.pt hf_noun_upload/
pip install -U huggingface_hub && hf auth login
hf upload chetanxpatil/noun-collapse ./hf_noun_upload . --repo-type=model
```

Then anyone can load it with only torch — no framework, no config wiring:

```python
from modeling_noun_collapse import NounCollapse
m = NounCollapse.from_pretrained("noun_collapse_pure.pt")
m.neighbors("india", k=8)          # nearest nouns
m.similarity("cat", "dog")         # cosine
m.encode(["a cat sat on the mat"]) # collapse a sentence -> one vector
```

## As a semantic initialization layer

These wells transfer into the chat-brain: `chat_reply.py --semantic-init
model/noun_collapse_pure.pt` warm-starts every shared word from Wikipedia
geometry before training (matched by word string; same 256-d). Warm start, not
freeze — dialogue usage still moves the wells. Whether it lowers chat dev-NLL is
an open A/B (train with vs without `--semantic-init`, same everything else).

Files: `noun_collapse_pure.py` (train, `--resume`/`--sample-parts`/`--max-occ`) ·
`noun_embed.py` (PPMI+SVD baseline on the same data) · `embed_eval.py` (SimLex) ·
`noun_bench.py` (speed) · `model/noun_collapse_pure.pt` (weights).

---

# Part 1 — The chat-brain

## Data rule (single source of truth)

Everything trains from the RAW ChatGPT export `conversations.json`, walked by
the **canonical path** (`current_node` → parent chain — exactly the thread
ChatGPT displays; edited/abandoned branches skipped). The old
`flattened_conversations.json` is retired: it lost session boundaries, turn
order, and ~half the raw text (616k lines vs 1.19M on the canonical walk).

- Walk implementation: `prep_chat_context.canonical_turns()` — imported by every
  rung that reads raw data. Fix once, fixed everywhere.
- Kept: `user`/`assistant` roles, `text` + `multimodal_text` (image-question
  text recovered). Dropped: system/tool payloads, `thoughts`, `reasoning_recap`,
  `code` tool-calls.

## The ladder

| rung | file | what it learns | checkpoint |
|---|---|---|---|
| CHAR | `char_typer_all.py` | type raw lines back char-by-char; ALL ~2,000 chars get wells, ENTER included; code/markdown/emoji intact | `model/char_typer_all.pt` |
| WORD | `chat_typer.py` | type cleaned sentences back word-by-word; `--max-vocab 0` = a well for every word (~67k) | `model/chat_typer.pt` |
| minting bridge | `chat_typer_live.py` + `char_fingerprint.py` | unseen word at inference → well minted from spelling (no retraining) | grows `chat_typer.pt` |
| SENTENCE (read) | `chat_reply.py :: read()` | context read as a collapse **trajectory** — order-aware; `state_i` = conversation up to word i | inside `chat_reply.pt` |
| CONTEXT (data) | `prep_chat_context.py` | session-aware examples: last 3 turns tagged `<you>`/`<me>` → next reply; sessions never bleed; tail-truncated at 48 words | `data/chat_context.tsv` |
| REASONING | `chat_reply.py` | per typed word: attend over context trajectory **+ its own typed words** (growing memory), pick nearest well, collapse; `--chat` = multi-turn REPL with `thinking` traces | `model/chat_reply.pt` |

One engine everywhere: `h ← h − strength·(1−cos(h,W))·norm(h−W)`. The neural
parts (attention, brain MLP) only pick the next well; collapse executes it.

## Commands

```bash
# char rung (raw export, canonical walk)
python3 char_typer_all.py --batch 1024

# word rung — full vocabulary
python3 chat_typer.py --max-vocab 0

# minting demo (type anything, unseen words are minted live)
python3 chat_typer_live.py

# reasoning data + training + talk
python3 prep_chat_context.py
python3 chat_reply.py --lr 1e-3 --epochs 60 --neg-samples 512
python3 chat_reply.py --chat
```

## Results so far (2026-07-02)

- **Word typer (20k vocab run):** held-out per-word 98.0%, exact-sentence 86.4%,
  **clean OOV-free 100.0%**. The entire exact-sentence gap was OOV words —
  motivated the full-vocab retrain (pending below).
- **Char typer:** CE 6.4 → ~0.002 by step 1500 on 1.19M canonical raw lines,
  1,978 char wells. (First converged run was flatten-sourced; canonical rerun
  in progress at time of writing.)
- **Reasoning v1 (8 epochs @ lr 3e-4):** reached the unigram stage (function-word
  loops) — expected: ~1,000 steps is 30× under-dosed vs the typer. Superseded by
  the 60-epoch run.
- Verified mechanisms (numpy replicas on real data): order-sensitivity of the
  collapse reader (reordered words: meanpool cos 1.000000 vs collapse cos 0.072),
  growing self-attend memory (typed words uniquely retrievable, rank 1), minting
  (unseen-word wells max-cos 0.54 from trained wells — no collisions).

## Pending / known-stale (kept honest)

1. **Word typer full-vocab retrain not done** — `model/chat_typer.pt` is still
   the 20k run. Rerun `python3 chat_typer.py --max-vocab 0`, then retrain
   `chat_reply.py` (26% of reply targets contained `<unk>` under 20k).
2. **`prep_chat_sentences.py` still reads the flatten** — the word corpus
   `data/chat_sentences.txt` predates the single-source rule. Should switch to
   `canonical_turns()` and regenerate before the next word-typer run.
3. **`prep_chat_pairs.py` + `data/chat_pairs.tsv` are deprecated** (flatten-based,
   context-free) — superseded by `prep_chat_context.py`; safe to delete.
4. **Minting still uses the hash fingerprint** — once `char_typer_all.pt` exists,
   swap `char_fingerprint.py`'s deterministic hash for the *trained* char wells
   (collapse the spelling through them to birth the new word's well).
5. Session awareness = hard walls + 3-turn window only. Rung 6 candidate: collapse
   the whole conversation-so-far into a session-state vector fed beside `z`.

---

# Part 2 — Chat demo — the premise generator (on-device)

A tiny (5.98M-param) NLI model trained **only on SNLI**, no pretrained
embeddings. You type a hypothesis; it types back a premise under a fixed label.
Same `generate()` path, made interactive.

> **Note for visitors (from the Reddit writeup).** Two corrections to that post,
> for accuracy:
> 1. **"attention-free" is imprecise.** The *collapse engine* uses no transformer
>    self-attention. But the shipped checkpoint (`premise_from_hyp_align_53.pt`,
>    `align=True`) adds **one lightweight single-head cross-attention step** for
>    cross-sentence alignment (`align_context`, a `torch.softmax` over hypothesis
>    words). Accurate framing: *no full/self-attention; one cross-attention step*.
>    A truly attention-free generator exists too (`premise_from_hyp.pt`, `align=False`).
> 2. **Classifier accuracy is 74.7% dev / 74.4% test**, not 72.7% — the 72.7%
>    figure in the post is a superseded under-estimate. See `SNLI_BASELINES.md`.
>
> Verified this session: 5.975M params, generative-classifier accuracy 52.9% on
> 1,500 SNLI dev pairs, ~4.5 ms/reply on CPU (independent NumPy reimplementation
> of the trained weights). See `CLAIMS_CHECKPOINT_MAP.md`.

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

## How small this really is (the constraints)

This is not a model that saw the internet. Everything it can do comes from a
deliberately tiny, narrow setup:

- **Training data: SNLI only — ~550k sentence pairs**, all image-caption-style
  ("a man in a black shirt is playing a guitar"). One domain, nothing else. No
  books, no web text, no dialogue.
- **Vocabulary: ~20k whole words, learned from scratch (random init).** There
  are **no pretrained embeddings** — no GloVe, no word2vec, no BERT. The model
  starts knowing *nothing* about any word and learns word meaning only from those
  550k captions.
- **No word generalization / no subwords.** It's a whole-word vocabulary, so any
  word outside the 20k simply becomes `<unk>`. It cannot sound out or generalize
  to unseen words the way a subword/BPE model can.
- **~6M parameters**, trained on a laptop.

So when it produces a coherent, grammatical sentence, that's happening with no
external knowledge, a 20k random-initialized vocab, and half a million captions —
which is *why* it's coherent only inside the SNLI domain and falls apart outside
it. That limitation is the point, not a bug.

## Honest notes

- Not a general chatbot and has no awareness. Trained only on SNLI captions, it
  can only produce SNLI-shaped sentences. Fluent grammar emerges fast because
  grammar is local/regular — that is not understanding.
- The classifier sibling (CollapseNLI) reaches 66.1% mean-pool / **74.7% dev,
  74.4% test** with cross-sentence alignment (official leak-free SNLI, no
  pretrained embeddings); the same-footing baseline (SNLI-only, no embeddings) is
  78.2%. The remaining gap is a *mechanism* limit (cross-sentence word
  interaction), not a training-time one. See `SNLI_BASELINES.md`.
- The speed comes from being tiny. Scaling parameters up makes it compute-bound
  and slow, and needs a GPU — you can't keep "5 ms on CPU" at billions of params.

## Files

**Chat-brain:** `char_typer_all.py` · `chat_typer.py` · `chat_typer_live.py` ·
`char_fingerprint.py` · `prep_chat_context.py` · `chat_reply.py` ·
`data/chat_context.tsv`, `data/chat_sentences.txt` ·
`model/chat_typer.pt`, `model/chat_reply.pt`, `model/char_typer_all.pt`

**SNLI demo:** `chat_premise.py` (interactive) · `chat_bench.py` (benchmark) ·
`premise_from_hyp.py` (model) · `sentence_typer.py`, `char_collapse_pure.py`
(encoders) · `model/premise_from_hyp_align_53.pt` (weights) ·
`SNLI_BASELINES.md` (the honest leaderboard comparison).
