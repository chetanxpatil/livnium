# Start Here — What Livnium Is, How Collapse Works, and What the Chat Model Does

*Written for new collaborators. Read this top to bottom before touching code.*

---

## 1. What is being built, and why

Livnium started as a piece of conserved geometry (a 3×3×3 cube state space — see the root `README.md`). Out of that came one idea that survived every honest test: **vector collapse** — a way to move a hidden state through space by pulling it toward learned attractor points ("wells") instead of using a transformer.

The current goal is a **chat model built entirely on that one mechanism**. No transformer blocks, no pretrained embeddings, no external knowledge. Every layer of the system — reading characters, reading words, reading a conversation, writing a reply — uses the *same* collapse equation. The purpose is to find out how far this single geometric primitive can go as a language engine, trained from scratch on a laptop.

The project keeps its failures on record on purpose (`docs/FINDINGS.md`, `COLLAPSE_ENGINE_VERDICT.md`). What you read in the docs is what was actually measured.

---

## 2. How vector collapse works

Everything in this repo runs on one update rule. A hidden state `h` (a vector, e.g. 256-d) is pulled toward a target vector `W` (a "well" — one learned vector per word, character, or class):

```
h ← h − strength · (1 − cos(h, W)) · norm(h − W)
```

Read it in three parts:

- `cos(h, W)` — how aligned the state already is with the well. If `h` already points at `W`, the factor `(1 − cos)` is ~0 and nothing moves. The further away, the harder the pull.
- `norm(h − W)` — the unit direction *away* from the well. Subtracting it moves `h` *toward* the well.
- `strength` — a single learned scalar (passed through a sigmoid) controlling step size. After each step the norm of `h` is clipped at 10 so it can't blow up.

So each well is a **gravity well / point-attractor**: states near it fall in, states far away get pulled harder. Applied over a sequence of words, the state traces a **trajectory** — and crucially that trajectory is *order-sensitive*: collapsing through "dog bites man" ends somewhere completely different from "man bites dog" (measured: reordered words give mean-pool cosine 1.000 vs collapse cosine 0.072). Mean-pooling can't tell those apart; collapse can. That's the whole reason it earns its place.

Two properties worth knowing:

- **O(1) per step.** One dot product, one subtraction. No attention matrix over the sequence. That's why the small models run in single-digit milliseconds on CPU.
- **It was proven to add signal, not just pass it through.** In the supervised SNLI classifier, an ablation showed the collapse layers contribute **+4.86%** accuracy over a plain linear head (68.9% vs 64.1%). The trained engine visibly warps space into three attractor basins around the Entailment / Neutral / Contradiction anchors — see `docs/COLLAPSE_VISUALIZATION.md` and `docs/COLLAPSE_STRUCTURE_REPORT.md` for the flow-field plots.

Where a neural net *is* used, its job is only to **pick** the next well; the collapse **executes** the move. Brain proposes, geometry disposes.

---

## 3. How Premise One worked (the SNLI premise generator)

`chat/premise_from_hyp.py` — the first generator, and the proof that collapse can *type*.

Normal NLI is discriminative: given (premise, hypothesis) → predict a label. Premise One **inverts** it: given the **hypothesis** and the **label**, it must *type the premise back, one word at a time*. To do that well it can't memorize a class index — it has to learn what "entailment", "neutral", "contradiction" actually *do* to a sentence, generatively.

The mechanism, step by step:

1. **Thought.** Mean-pool the hypothesis word wells into one vector, concatenate the label embedding, pass through one linear layer: `z = think([hyp ; label])`. The writing state starts at `h = z`.
2. **Pick.** At each step a small MLP ("the brain") builds a query from `[h ; z]`, and the next word is whichever well has the highest `cos(query, well) / temp`. Punished with cross-entropy against the true premise word.
3. **Collapse.** `h ← collapse(h, well[true word])` — the state physically walks onto the word just typed (teacher forcing during training), and the next pick happens from there.
4. Repeat until `<eos>`.

The best checkpoint (`model/premise_from_hyp_align_53.pt`) adds **one** lightweight cross-attention step: at each pick, a label-conditioned attention over the individual hypothesis word wells feeds aligned content into the query, so each premise word is chosen with per-word hypothesis correspondence in hand. (Correct framing: no self-attention anywhere; one cross-attention step. A truly attention-free variant exists too.)

**Free classifier trick:** at eval, score the real premise under all 3 labels and pick the label that explains it best — `argmax_label P(premise | hypothesis, label)`. That generative classifier scores ~53% on SNLI dev; the discriminative sibling reaches 74.7% dev / 74.4% test with alignment (no pretrained embeddings anywhere).

The whole thing is **5.98M parameters**, trained only on SNLI's ~550k captions, ~20k whole-word vocab learned from random init, ~5 ms per reply on CPU. It's coherent only inside SNLI's caption world — that limitation is the point: it shows exactly how much language a tiny collapse engine can extract from a narrow corpus.

Try it: `cd chat && python3 chat_premise.py`

---

## 4. How the chat model works (what's being trained now)

`chat/` Part 1 — "the chat-brain." A ladder of models, every rung the same collapse engine, culminating in `chat_reply.py`: a model that reads a conversation and types a reply.

### The ladder

| rung | file | what it learns |
|---|---|---|
| CHAR | `char_typer_all.py` | a well per character (~2,000, ENTER included); types raw lines back |
| WORD | `chat_typer.py` | a well per word; types sentences back word-by-word |
| minting bridge | `chat_typer_live.py` + `char_fingerprint.py` | unseen word → a well minted from its *spelling*, no retraining |
| SENTENCE (read) | `chat_reply.py :: read()` | conversation read as a collapse trajectory — order-aware |
| CONTEXT (data) | `prep_chat_context.py` / `prep_dailydialog.py` | last turns tagged `<you>`/`<me>` → next reply |
| REASONING | `chat_reply.py` | attend over context + own typed words, pick a well, collapse |

### One forward pass of `chat_reply.py` = READ → THINK → WRITE

- **READ.** Collapse the state through every context word in order. Each intermediate state is kept — the context becomes a *trajectory* of states, not one pooled vector. (A distilled "fast reader" replays this walk as a single causal conv when `model/fast_reader.pt` exists.)
- **THINK.** `z = linear(final read state)` — the seed thought. Writing starts at `h = z`.
- **WRITE.** Per typed word: normalize `h`, attend over `[context trajectory + the states of its own already-typed words]` (memory *grows* as it types — self-attend), build a query from `[h ; z ; attended ctx]` through the brain MLP, pick the nearest well via cosine, then `h ← collapse(h, well[word])`. Loss is per-word cross-entropy against the real reply.

The char layer is read-side only: an out-of-vocabulary context word gets a deterministic well minted from its letters, so the reader never collapses through `<unk>` mush — but the writer can only *say* trained words. Spelling earns a word the right to be heard; only training earns the right to be said.

Training levers (all default-on, each with an off switch): vocab cut to words seen ≥2×, meaning-weighted loss (rare content words punish harder than "the"), positional scaffold annealed to zero, scheduled sampling, sampled softmax (512 negatives), early stopping on dev NLL.

### Where DailyDialog fits — the two-stage recipe

The personal ChatGPT-export data (~18k pairs) is too small to teach general conversational fluency on its own. So training is now two-stage (`prep_dailydialog.py`):

```bash
# make the data: ~11k human everyday dialogues → same ctx<TAB>reply shape
python3 prep_dailydialog.py                      # → data/dd_context.tsv

# stage A — general fluency pretrain on DailyDialog
#   --extra-vocab keeps the personal words in the vocab for stage B
python3 chat_reply.py --data data/dd_context.tsv \
    --extra-vocab data/chat_context.tsv --ckpt model/chat_reply_general.pt

# stage B — fine-tune on personal chats, warm from stage A
python3 chat_reply.py --data data/chat_context.tsv \
    --resume model/chat_reply_general.pt --pos-anneal 0 --lr 5e-4

# talk to it
python3 chat_reply.py --chat
```

DailyDialog turns become the same shape as the personal data: previous turns tagged so the replier is always `<me>`, one tokenizer (`prep_chat_context.clean`) shared everywhere. Stage A teaches *how conversation flows*; stage B teaches *this person's voice*, on the same shared wells.

### Data rule (single source of truth)

Personal data trains only from the raw ChatGPT export walked by the canonical path (`current_node` → parent chain), implemented once in `prep_chat_context.canonical_turns()`. Fix once, fixed everywhere. The old flattened export lost half the text and is retired.

---

## 5. How to get set up, step by step

```bash
git clone <repo-url> && cd livnium
pip3 install torch pyarrow          # core math needs nothing; chat/ needs these
python3 -m pytest -q                # verify the proven core on your machine
```

Reading order:

1. Root `README.md` — the project, the honest results, the one-paragraph math.
2. This file.
3. `docs/COMPONENTS.md` — plain-language tour of every file.
4. `chat/README.md` — the chat-brain ladder + Premise One, with current results and known-stale items.
5. `docs/COLLAPSE_VISUALIZATION.md` — see the attractor basins with your own eyes.
6. Then code, in this order: `chat/premise_from_hyp.py` (smallest complete generator) → `chat/chat_typer.py` → `chat/chat_reply.py` (the full READ/THINK/WRITE model).

Quickest hands-on demo (no training needed, checkpoint ships in `chat/model/`):

```bash
cd chat && python3 chat_premise.py
you > is the girl standing
ai  > a girl in a pink shirt standing in a doorway.   [neutral]
```

---

## 6. Honest limits, stated once

This is not a general chatbot and does not "understand." It has no pretrained knowledge, no subwords on the write side, and is coherent only in the domains it was trained on (SNLI captions; now everyday dialogue + personal chats). Grammar emerges fast because grammar is local and regular — that is fluency, not understanding. Every claim above has a measured number behind it; when a claim died, it's marked falsified in `docs/FINDINGS.md` rather than deleted. That's the culture of the repo — keep it that way.
