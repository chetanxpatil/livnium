# Model Card — Livnium Collapse-NLI v1

A small, label-supervised natural-language-inference (NLI) model from the
experimental `models/collapse-nli/` line. Given two sentences — a **premise** and a
**hypothesis** — it predicts whether the hypothesis is **entailed by**,
**contradicts**, or is **neutral** toward the premise.

> **Status: experimental.** This is not the proven `livnium_core`. It is an honest
> research result, kept with its limits stated plainly.

---

## TL;DR

| | |
|---|---|
| Task | SNLI 3-way NLI (entailment / neutral / contradiction) |
| **SNLI test accuracy** | **68.92%** (dev-selected checkpoint, read once on test) |
| Best dev | 69.62% @ epoch 20 |
| Clears | majority 34.3, bag-of-words 59.4, GloVe-avg 60.7, **hyp-only artifact 61.5** |
| Does **not** reach | attention/BERT-class NLI (~86–91%) |
| Size | ~12.9M params (50k-word × 256-d embedding table + collapse engine) |
| Honest caveat | A post-hoc frozen-embedding probe scores 64.06% (linear head), 68.92% (collapse), 70.13% (MLP). Because the embeddings were optimized for collapse, a matched end-to-end multi-seed ablation is **still required** — see Limitations. |

The number that matters: clearing the **hypothesis-only artifact (61.5)** by ~7
points is the bar the project's earlier static-embedding runs *failed* to clear
(see `docs/results/nli.md`). So this is a genuine step up — for a model with **no
word order and no attention**.

---

## How it's made (the pipeline, in plain language)

The model has two learnable parts: a **word-embedding table** and a small
**collapse engine** with three label "anchors." Training is end-to-end on SNLI.

1. **Vocabulary.** Read all SNLI training pairs, lowercase, split on whitespace,
   keep the top 50,000 words (`<pad>` and `<unk>` are reserved). ~549k labeled
   examples.
2. **Embed.** Each word maps to a learned 256-dimensional vector
   (`CollapseEmbeddingModel.emb`, an `nn.Embedding`).
3. **Pool.** Mean-pool the word vectors of the premise → `u`, and of the
   hypothesis → `v` (padding ignored). Each sentence becomes one 256-d vector.
4. **Pair.** Form `pair = u - v` — the hypothesis *relative to* the premise. (A
   difference vector is a classic, decent NLI feature.)
5. **Collapse.** Run `pair` through the **collapse engine** (`VectorCollapseEngine`,
   `num_layers=4`): a residual update that "warps" the vector toward the geometry
   of three learned anchors **E / N / C** (strengths 0.1 / 0.1 / 0.05). The last
   MLP layer is zero-initialized so the block starts as the identity and the
   thermodynamic force shapes the dynamics first.
6. **Classify.** Normalize the collapsed vector and take its dot product with the
   three unit anchors → 3 logits. A temperature (`temp=0.1`) softmax gives the
   label probabilities.
7. **Loss.** Cross-entropy against the gold label **plus** an explicit
   anchor-separation penalty (`lambda_sep=1.0`) that pushes the E/N/C anchors
   apart so they don't collapse onto each other.
8. **Optimize.** Adam, `lr=3e-4`, batch 512, 20 epochs on Apple-Silicon MPS
   (~6 min/epoch). Both the embeddings *and* the engine are trained.
9. **Tripwire.** `--verify-engine-trained` asserts, after each epoch, that the
   engine is actually moving and the E/N/C anchors stay separated (pairwise cosine
   under 0.50). It passed every epoch — the anchors ended near-orthogonal.

### Training command that produced it

```bash
cd collapse_retrain
python3 train_collapse_embeddings.py \
    --task nli \
    --nli-path ../data/snli_1.0_train.jsonl \
    --output-dir model_nli_v1 \
    --epochs 20 --max-lines 0 \
    --verify-engine-trained --sep-threshold 0.5 \
    --lambda-sep 1.0 --temp 0.1 --device auto
```

(SNLI corpus: https://nlp.stanford.edu/projects/snli/snli_1.0.zip — put
`snli_1.0_train.jsonl` in `data/`.)

---

## How to use it (manual)

The trained weights are **not** in the repo (checkpoints are gitignored — large
and regenerable). Train with the command above, or use a checkpoint you already
have under `model_nli_v1/`.

### 1. Classify one pair

```bash
cd collapse_retrain
python3 predict.py \
    --ckpt model_nli_v1/nli_epoch20.pt \
    --premise "A man is playing a guitar on stage." \
    --hypothesis "A person is performing music."
```

Output:

```
Prediction: ENTAILMENT
  entailment      71.30%
  neutral         18.04%
  contradiction   10.66%
```

(Illustrative numbers — yours will vary with the checkpoint.)

### 2. Score a whole dataset (dev or test)

```bash
# all epochs on dev, auto-pick the best:
python3 eval_nli.py --ckpt-dir model_nli_v1 --data ../data/snli_1.0_dev.jsonl

# the dev-chosen checkpoint, once, on test:
python3 eval_nli.py --ckpt model_nli_v1/nli_epoch20.pt --data ../data/snli_1.0_test.jsonl
```

`eval_nli.py` prints a per-epoch accuracy table, the best checkpoint, and a
confusion matrix with the SNLI reference bars.

### Requirements

```bash
pip install "livnium-core[experimental]"   # torch + tqdm (+ numpy)
```

### Checkpoint format (`nli_epoch{N}.pt`)

A dict with: `embeddings` (the `[vocab, dim]` table), `vocab`
(`idx2word`/`pad_idx`/`unk_idx`), `dim`, `collapse_engine` (state dict),
`collapse_config` (`num_layers`, strengths). `predict.py` / `eval_nli.py` rebuild
the model from exactly these fields.

---

## Results in detail

SNLI test confusion matrix (epoch 20, rows = true, cols = predicted):

```
true\pred       entailment   neutral   contradiction
entailment           2425       443             500
neutral               626      1958             635
contradiction         384       465            2388
```

All three classes are predicted (no class collapse); **neutral** is the weakest,
which is normal — neutral is the hardest NLI class and is confused with both
others. Dev climbed monotonically 64.1 → 69.6 with no overfitting droop, and the
train/test gap (~78% vs ~69%) is healthy.

---

## Limitations & honest notes

- **Not attention-class.** Mean-pooling discards word order and cross-sentence
  interaction, which is exactly what caps SNLI around the high 60s. ~69% is a
  respectable *non-attention* baseline, not state of the art.
- **Geometry vs. embeddings is only partially ablated.** A post-hoc probe on
  the *frozen* trained embeddings scores **64.06%** with a plain linear head,
  **68.92%** with the collapse engine and **70.13%** with an MLP head. That
  favors collapse over a linear readout, but the embeddings were originally
  optimized *for* collapse, so the comparison is biased toward it — and the MLP
  outscoring collapse shows the geometry is not uniquely responsible. A matched
  end-to-end multi-seed ablation (train each head from scratch, same budget,
  several seeds) is still required. Do **not** claim the probe proves the
  geometry is the cause.
- **Out-of-vocabulary.** Words outside the 50k SNLI vocab become `<unk>`; the
  model is not robust to very different domains (e.g. code, or technical text).
- **SNLI only.** No ANLI / MultiNLI / adversarial numbers yet; expect those to be
  much harder (the project's own history shows ANLI sits near chance for static
  representations).

---

## Provenance

Produced by `models/collapse-nli/train_collapse_embeddings.py --task nli`. Reproduce
end-to-end with the command above. See `docs/collapse/COLLAPSE_ENGINE_VERDICT.md` for the
engine's design history and `docs/core/COMPONENTS.md` for how this fits the wider repo.
