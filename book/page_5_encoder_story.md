# Page 5 — The Encoder Story

## Chapter 1: Why the Encoder Matters So Much

The collapse engine is the heart of Livnium. But the engine only works as well as the vector it starts from.

`h0 = v_h - v_p`

This one line is the bottleneck. If `v_p` and `v_h` are poor representations of their sentences — if the encoder cannot distinguish "A man is sleeping" from "A man is awake" — then `h0` carries no useful signal. The most sophisticated attractor dynamics in the world cannot recover meaning from noise.

The entire history of Livnium's accuracy improvements is a history of getting better `h0` vectors. The dynamics stayed roughly the same. The encoder is what changed.

---

## Chapter 2: Stage 1 — The Legacy Encoder (56%)

**File:** `tasks/snli/encoding_snli.py` → `SNLIEncoder`
**Flag:** `--encoder-type legacy`

```python
class SNLIEncoder(nn.Module):
    def __init__(self, vocab_size, dim=256, pad_idx=0):
        self.text_encoder = TextEncoder(vocab_size, dim, pad_idx)
        # TextEncoder = embedding table + mean pool

    def build_initial_state(self, prem_ids, hyp_ids):
        v_p = mean_pool(embed(prem_ids))
        v_h = mean_pool(embed(hyp_ids))
        h0 = v_h - v_p
        return h0, v_p, v_h
```

The encoder is just a vocabulary lookup table plus mean pooling. Every word gets a 256-dim vector. The sentence vector is the average of its word vectors. The vocabulary is built from scratch from the SNLI training set.

**What this cannot do:**
- Distinguish word *order* — "A bites B" and "B bites A" produce the same mean-pool vector
- Handle negation — "The man is sleeping" and "The man is NOT sleeping" produce nearly identical vectors (negation barely moves the mean)
- Understand context — "bank" in "river bank" vs "bank account" get the same vector

**Result: ~56% accuracy.** This is barely above random (random = 33%, human-level = 89%). The dynamics cannot compensate for the garbage in `h0`.

**Why it was fast:** No pretrained knowledge. A ~50K × 256 embedding table is small. Training just that table plus the collapse engine and head is computationally cheap.

---

## Chapter 3: Stage 2 — The Livnium Pretrained Encoder (76%)

**File:** `embed/text_encoder.py` → `PretrainedTextEncoder`
**Loaded from:** `pretrained/collapse4/embeddings_final.pt`
**Flag:** `--encoder-type pretrained --embed-ckpt /path/to/embeddings_final.pt`

This is where the architecture becomes something genuinely different from standard NLP.

The embeddings in `embeddings_final.pt` were not trained by predicting the next word (like GPT), or by masking and reconstructing (like BERT). They were trained by Livnium's *own* collapse dynamics on the embedding space.

The `embed/` folder is a mini-Livnium:

```
embed/
├── text_encoder.py       ← PretrainedTextEncoder
├── collapse_engine.py    ← A simpler VectorCollapseEngine
└── basin_field.py        ← A simpler BasinField
```

During embedding pretraining, the word vectors were shaped so that *semantically similar words naturally cluster into the right basins* when the collapse dynamics run over them. Entailment-related words ended up in one region, contradiction-related words in another, neutral-related words in between.

When these pre-shaped vectors are loaded into the SNLI training run (with `freeze=False`), two things happen:
1. The embeddings start from a geometrically meaningful position, not random noise
2. They continue to adapt during SNLI training — the 13M embedding parameters tune alongside the collapse engine

**The key difference from Stage 1:** Not just "better starting point." The embedding space itself was pre-shaped to be compatible with attractor dynamics. The vectors arrived already oriented in a geometry that the collapse engine can work with.

**Result: 76.32% accuracy.** A 20-point jump. Not from changing the dynamics — from fixing `h0`.

**Why it was slow:** The 13M+ embedding parameters were all being updated every batch, plus the collapse engine and head. Significantly more computation than Stage 1.

---

## Chapter 4: The Ceiling of Bag-of-Words

76.32% is the wall that the bag-of-words architecture runs into, regardless of how well the vectors are pretrained. The ceiling is structural, not parametric.

Here's why. The mean-pool operation *destroys information*:

**Problem 1: Order blindness**
```
"The dog bit the man."   →  mean({the, dog, bit, the, man})
"The man bit the dog."   →  mean({the, man, bit, the, dog})
```
These two sentences have the same words and thus the same mean-pool vector. But their NLI relationships to any hypothesis are completely different.

**Problem 2: Negation collapse**
```
"The man is sleeping."       →  mean({the, man, is, sleeping})
"The man is not sleeping."   →  mean({the, man, is, not, sleeping})
```
The word "not" has a vector, but it's just averaged in with everything else. Its effect on the sentence meaning is 1/N of the total, where N is the sentence length. For a 10-word sentence, negation is diluted to a 10% signal.

**Problem 3: No cross-sentence interaction**
The premise and hypothesis are encoded independently. The encoder never sees them together. Whether "A person" refers to the same person as "The man" in the hypothesis is invisible to a mean-pool encoder — it has to be inferred entirely by the collapse dynamics from the difference vector.

The dynamics work hard to compensate for these limitations. With good pretrained embeddings they get 76%. But they cannot fully recover information that was destroyed by mean pooling.

---

## Chapter 5: Stage 3 — BERT via llama.cpp (target: 84-87%)

**File:** `tasks/snli/encoding_snli.py` → `LlamaCppBERTSNLIEncoder`
**Model:** `bert-base-uncased-Q8_0.gguf`
**Flag:** `--encoder-type llamacpp --llamacpp-model /path/to/bert.gguf`

BERT (Bidirectional Encoder Representations from Transformers) solves all three problems above.

**BERT's architecture in one sentence:** Every token in the sentence attends to every other token through 12 layers of self-attention, producing a context-dependent representation for each token. The final `[CLS]` token representation is used as the sentence embedding.

**Problem 1 solved — Order:**
The attention mechanism sees all pairwise relationships between tokens. "dog bit man" produces a completely different attention pattern than "man bit dog." The CLS token encodes the directionality of the sentence.

**Problem 2 solved — Negation:**
"not" doesn't get averaged away. In the attention layers, "not" directly modifies the tokens near it — "is not sleeping" produces a very different representation of "sleeping" than "is sleeping" because "not" attends to and modifies "sleeping" directly.

**Problem 3 partially helped:**
Even though BERT still encodes premise and hypothesis separately (without cross-sentence interaction), the richer representations mean that `h0 = v_h - v_p` captures much more semantic content. The difference between BERT-level sentence vectors is far more informative than the difference between mean-pool vectors.

**Why llama.cpp instead of HuggingFace transformers:**
The `BERTSNLIEncoder` class uses HuggingFace's full PyTorch BERT, which is excellent for quality but heavyweight for repeated inference. llama.cpp runs a quantised (Q8_0) version of the same model with:
- 8-bit weights instead of 32-bit (118 MB vs ~440 MB)
- Metal GPU acceleration on Mac
- ~3-5x faster inference per sentence

With the pre-computed embedding cache (`precompute_embeddings.py`), even this speed difference disappears — BERT runs exactly once per sentence for the entire training, and every subsequent epoch uses the cached 768-dim vectors.

---

## Chapter 6: What Changes (and What Doesn't) with BERT

**Changes:**
- `h0` is now a 768-dimensional vector instead of 256
- `h0` encodes order, negation, and context
- The collapse engine and head automatically resize to dim=768 (detected from the GGUF probe)
- Training is faster with the cache (no BERT in the loop)

**Doesn't change:**
- The collapse dynamics — same physics equations, same basin field
- The training loop — same loss functions, same optimiser
- The `h0 = v_h - v_p` formula
- The BERT weights themselves — still frozen. Only the collapse engine and SNLIHead train.

The experiment is clean: **everything identical, only the quality of h0 changes.** If accuracy jumps from 76% to 84%+, that directly quantifies how much information the mean-pool encoder was destroying.

---

## Chapter 7: The Full Accuracy Journey

| Stage | Encoder | Trainable | Accuracy | Why |
|---|---|---|---|---|
| 1 | Legacy (random BoW) | Embeddings + engine + head | ~56% | No semantic knowledge, order-blind |
| 2 | Pretrained Livnium BoW | Embeddings + engine + head | 76.32% | Semantically shaped vectors, but still mean-pool |
| 3 | Frozen BERT (HuggingFace) | Engine + head only | ~61% | BERT frozen — collapse engine can't reshape BERT's geometry |
| 4 | Joint BERT (bi-encoder) | BERT + engine + head | **82.06%** | BERT fine-tunes alongside collapse dynamics (5 epochs) |
| 5 | Joint BERT (cross-encoder) | BERT + engine + head | in progress | [CLS] premise [SEP] hypothesis — fixes role-reversal failures |
| 6? | Livnium-native encoder | Small encoder + engine + head | target: 84%+ | Ditch BERT entirely — see Page 6 |

Stage 4 is a potential next step: keep BERT frozen but add a small trainable linear layer (768 → 256) that projects BERT vectors into the Livnium embedding space before computing `h0`. This would let the model learn which dimensions of the BERT space are most useful for the collapse dynamics, without needing to fine-tune BERT itself.

---

## Chapter 8: The Precompute System

**File:** `model/precompute_embeddings.py`

Because BERT is frozen, its outputs for any given sentence are always identical. Running BERT 3 times (once per epoch) on the same 550K sentences is pure waste.

The precompute script:
1. Collects all unique sentences from train + dev (approximately 300-350K unique strings)
2. Runs each through BERT exactly once
3. Saves a `{sentence: 768-dim vector}` dictionary to a `.pt` file

During training with `--embed-cache`, `build_initial_state` becomes a dictionary lookup:

```python
v_p = cache[premise]    # 0.001ms
v_h = cache[hypothesis] # 0.001ms
h0  = v_h - v_p + noise
```

Instead of:
```python
v_p = bert(premise)    # 11ms
v_h = bert(hypothesis) # 11ms
h0  = v_h - v_p + noise
```

The pre-compute takes ~30-60 minutes once. After that, every training run uses the cache and each epoch takes ~5 minutes instead of ~5 hours. The dynamics, loss, and accuracy are mathematically identical to running BERT live.

---

## Chapter 9: What to Watch For in the BERT Run

When the BERT training completes, compare three numbers carefully:

**1. Overall accuracy vs 76.32%**
Anything above 80% confirms that the encoder was the bottleneck. Anything above 84% means BERT is providing substantial gains.

**2. Per-class accuracy (Neutral)**
The hardest class. The pretrained BoW model likely struggles most on neutral — pairs where the premise and hypothesis are related but not in a clear E or C direction. BERT should show the biggest gain here because neutral requires understanding the *nature* of the relationship, not just the presence of shared concepts.

**3. The confusion matrix**
Look at E→C and C→E errors (the model calling Entailment when it's Contradiction and vice versa). With BoW, these errors were high because "A bites B" and "B bites A" produce similar mean-pool vectors. With BERT, these should drop sharply.

---

*End of Book v1.0*

---

## Appendix: Quick Reference

| File | Role |
|---|---|
| `physics_laws.py` | divergence, tension, barrier constant |
| `core/vector_collapse_engine.py` | The main attractor dynamics engine |
| `core/basin_field.py` | Dynamic multi-basin management |
| `tasks/snli/encoding_snli.py` | All encoder types |
| `tasks/snli/head_snli.py` | Classification head |
| `model/train.py` | Training loop, all flags |
| `model/eval.py` | Test evaluation |
| `model/precompute_embeddings.py` | One-time BERT cache builder |
| `embed/text_encoder.py` | Pretrained BoW encoder (loads .pt checkpoint) |
| `embed/collapse_engine.py` | Mini-Livnium used during embedding pretraining |

| Flag | What it does |
|---|---|
| `--encoder-type llamacpp` | Use llama.cpp BERT encoder |
| `--llamacpp-model` | Path to BERT GGUF file |
| `--embed-cache` | Path to precomputed embedding cache |
| `--barrier` | Equilibrium cosine (default 0.38) |
| `--strength-neutral-boost` | Extra neutral pull at E-C boundary |
| `--rot-rank` | Enable rotational dynamics (0=off, 8=start) |
| `--lock-threshold` | Enable locking zones (0=off, 0.5=start) |
| `--strength-null` | Enable virtual null endpoint (0=off, 0.03=start) |
| `--adaptive-metric` | Enable learned diagonal metric |
| `--disable-dynamic-basins` | Use only 3 fixed anchors |
