---
language:
- en
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0/
library_name: pytorch
pipeline_tag: feature-extraction
tags:
- word-embeddings
- from-scratch
- attention-free
- collapse
- vector-collapse
- livnium
datasets:
- wikipedia
metrics:
- spearmanr
---

# Noun-Collapse — Wikipedia word embeddings from pure vector collapse

Word embeddings learned with **nothing but a collapse dynamical system** — no
MLP, attention, transformer block or separate learned output matrix, and no
pretrained embeddings. The entire model
is **one 256-d well per word, a start state, and two scalars** (pull strength,
readout temperature). ~25.6M numbers total; ~99% of them are the well table.

Meaning is **read out of the geometry**: the same wells that pull a state during
encoding are the vectors you look up as embeddings.

> Part of [Livnium](https://github.com/chetanxpatil/livnium). Honest by design —
> the number below is reported against real baselines, not a chance floor.

## How it was trained

CBOW-style fill-in-the-blank, executed by the collapse engine instead of a net:
for every noun occurrence, a state is collapsed through the noun's ±5-word
**ordered** context and must end up pointing at the missing noun (sampled-softmax
cross-entropy over nouns). The update law, applied once per context word:

```
h ← h − strength · (1 − cos(h, W)) · norm(h − W)
```

- **Data:** English Wikipedia, ~5M lines (~7.5% of the corpus).
- **Signal:** 94.75M occurrences of WordNet noun-eligible tokens
  (lexicon-matched, not contextually POS-tagged), one streaming pass.
- **Compute:** ~3.2 h on an Apple-silicon MacBook (MPS).
- **Nouns:** WordNet noun lexicon; 100k-word context vocab, 23,758 noun targets.

Because the context is read as an ordered collapse trajectory (not a bag),
word order is physically encoded — unlike CBOW/PPMI.

## Quality — SimLex-999 (similarity, not association)

| model | data | SimLex-999 ρ (nouns) |
|---|---|---|
| **this model** | 7.5% of Wikipedia, noun-only | **0.362** (662/666 pairs) |
| word2vec / GloVe (published) | full Wikipedia+Gigaword | ~0.37–0.44 |
| PPMI+SVD (reference) | full corpus | ~0.38 |

Near the published word2vec/GloVe band on a fraction of the data, with no MLP,
attention, transformer block or separate learned output matrix. ρ is computed
with tie-aware `scipy.stats.spearmanr` (0.3616). Published baselines use
different corpora/preprocessing — a matched-corpus comparison is still pending.

## Speed (M-series MacBook)

- Embed one already-tokenized 10-word context: **0.23 ms on CPU** (encoder
  latency, not end-to-end application latency).
- Bulk: **2.3M words/s** on MPS at batch 1024.
- Nearest-noun query vs 23,758 wells: **0.48 ms**.

## Usage

```bash
pip install torch huggingface_hub
hf download chetanxpatil/noun-collapse --local-dir noun-collapse
cd noun-collapse
```

```python
from modeling_noun_collapse import NounCollapse

m = NounCollapse.from_pretrained("noun_collapse_pure.pt")

m.vector("physics")            # 256-d unit embedding of a word
m.similarity("cat", "dog")     # cosine similarity
m.neighbors("india", k=8)      # nearest nouns
m.encode(["a cat sat on the mat"])   # collapse a sentence -> one state vector
```

Example neighbors:

```
cat     -> tabby dog pet felis mouse stray feline
physics -> chemistry mathematics astronomy quantum mechanics astrophysics
war     -> vietnam outbreak world cold ii boer veteran
india   -> gujarat pakistan nepal sikkim delhi bombay punjab bengal
```

## Files

- `noun_collapse_pure.pt` — the checkpoint (`wells`, `stoi`, `noun_ids`,
  `start`, `strength`, `temp`, `config`).
- `modeling_noun_collapse.py` — standalone loader/encoder (torch only).
- `config.json` — architecture metadata.

## Limitations (read before citing)

- **Similarity, not logic.** It learns that *cat* and *animal* are close, not
  that a cat *is* an animal. No facts, no hierarchy, no negation.
- **Frequency-bound.** Common nouns have sharp neighborhoods; rare nouns stay
  near their random init.
- **7.5% of Wikipedia, single pass, no LR schedule** — headroom remains; this
  is the honest first result, not a tuned ceiling.
- Whole-word vocab (no subwords): out-of-vocab words have no vector.

## License

PolyForm Noncommercial 1.0.0 — free for individuals, students, researchers,
nonprofits. Commercial use requires a paid license. See the Livnium repo.
