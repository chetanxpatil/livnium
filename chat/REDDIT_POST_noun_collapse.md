**Title:** I replaced the neural network in a word-embedding model with a physics-style attractor system — no MLP, attention, transformer block or separate learned output matrix. It hits SimLex-999 ρ=0.36 on 7.5% of Wikipedia. Honest writeup.

---

Posting the honest version. This is one instance of a general mechanism I've been building (a "vector collapse" engine); word embeddings just turned out to be a clean way to test whether the mechanism actually learns meaning on its own. Real numbers, and a "what it is NOT" section at the bottom so the comments don't have to.

**The idea.** Standard embedding models (word2vec/GloVe) and everything since lean on either a learned network or a big matrix factorization. I wanted to know how far you get with *only a dynamical system*. So the entire model is:

- one 256-d vector per word (a "well"),
- a start state,
- two scalars: a pull *strength* and a readout *temperature*.

That's it. **No MLP, attention, transformer block or separate learned output matrix — and no pretrained anything.** ~25.6M numbers total, ~99% of which are just the word table.

**How it "reads."** One update rule, applied once per context word, pulls a moving state toward that word's well:

```
h  ←  h − strength · (1 − cos(h, W)) · norm(h − W)
```

The strength is learned and ends up *weak* (~0.11), so no single word yanks the state onto itself — the final position is a compromise shaped by the whole ordered context. Because it's a trajectory, not a bag, word order is physically encoded (reversing a sentence moves the endpoint to cosine 0.07 vs the original, where mean-pooling gives 1.00). Meaning is then **read straight out of the geometry**: the same wells that pull the state are the vectors you look up as embeddings. There is no separate decoder.

**Training.** CBOW-style fill-in-the-blank, executed by the collapse dynamics instead of a network: for every noun occurrence, collapse a state through its ±5-word context and make the endpoint point at the missing noun (sampled-softmax cross-entropy over nouns). Gradient descent only reshapes the wells.

- **Data:** English Wikipedia, ~5M lines (**~7.5% of the corpus**, ~300M tokens).
- **Signal:** **94.75M occurrences of WordNet noun-eligible tokens** (lexicon-matched, not contextually POS-tagged), single streaming pass.
- **Vocab:** 100k context words, 23,758 noun targets (WordNet noun lexicon).
- **Compute:** **~3.2 hours on an M-series MacBook** (MPS). No cluster.

**Quality — SimLex-999 (similarity, not association; coffee/cup scores LOW):**

| model | data | SimLex-999 ρ (nouns) |
|---|---|---:|
| **pure collapse (this)** | 7.5% of Wikipedia, noun-only | **0.362** (662/666 pairs) |
| word2vec / GloVe (published) | full Wikipedia+Gigaword, billions of tokens | ~0.37–0.44 |
| PPMI+SVD (reference) | full corpus | ~0.38 |

So it lands *inside* the classic word2vec/GloVe band, on a fraction of the data, with no neural network in the loop.

**What the neighborhoods look like (nearest nouns by cosine):**

```
physics   -> chemistry mathematics astronomy quantum mechanics astrophysics
chemistry -> physics biology biochemistry nobel organic pharmacology
pakistan  -> karachi punjab lahore peshawar bangladesh india afghanistan
france    -> belgium vichy britain italy marseille spain germany
cat       -> tabby dog pet felis mouse stray feline
apple     -> macintosh ipod blackberry android pc cherry laptop
```

Synonyms, hypernyms, sibling terms, geographic manifolds — none of it hand-specified.

**Speed (M-series MacBook, `noun_bench.py`):**

| batch | CPU words/s | MPS words/s |
|---:|---:|---:|
| 1 | 43,863 | 5,174 |
| 1024 | 1,464,201 | **2,311,023** |

- Embed one already-tokenized 10-word context: **0.23 ms on CPU** (encoder latency, not end-to-end application latency) — real-time, no GPU needed.
- Bulk: **2.3M words/s** on GPU at batch 1024.
- Nearest-noun query vs 23,758 wells: **0.48 ms**.

Same crossover as everything tiny: CPU wins single items (the collapse math is so small the run is launch-bound), GPU runs away in bulk past ~batch 256. The 10-step sequential walk amortizes to ~4 µs/window under batching.

**What it is NOT** (saving you the comment):
- **Similarity, not logic.** It learns *cat* and *animal* are close, not that a cat *is* an animal. No facts, no hierarchy, no negation. It's a meaning *substrate*, not a reasoner.
- **One vector per word = dominant sense wins.** "apple" collapsed to the *company* (macintosh, ipod, android) because Wikipedia talks about the company more than the fruit — "cherry" is the only fruit neighbor. No sense disambiguation.
- **Frequency-bound.** Common nouns have sharp neighborhoods; rare ones barely leave their random init.
- **7.5% of Wikipedia, single pass, fixed LR, no schedule.** This is the honest first number, not a tuned ceiling — there's obvious headroom.
- Whole-word vocab, no subwords: OOV words have no vector.
- The apples-to-apples baseline (PPMI+SVD on the *same* 5M lines) is still running; comparing to *published* word2vec is suggestive, not a controlled win.

**Why I think it's interesting anyway:** it's a fully inspectable alternative to attention for the "compress a sequence into meaning" job — a contraction toward learned point-attractors, with an empirical directional energy you can actually measure (the state descended toward the wells on 100% of 12k sampled steps; the chord force is non-conservative, so this is a measured property, not a proven Lyapunov energy). It's the embedding-layer instance; the same engine also does NLI and text generation in the repo.

**Code, model card, benchmark, and the standalone loader:** https://github.com/chetanxpatil/livnium/tree/main/chat
**Model on the Hub (loads in 3 lines of torch):** https://huggingface.co/chetanxpatil/noun-collapse

Two things I'd genuinely like input on: (1) has anyone gotten point-attractor / Hopfield-style dynamics to beat a plain PMI factorization on intrinsic similarity at *matched* data, or does the count-based method always win there? (2) cheapest honest way to add polysemy (multi-sense wells) without bolting on a full network and losing the "it's just geometry" property?
