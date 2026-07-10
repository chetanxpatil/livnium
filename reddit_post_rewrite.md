# I swapped the neural net in a word-embedding model for a physics-style attractor system. No MLP, attention, transformer block or separate learned output matrix. Gets ρ=0.36 on SimLex-999 using 7.5% of Wikipedia.

This is one piece of a larger thing I've been building (a "vector collapse" engine). Word embeddings were just a clean way to check whether the mechanism learns meaning on its own. Real numbers below, plus a list of what it can't do so we don't have to argue about it in the comments.

## The idea

word2vec/GloVe and everything after lean on a learned network or a big matrix factorization. I wanted to see how far you get with only a dynamical system. The whole model is:

- one 256-d vector per word (a "well")
- a start state
- two scalars: pull strength and readout temperature

That's it. ~25.6M numbers, ~99% of which is just the word table.

## How it reads a context

One update rule, applied once per context word, pulls a moving state toward that word's well:

`h ← h − strength · (1 − cos(h, W)) · norm(h − W)`

Strength is learned and comes out weak (~0.11), so no single word drags the state onto itself. The final position is a compromise shaped by the whole ordered context. Because it's a trajectory and not a bag, word order actually matters — reverse a sentence and the endpoint moves to cosine 0.07 vs the original (mean-pooling gives you 1.00). You read meaning straight out of the geometry: the wells that pull the state are the same vectors you look up as embeddings. No separate decoder.

## Training

CBOW-style fill-in-the-blank, but run by the collapse dynamics instead of a network. For every noun occurrence, collapse a state through its ±5-word context and make the endpoint point at the missing noun (sampled-softmax cross-entropy over nouns). Gradient descent only reshapes the wells.

- Data: English Wikipedia, ~5M lines (~7.5% of the corpus, ~300M tokens)
- Signal: 94.75M occurrences of WordNet noun-eligible tokens (lexicon-matched, not POS-tagged), single streaming pass
- Vocab: 100k context words, 23,758 noun targets (WordNet)
- Compute: ~3.2 hrs on an M-series MacBook (MPS). No cluster.

## Quality — SimLex-999

(similarity, not association, so coffee/cup scores low)

| model | data | ρ (nouns) |
|---|---|---|
| pure collapse (this) | 7.5% Wikipedia, noun-only | 0.362 (662/666 pairs) |
| word2vec / GloVe (published) | full Wikipedia+Gigaword | ~0.37–0.44 |
| PPMI+SVD (reference) | full corpus | ~0.38 |

So it lands in the word2vec/GloVe range on a fraction of the data with no network in the loop.

## Nearest nouns by cosine

```
physics   -> chemistry mathematics astronomy quantum mechanics astrophysics
chemistry -> physics biology biochemistry nobel organic pharmacology
india     -> mumbai gujarat nepal sikkim delhi bombay punjab bengal
france    -> belgium vichy britain italy marseille spain germany
cat       -> tabby dog pet felis mouse stray feline
apple     -> macintosh ipod blackberry android pc cherry laptop
```

Nothing there was hand-specified.

## What it can't do

- It's similarity, not logic. It learns that cat and animal are close, not that a cat *is* an animal. No facts, no hierarchy, no negation.
- One vector per word means the dominant sense wins. "apple" collapsed to the company because Wikipedia talks about the company more than the fruit. No sense disambiguation.
- Frequency-bound. Common nouns get sharp neighborhoods; rare ones barely move from their random init.
- 7.5% of Wikipedia, single pass, fixed LR, no schedule. This is a first number, not a tuned ceiling.
- Whole-word vocab, no subwords, so OOV words have no vector.
- The apples-to-apples baseline (PPMI+SVD on the same 5M lines) is still running. Comparing to published word2vec is suggestive, not a controlled win.

**Why I think it's worth a look:** it's a fully inspectable alternative to attention for the "compress a sequence into meaning" job — a contraction toward learned point-attractors, with an empirical energy you can actually measure (the state descended toward the wells on 100% of 12k sampled steps; the force is non-conservative, so that's a measured property, not a proven Lyapunov energy). This is the embedding-layer version; the same engine also does NLI and generation in the repo.

Code, model card, benchmark, loader: https://github.com/chetanxpatil/livnium/tree/main/chat
Model on the Hub (loads in 3 lines of torch): https://huggingface.co/chetanxpatil/noun-collapse

Two things I'd actually like input on:

1. Has anyone gotten Hopfield/point-attractor dynamics to beat a plain PMI factorization on intrinsic similarity at matched data, or does the count-based method always win there?
2. Cheapest honest way to add polysemy (multi-sense wells) without bolting on a full network and losing the "it's just geometry" part?
