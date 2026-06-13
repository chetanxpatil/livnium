# Compression & "Dark Matter" — a real positive result

*The one experiment in this whole project that beat its baseline and pointed at
something true and buildable. Validated on SNLI hypotheses (~768k chars).*

## The idea (Chetan's, in his words)
> Collapse incoming info to "what we know" until you hit a variety. Most of it is
> predictable mass — call it dark matter. The varieties are the bright info.

This is **compression = prediction = learning** (Shannon / MDL / predictive coding).

## Result 1 — the dark-matter split (lossless)
Collapsing text to an order-3 context model ("what we know"):

```
raw                         : 8.00 bits/char
collapse to what we know    : 1.72 bits/char
gzip (general baseline)     : 2.30 bits/char     <- context modeling beat it*

DARK MATTER (predictable)   : 78.5%   (collapses away, carries ~no info)
BRIGHT INFO (varieties)     : 21.5%   (the real surprise)
```
\*Honest caveat: the n-gram bits don't include model-storage cost; gzip is
self-contained. The robust finding is the **78.5% redundancy split**, and that
context modeling is the right family (PPM / arithmetic coding / language models).

## Result 2 — the lossy step ("dark matter plays a role")
Train/test split, then prune the rare one-off varieties (noise):

```
full model        : 2.052 bits/char held-out,  21,338 entries
drop seen <5x     : 2.058 bits/char,  85.9% size
drop seen <10x    : 2.074 bits/char,  78.1% size
```
Throwing away ~22% of the model (the rare varieties) costs almost nothing on
held-out data → those varieties were low-value noise. **The representation can
be lossy without losing meaning.** This is the lossy-learned principle Livnium's
reversible core structurally could not do (see LIMITS.md §1.1).

## Why this matters
This is the first thread that is **lossy AND learned** — the two requirements
for a real representation (see the ML ladder). And it is exactly the training
objective of language models: next-token prediction = collapse-to-known + encode
the surprise. Chetan's intuition converged, independently, on the core principle
of modern ML.

## Next step
Build a tiny **next-character language model** (n-gram → small neural net) — which
*is* this "collapse to known, keep the varieties" engine, learned from data.
That is rung 2 of `ML_LADDER.md`, and the natural continuation of this result.
