# Livnium Core — Limits of the System

Understanding what a system *cannot* do is what lets you use it for what it
*can*. These limits are not failures; they are the shape of the tool. Several
are **structural** (provable ceilings, not bugs), some are **empirical**
(measured), and the last is a **category** point.

---

## 1. Structural limits (provable — these will never change with more work)

### 1.1 Reversibility ⇒ it cannot abstract
The core is a **bijection / reversible permutation system**: no information is
ever destroyed. But abstraction *is* destruction of the irrelevant — keep the
meaning, throw away spelling, word order, surface noise. A system that cannot
forget cannot generalize. **Livnium's defining virtue (losslessness) is exactly
what forbids it from learning representations.** These are opposite requirements;
you cannot have both in the same mechanism.

### 1.2 Conservation by construction ⇒ inert toward any external task
`ΣSW` is invariant no matter what you encode — Shakespeare or random noise both
conserve. A quantity that is identical for signal and noise carries **zero bits**
about the signal. So the conservation laws, though true, **constrain nothing
about any outside problem.** They describe the container, not the contents.

### 1.3 Position comes from token identity, not from data
Coordinates are assigned by the base-27 encoding of the *symbol/spelling*, never
learned from how words are used. So **distance in the representation = spelling
distance, not meaning distance.** "cat"/"car" land close; "not"/"no" can land
far. Nothing in the construction ties geometry to semantics, so the geometry
cannot reflect meaning. (This is why "not" — the word that flips entailment — is
invisible to it.)

### 1.4 The dynamics are a small, discrete group
The allowed transformations are exactly **24 rotations** (a finite permutation
group ≅ S₄), plus wreath-product composition in the hierarchy. There is no
continuous deformation, no learnable warp, no parameters. Expressive power is
bounded by the group: you can reorient the world, not reshape it.

### 1.5 There is no learning signal in the core
The core has **no objective function, no gradient, no training** — it is a static
algebraic structure. Anything that "learns" (the nova embeddings, a logistic
head) is *external* to Livnium and is doing the actual work. The core itself
cannot improve from data because there is nothing in it that responds to data.

### 1.6 Polarity is an underived overlay
Everything else falls out of the lattice. **Polarity (`cos θ`) does not** — it
introduces a continuous "motion vector" with no structural reason to exist on a
discrete 24-move lattice. It is assumed, not derived, and it is the piece that
empirically failed (§2). Treat it as speculative, not core.

---

## 2. Empirical limits (measured, not argued)

- On **artifact-free NLI (ANLI)** the geometry scores **~29–34% = chance**, below
  the bag-of-words bar (~37–41%). Increasing data 3k→30k did not move it.
- The **SNLI ~76%** that once looked like success was the *dataset* leaking the
  label through hypothesis wording — not reasoning by the geometry. Strip the
  leak (ANLI) and the apparent skill vanishes.
- A linear classifier on Livnium features **cannot exceed what the features
  encode**, and the features encode spelling/geometry, not inference.

See `BENCHMARKS.md` for the full table and controls.

---

## 3. Where the same limits become strengths (correct uses)

The properties that kill it for ML are virtues elsewhere:

- **Lossless, reversible encoding / addressing / canonical hashing** — when you
  *want* to preserve everything exactly and invert it later.
- **Exact base-27 ⇄ decimal ⇄ binary** numerals with correct carry.
- **A clean teaching / visualization object** for group actions, conservation
  laws, and positional number systems (see the world visualizer).
- **Combinatorial enumeration** — closed-form class counts, the 24-rotation
  group, wreath-product scaling.

Use it where reversibility and exact conservation are the *goal*, not where lossy
abstraction is required.

---

## 4. The category limit (the one-sentence version)

> Livnium Core is a **data structure / algebra**, not a **model**. It stores,
> addresses, permutes, and conserves with perfect fidelity — and for the same
> reason it cannot compress, generalize, learn, or represent meaning. Asking it
> to do machine reasoning is asking a ruler to weigh things: not a flaw in the
> ruler, just the wrong tool for that measurement.

Knowing this boundary is the whole value of the audit. The math inside the
boundary is real and yours. The boundary itself is fixed.
