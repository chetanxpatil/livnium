# Livnium

[![CI](https://github.com/chetanxpatil/livnium/actions/workflows/ci.yml/badge.svg)](https://github.com/chetanxpatil/livnium/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm NC](https://img.shields.io/badge/license-PolyForm--NC--1.0.0-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![HF Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-noun--collapse-yellow.svg)](https://huggingface.co/chetanxpatil/noun-collapse)

**A small, honest piece of mathematics that began with a Rubik's cube.**

This repository is three things, kept deliberately separate:

```
Livnium Core       — proven cube mathematics
Collapse Engine    — attractor dynamics
Research Models    — nouns, SNLI, generation and vision
```

It started as one person's year-long obsession, went through a phase of big
claims, and then got put through honest testing. The repo keeps **both** the
parts that survived and the parts that didn't — because that is what makes it
trustworthy.

If you read nothing else, read this:

> The mathematics is real and proven. The early claim that it "beats AI" is not —
> it was tested directly and it sits at chance on a fair benchmark. The repo keeps
> the truth on both counts, on purpose.

---

## Part 1 — Livnium Core: proven cube mathematics

Take a 3×3×3 cube — 27 little cells. Give each cell one symbol from a 27-character
alphabet (`0` and the letters `a`–`z`), where `0` is the center, the "Om." Every
cell gets a coordinate, and a **weight** based on how exposed it is: a hidden core
cell weighs 0, a face-center 9, an edge 18, a corner 27. Add up all the weights and
you get a number that **never changes**, no matter how you turn the cube — there are
exactly 24 ways to rotate it, and every one of them just shuffles the cells while
conserving the total. That conserved, reversible, perfectly self-consistent little
universe is **Livnium Core.** It is clean, it is correct, and it is the keeper.

A tiny Python package, `livnium_core`, with no dependencies and a full test suite.
Everything in it is proven and checked:

- **Base-27 codec** — a number system over the alphabet `0,a..z`. Lossless and
  reversible: `z + a = a0` (because 26 + 1 = 27, and it rolls over exactly).
- **Exposure classes & weight** — every cell has an exposure `f ∈ {0,1,2,3}` and a
  weight `SW = 9f`. There are closed-form formulas for how many cells of each kind
  exist at any size, and for the conserved total `ΣSW`.
- **The 24 rotations** — the rotation group of the cube (mathematically, S₄). Each
  rotation is reversible and preserves every quantity above. This is the
  conservation law, made concrete.
- **Hierarchy** — cubes can nest inside cubes, with the bookkeeping adding up
  cleanly across levels.

### Try it in 30 seconds

```bash
git clone https://github.com/chetanxpatil/livnium.git
cd livnium
python -m pytest -q          # the whole proven core, verified on your machine
```

```python
from livnium_core import int_to_base27, base27_to_int, symbolic_weight_total, rotation_group

int_to_base27(27)            # 'a0'   — the codec rolls over at 27
base27_to_int('ch')          # 89
symbolic_weight_total(7)     # 2646   — the conserved total for a 7×7×7 cube
len(rotation_group())        # 24     — the rotations of a cube
```

There is also a **3D visualizer** — open [`visualizer/index.html`](visualizer/index.html)
in any browser to see the lattice, the exposure classes, and the rotations.

---

## Part 2 — Collapse Engine: attractor dynamics

> Livnium treats language as **motion through a learned geometric landscape**.
> Each word is a well that softly bends the state passing through it; the pull
> strength is learned and weak, so no single word overwrites the path, and the
> final state is a compromise shaped by the whole sequence. (Measured:
> reordering a sentence's words moves the endpoint to cosine 0.07 between the
> two readings, where plain averaging gives 1.00 — order is physically encoded
> in the path.) Inference is motion through fixed geometry; **learning is
> geometry being carved by where motion missed.**

The family shares one idea — pull the state toward a well, harder when
misaligned — but the repo now contains **four distinct collapse rules**, and
they are not interchangeable:

| Variant | Rule | Character |
|---|---|---|
| **Livnium v1** (chord-directed) | `h ← h − s · (1 − cos(h, W)) · norm(h − W)` | Hand-designed; **non-conservative** (no exact global scalar potential — see `experiment/findings.md`) |
| **Livnium v2** (exact energy gradient) | `h ← h + s · (W − cos(h, W)·ĥ)/‖h‖` | Exact gradient of `V(h) = −cos(h, W)`; conservative |
| **Direct collapse** | closed-form step (`vector_collapse`, `mode="direct_collapse"`) | Closed-form approximation, no iteration |
| **MLP collapse** | learned residual + away-force (`mode="mlp_collapse"`) | Learned variant; not a fixed physical law |

On stability: the noun dynamics are measurable and attractor-directed, but the
current chord force (v1) is **non-conservative** — it is not the gradient of a
global scalar potential, so earlier statements about a proven Lyapunov energy
for that exact rule are withdrawn. The empirical energy-descent measurements in
`chat/LYAPUNOV_TEST.md` remain valid as *empirical* observations of a Lyapunov
candidate on sampled trajectories. An exact cosine-gradient variant (v2) with a
true closed-form potential is implemented separately in
`experiment/pure_reply.py`.

The standalone, configurable engine lives in [`vector_collapse/`](vector_collapse/)
(installable; requires the `[experimental]` extra). For the full mechanics,
reading order, and how to run it in two minutes, see
[`docs/START_HERE.md`](docs/START_HERE.md).

---

## Part 3 — Research Models: nouns, SNLI, generation and vision

These are experiments, not the proven core. Checkpoints are published
externally — see [`CHECKPOINTS.md`](CHECKPOINTS.md) for URLs and SHA-256 hashes.

### The honest NLI results, in plain language

The test was **natural language inference**: given two sentences, decide whether the
second *follows from*, *contradicts*, or is *unrelated to* the first. We measured
Livnium against boring, dumb baselines on the same data — because nothing counts as
a win until it beats the dumbest thing that works.

| What we tried | How it scored (SNLI) | What it means |
|---|---|---|
| **Letters → cube** (each letter is a symbol) | **43%** | Above random (33%), but far below word-counting (~59%). It only sees spelling, not words. |
| **Words → cube** (each word gets its own cell) | **60%** | Jumps up to match plain word-counting — because now it's *doing* word-counting, dressed in geometry. |
| The cube's *shape* alone (no words) | **38%** | Basically random. The geometry by itself carries almost no meaning. |
| **Supervised Collapse Model** (learned embeddings + 4-layer attractor collapse) | **68.9%** | Clears the hypothesis-only artifact (61.5%) and GloVe avg (60.7%). A post-hoc frozen-embedding probe favors collapse over a linear head (+4.86%), but a matched end-to-end ablation is still pending. |

And on **ANLI** — a harder benchmark built specifically so you can't cheat with
word-counting — Livnium scores at chance (~33%), like every word-counting method.

*Note on the Supervised Collapse model:* By training word embeddings end-to-end with a 4-layer vector collapse engine (`VectorCollapseEngine`) that warps difference vectors toward three learned point-attractors (Entailment, Neutral, Contradiction), the model reaches **68.87% test accuracy** on SNLI. On the ablation: a post-hoc frozen-embedding probe scores 64.06% with a linear head, 68.92% with collapse and 70.13% with an MLP. Because the embeddings were originally optimized for collapse, a matched end-to-end multi-seed ablation is still required before attributing the gain to the collapse dynamics. On speed: single-pair inference runs in **0.33 ms** on CPU and over **215,000 pairs/sec** on Apple Silicon GPU (MPS) — these figures measure already-tokenized encoder latency, not end-to-end application latency, and the collapse step is constant in sequence length after $O(L)$ pooling, for fixed dimension, anchors and collapse steps. See [results/RESULTS.md](results/RESULTS.md) and [docs/COLLAPSE_VISUALIZATION.md](docs/COLLAPSE_VISUALIZATION.md) for details.

### Word embeddings from pure collapse

The v1 engine, pointed at Wikipedia (94.75M occurrences of WordNet
noun-eligible tokens — lexicon-matched, not contextually POS-tagged — ~7.5% of
the corpus), learns real word meaning with *no MLP, attention, transformer
block or separate learned output matrix* — just one well per word and two
scalars. It scores **SimLex-999 ρ = 0.362** (tie-aware `scipy.stats.spearmanr`),
near the published word2vec/GloVe band (~0.37–0.44) though on different
corpora — a matched-corpus baseline is still pending — and embeds one
already-tokenized context in 0.23 ms on CPU (encoder latency, not end-to-end
application latency). The model is live on the Hub:
[🤗 chetanxpatil/noun-collapse](https://huggingface.co/chetanxpatil/noun-collapse).
Details in [`chat/README.md`](chat/README.md).

### Generation and vision

- **Premise generator / chat-brain** (`chat/`): a 5.98M-param on-device sequence
  model — type a hypothesis, it types back a premise (~53% gold-label match,
  chance 33%). See [`docs/START_HERE.md`](docs/START_HERE.md).
- **Vision collapse** (`vision/`): every attractor is a pixel; images collapse
  through their own pixels. Smoke-tested at 64², not yet trained at full scale.

### The lesson, stated once

The cube is a beautiful, lossless *container*. But understanding meaning
requires *throwing information away* — keeping what matters, discarding
spelling and surface noise. A system that can never forget can never abstract.
So the accuracy was never going to come from the geometry; it comes from the
words you put in it, and once you're counting words you're not reasoning. This
isn't a failure of the math — it's the **shape** of the tool, and knowing it is
what makes the tool usable. (The full reasoning is in
[`docs/LIMITS.md`](docs/LIMITS.md).)

There was **one** genuine bright spot — a compression result where "collapse text to
what you already know" beat gzip and showed ~78% of ordinary text is predictable
"dark matter." That one points somewhere real; see
[`docs/COMPRESSION_NOTE.md`](docs/COMPRESSION_NOTE.md).

The full story: [`docs/ORIGINS.md`](docs/ORIGINS.md) is the real history.
[`docs/FINDINGS.md`](docs/FINDINGS.md) is a complete inventory of every claim
made over the year, each marked **proven**, **standard**, **partial**, or
**falsified**. The honest test results live in
[`results/RESULTS.md`](results/RESULTS.md) and
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## Repository map

```
livnium/
├── README.md              ← you are here
├── LICENSE                ← source-available; all rights reserved
├── CHECKPOINTS.md         ← checkpoint manifest: URLs + SHA-256
├── livnium_core/          ← Part 1: the proven math (pure Python, no dependencies)
├── tests/                 ← the test suite (run: pytest)
├── vector_collapse/       ← Part 2: the standalone collapse engine
├── chat/                  ← Part 3: noun embeddings, premise generator, chat-brain
├── collapse_retrain/      ← Part 3: supervised collapse NLI (experimental)
├── cortex_v2/             ← Part 3: experimental MPS simulator + collapse prototype
├── vision/                ← Part 3: vision collapse (research)
├── docs/
│   ├── COMPONENTS.md      ← plain-language tour of every file (start here)
│   ├── START_HERE.md      ← collapse engine mechanics + reading order
│   ├── ORIGINS.md         ← the real story, from the first day
│   ├── FINDINGS.md        ← every claim, marked proven / standard / partial / falsified
│   ├── FORMULAS.md        ← the formal definitions and proofs
│   ├── LIMITS.md          ← what it provably cannot do, and why
│   ├── BENCHMARKS.md      ← the honest NLI test
│   ├── REARRANGEMENT.md   ← face turns and the Rubik's group
│   ├── COMPRESSION_NOTE.md← the one positive result worth chasing
│   ├── ML_LADDER.md       ← the path forward, learned the hard way
│   ├── COLLAPSE_VISUALIZATION.md ← 2D projection plots of the attractor flow field
│   └── COLLAPSE_STRUCTURE_REPORT.md ← formal dynamical analysis of anchors and basins
├── results/
│   ├── RESULTS.md         ← measured numbers: Livnium vs baselines, with kill-tests
│   └── *.py               ← the exact, reproducible experiment scripts
├── experiment/            ← v2 exact-gradient collapse + findings
├── COLLAPSE_ENGINE_VERDICT.md  ← forensic post-mortem of the cortex_v2 experiment
└── visualizer/index.html  ← interactive 3D view of the lattice
```

> **Heads-up on the experimental folders.** Everything under Part 3 is a
> research prototype, not part of the proven core. They depend on `numpy`
> and `torch` and are kept for transparency — see `COLLAPSE_ENGINE_VERDICT.md`
> for an honest account of what worked and what didn't.

---

## What Livnium is *not*

Not a replacement for neural networks. Not a new physics. Not magic.
It is a clean piece of combinatorial geometry with a reversible codec — and an
unusually honest record of one person's attempt to find out how far an idea could
go. Some of those answers were "not as far as I hoped," and they're here too.

---

## License

**Free for noncommercial use. Companies pay for commercial use.**

Livnium is released under the [PolyForm Noncommercial License](LICENSE) — free for
individuals, students, researchers, hobbyists, nonprofits, schools, and government.
Use it, study it, modify it, build on it.

If you're a **company** and want to use Livnium **commercially** — in a product, a
service, or internal operations — you need a paid commercial license. That's the
model: free for the curious, paid for the commercial. See [`COMMERCIAL.md`](COMMERCIAL.md).

*Livnium by Chetan Patil.*
