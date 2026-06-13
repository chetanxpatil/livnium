# Livnium

**A small, honest piece of mathematics that began with a Rubik's cube.**

Livnium is a *conserved geometric state space*: a way of placing symbols onto the
cells of a cube and moving them around without ever losing information. It started
as one person's year-long obsession, went through a phase of big claims, and then
got put through honest testing. This repository keeps **both** the parts that
survived and the parts that didn't — because that is what makes it trustworthy.

If you read nothing else, read this:

> The mathematics is real and proven. The early claim that it "beats AI" is not —
> it was tested directly and it sits at chance on a fair benchmark. The repo keeps
> the truth on both counts, on purpose.

---

## The one-paragraph version

Take a 3×3×3 cube — 27 little cells. Give each cell one symbol from a 27-character
alphabet (`0` and the letters `a`–`z`), where `0` is the center, the "Om." Every
cell gets a coordinate, and a **weight** based on how exposed it is: a hidden core
cell weighs 0, a face-center 9, an edge 18, a corner 27. Add up all the weights and
you get a number that **never changes**, no matter how you turn the cube — there are
exactly 24 ways to rotate it, and every one of them just shuffles the cells while
conserving the total. That conserved, reversible, perfectly self-consistent little
universe is **Livnium Core.** It is clean, it is correct, and it is the keeper.

---

## What's actually here

### 1. A real, tested mathematical core
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

### 2. The honest results
When the question "does this geometry help a machine *understand language*?" was
tested fairly, the answer was **no** — and that's written down in full in
[`results/RESULTS.md`](results/RESULTS.md) and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).
The short version is below.

### 3. The story
[`docs/ORIGINS.md`](docs/ORIGINS.md) is the real history — where every idea came
from, reconstructed from the original notes. [`docs/FINDINGS.md`](docs/FINDINGS.md)
is a complete inventory of every claim made over the year, each marked **proven**,
**standard**, **partial**, or **falsified**.

---

## The honest results, in plain language

The test was **natural language inference**: given two sentences, decide whether the
second *follows from*, *contradicts*, or is *unrelated to* the first. We measured
Livnium against boring, dumb baselines on the same data — because nothing counts as
a win until it beats the dumbest thing that works.

Two ways of feeding text into the cube were tried:

| What we tried | How it scored (SNLI) | What it means |
|---|---|---|
| **Letters → cube** (each letter is a symbol) | **43%** | Above random (33%), but far below word-counting (~59%). It only sees spelling, not words. |
| **Words → cube** (each word gets its own cell) | **60%** | Jumps up to match plain word-counting — because now it's *doing* word-counting, dressed in geometry. |
| The cube's *shape* alone (no words) | **38%** | Basically random. The geometry by itself carries almost no meaning. |

And on **ANLI** — a harder benchmark built specifically so you can't cheat with
word-counting — Livnium scores at chance (~33%), like every word-counting method.

**The lesson, stated once:** the cube is a beautiful, lossless *container*. But
understanding meaning requires *throwing information away* — keeping what matters,
discarding spelling and surface noise. A system that can never forget can never
abstract. So the accuracy was never going to come from the geometry; it comes from
the words you put in it, and once you're counting words you're not reasoning. This
isn't a failure of the math — it's the **shape** of the tool, and knowing it is what
makes the tool usable. (The full reasoning is in [`docs/LIMITS.md`](docs/LIMITS.md).)

There was **one** genuine bright spot — a compression result where "collapse text to
what you already know" beat gzip and showed ~78% of ordinary text is predictable
"dark matter." That one points somewhere real; see
[`docs/COMPRESSION_NOTE.md`](docs/COMPRESSION_NOTE.md).

---

## Try it in 30 seconds

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

## Repository map

```
livnium/
├── README.md              ← you are here
├── LICENSE                ← source-available; all rights reserved
├── livnium_core/          ← the proven math (pure Python, no dependencies)
├── tests/                 ← the test suite (run: pytest)
├── docs/
│   ├── ORIGINS.md         ← the real story, from the first day
│   ├── FINDINGS.md        ← every claim, marked proven / standard / partial / falsified
│   ├── FORMULAS.md        ← the formal definitions and proofs
│   ├── LIMITS.md          ← what it provably cannot do, and why
│   ├── BENCHMARKS.md      ← the honest NLI test
│   ├── REARRANGEMENT.md   ← face turns and the Rubik's group
│   ├── COMPRESSION_NOTE.md← the one positive result worth chasing
│   └── ML_LADDER.md       ← the path forward, learned the hard way
├── results/
│   ├── RESULTS.md         ← measured numbers: Livnium vs baselines, with kill-tests
│   └── *.py               ← the exact, reproducible experiment scripts
└── visualizer/index.html  ← interactive 3D view of the lattice
```

---

## What Livnium is *not*

Not a quantum computer. Not a replacement for neural networks. Not a new physics.
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
