# Components — what every part of Livnium is, and is not

This is a plain-language tour of the whole repository. It is deliberately honest:
it says what each piece *is good for*, and what it *is not*, so the project reads
as a real system plus honest experiments — not as hype.

The repo splits into three layers:

| Layer | Folders | What it means |
|---|---|---|
| **1. The real core** | `livnium_core/` | A reversible symbolic-geometric system. Proven. |
| **2. Honest evidence** | `tests/`, `results/` | Proof the core works; honest ML benchmark. |
| **3. Experiments** | `cortex_v2/`, `collapse_retrain/` | Research prototypes; the big claim is **not** proven yet. |
| **The written record** | `docs/` | History, formulas, limits, findings. |

One-line mental model:

> **Livnium Core = symbolic geometry + conservation + reversible computation.**
> The ML experiments are evidence about where the boundary is — not proof of intelligence.

---

## 1. `livnium_core/` — the real, reusable system

Pure Python, **zero dependencies**. This is the part you can safely call "Livnium
Core." It is not AI by itself; it is a symbolic-geometric system: numbers, symbols,
cube cells, rotations, a layer language, hierarchy, and conservation rules.

### `base27.py` — the symbol/number codec

Converts losslessly between symbols and integers. The alphabet is `0 a b c … z`
where `0` = value 0 (the core / "Om") and `a…z` = 1…26. Because it is base 27:

```text
z + a = a0        # 26 + 1 = 27, which is written "a0" in base 27
```

**Good for:** encoding symbols, reversible IDs, base-27 arithmetic, the Livnium
alphabet logic. **Not:** language understanding — it is a clean codec.

### `lattice.py` — the cube geometry and conservation law

Defines the cells of an odd-sized cube (3×3×3, 5×5×5, 7×7×7, …). Each cell has an
*exposure* `f ∈ {0,1,2,3}` (how many coordinates touch the outer boundary) and a
*symbolic weight* `SW = 9f`:

```text
center cell      → exposure 0 → SW 0
face-center cell → exposure 1 → SW 9
edge cell        → exposure 2 → SW 18
corner cell      → exposure 3 → SW 27
```

No matter how you rotate the cube, the count of each class — and the total
symbolic weight — stays the same. That is the conservation law.

**Good for:** proving cube structure, counting cell classes, checking invariants.
**Not:** it does not learn or classify; it gives the geometry and the ledger.

### `rotations.py` — the 24 whole-cube rotations

Builds every valid rigid rotation of the cube: exactly **24** integer matrix
transformations, forming a group isomorphic to S₄. Each one keeps centers as
centers, faces as faces, edges as edges, corners as corners — so symbolic weight is
preserved. The cube can change orientation, but it cannot break its structure.

**Good for:** orientation states, cube symmetry, reversible transforms, testing
conservation. **Not:** random movement — it is a structure-preserving group.

### `moves.py` — Rubik-style face turns

Where `rotations.py` moves the whole body, `moves.py` rearranges internal positions:
a face turn rotates **one layer only** and leaves the rest fixed. This is a richer,
more dynamic move set.

**Good for:** Rubik-like rearrangement, permutation experiments, testing whether
structure survives under richer moves. **Not:** intelligence — it is a controlled
permutation system.

### `layer_language.py` — a small symbolic language

A fully-defined symbolic algebra grounded in base-27. It parses expressions and
computes deterministic results. The notation is **not** arithmetic on letters — it
uses *shapes* with a *depth* and two operators:

```text
shapes : o (hollow, sign -1)   *  (filled, sign +1)
depth  : an integer after the shape, optionally with ^   e.g.  o^2,  *9
ops    : |  the LAYER operator (output is the function F(left → right))
         ~  the RELATIONSHIP operator (pure relationship, no layer function)

example : o^2 | *9
```

It computes **structure** (relationships between the symbols' own depths and signs),
deterministically — a perfect reference frame. It does **not** encode meaning:
`F(cat → mom)` would transform the codes, not the fact that a cat loves a mom.

**Good for:** symbolic algebra, a toy formal language, exact symbolic demos.
**Not:** natural language.

### `hierarchy.py` — multi-scale Livnium

Lets cubes nest inside cubes (a small lattice inside each cell of a bigger one —
the old "3 inside 5 inside 7" idea). The conservation ledger stays additive across
scales:

```text
SW_global = N³ · SW(M) + SW(N)
```

**Good for:** nested systems, multi-scale geometry, conservation across levels.
**Not:** it does not automatically produce emergence — it gives the math for nested
conserved structure.

### `__init__.py` — the public API

The package doorway: it defines what Livnium Core officially exposes. The real
public names are:

```python
from livnium_core import (
    int_to_base27, base27_to_int, base27_to_binary, binary_to_base27, ALPHABET,
    exposure, SW, class_counts, symbolic_weight_total,
    rotation_group, ROT_X, ROT_Y, ROT_Z,
    capacity, global_ledger, wreath_group_order,
    face_permutation, apply_sequence, solved_state, FACES,
    ll_parse, ll_evaluate,
)
```

Note: the codec functions are `int_to_base27` / `base27_to_int` (not `encode` /
`decode`), and the lattice is **functional** (`exposure`, `SW`, …) — there is no
`Lattice` class.

---

## 2. `tests/` — the proof layer

Six files (`test_base27`, `test_lattice`, `test_rotations`, `test_moves`,
`test_layer_language`, `test_hierarchy`) verify that the system is not hand-wavy:
base-27 round-trips, lattice counts are correct, rotations preserve symbolic
weight, moves are permutation-safe, the layer language is deterministic, and the
hierarchy formula adds up.

**31 passing tests** is what lets the project honestly say *the core is proven* —
not proven as "AI," but proven as a reversible symbolic-geometric system. If a
later change breaks conservation, CI catches it.

---

## 3. `results/` — the honest ML benchmark

This folder answers one question — *does Livnium geometry alone help a machine
understand language?* — and the honest answer was **no, not by itself.** That is
not a failure; it is a clean boundary.

The task is **NLI** (natural language inference): given a premise and a hypothesis,
decide entailment / contradiction / neutral.

### `rung2_livnium.py` — char-level geometric encoder

Flow: `character → base-27 value → lattice cell → fixed geometric feature vector →
classifier`. Measured SNLI accuracy: **43.2%** — above chance, but well below
word-counting (~59%). It only sees spelling, not words.

### `rung2_livnium_word.py` — word-level encoder

Gives each word its own cell instead of each character. Score jumps to **~60%
(59.9%)** — but the geometry-*only* signal (lattice shape, no word identity) is
**38.0% ≈ chance**. So the accuracy comes from *word identity occupying cells*, i.e.
word-counting dressed in geometry, not from the geometry itself.

### `rung2_lib.py` — shared benchmark harness

Data loading, baselines (including the GloVe baseline), feature extraction, and the
train/eval split. It keeps the comparison fair — Livnium is tested against normal
baselines, not in a friendly sandbox.

### `RESULTS.md` / `README.md`

The official benchmark records — what happened, not what was wished. The correct
claim: *Livnium Core is real as a symbolic-geometric system; Livnium geometry alone
did not solve NLI.* On ANLI (the artifact-free task) everything sits at chance.

---

## 4. `cortex_v2/` — experiment: state-vector simulator

Research prototype, not the keeper. Depends on `numpy`.

### `mps.py` — Matrix Product State simulator

Represents large state vectors efficiently by splitting them into smaller tensors.
A built-in "governor" truncates small components during the SVD split, keeping the
state from exploding in size.

**Good for:** tensor-network / state-vector simulations, state-compression
experiments, testing whether Livnium geometry can guide pruning. **Not:** exotic
hardware — it is a classical simulator running on an ordinary machine.

### `cortex_v2/lattice.py` — reduced 3×3×3 lattice + SU(2) bridge

A smaller, optimized lattice model. A key proven fact: a rotation's "alpha signal"
is independent of which symbols sit where, because rotations permute the lattice
bijectively. The SU(2) "lift" maps a cube orientation into a spinor-like
representation.

**Good for:** rotation-to-state experiments, the geometry-to-MPS bridge. **Not:**
proof that Livnium is exotic physics or that it understands language.

### `test_regressions.py` — safety tests for known bugs

Guards against two bugs that already happened: long-range CNOT norm collapse, and
cross-process nondeterminism. Experimental code can break silently; these stop old
bugs from returning.

---

## 5. `collapse_retrain/` — experiment: embedding / collapse trainer

The ML experiment area. Depends on `torch`. **Not proven, and currently has no
trained checkpoint.**

### `train_collapse_embeddings.py`

Trains word embeddings on WikiText-103 with a Livnium-style energy warp — roughly
"word2vec-style training + extra collapse energy." It asks: *can a Livnium-style
energy shape embeddings into a better semantic space?* Status: **untrained, no
checkpoint, no result** — so no claim should be made from it yet.

### `vector_collapse.py` — the collapse engine

Pulls a vector state toward one of three learned anchors (Entailment /
Contradiction / Neutral): the state moves toward a basin, and the label is read from
where it lands.

What the record actually shows: a single ablation found the collapse engine
contributed **+4 to +9 points over a dummy** on SNLI (3 checkpoints), marked
🟡 *partial* in `FINDINGS.md` — and that was in the older Nova/BERT-features setting
(82.2%), separate from the pure-geometry-at-chance result. So treat it as *weakly
supported, not proven* — an inductive-bias experiment, not an established mechanism.

### `basin_field.py` — dynamic basins

Manages regions of attraction that can spawn, route, update, and prune during
training, instead of fixed class centers. All that routing and pruning is why this
path is CPU-heavy.

**Good for:** adaptive label regions, dynamic clustering, routing experiments.

### `text_encoder_collapse.py` — inference-time encoder

Loads a trained checkpoint and encodes text through the collapse model. Since no
checkpoint exists yet, this is a ready doorway rather than a working final model.

---

## 6. `docs/` — the written record

This folder keeps the project from drifting into mythology.

- **`ORIGINS.md`** — the real history (Rubik's cube, 26 outer nodes + core,
  rotation, base-27, lattice, collapse experiments), without pretending every early
  idea was already correct.
- **`FINDINGS.md`** — every claim labelled *proven / standard / partial / falsified*.
  The anti-hype file.
- **`FORMULAS.md`** — formal definitions and proofs (base-27, exposure, symbolic
  weight, conserved total, rotation preservation, hierarchy formula).
- **`LIMITS.md`** — what Livnium provably cannot do: reversible systems don't
  naturally abstract; learning often needs compression/forgetting; geometry alone
  does not create semantics.
- **`BENCHMARKS.md`** — the NLI benchmark: what was tested, the baselines, the
  scores, and the valid conclusion.
- **`REARRANGEMENT.md`** — face turns vs rigid rotations (closer to the original
  "strings criss-cross around the core" idea).
- **`COMPRESSION_NOTE.md`** — the one positive direction worth chasing: ~78% of
  ordinary text is predictable "dark matter" that collapses away.
- **`ML_LADDER.md`** — the disciplined roadmap: climb rung by rung, kill your
  positives, believe a result only after it survives.

---

## 7. Top-level files

- **`README.md`** — the front page: what Livnium is, what's proven, what failed, how
  to run tests and reproduce benchmarks.
- **`LICENSE`** — PolyForm Noncommercial: free to use/study/modify non-commercially.
- **`COMMERCIAL.md`** — paid terms for commercial use.
- **`COLLAPSE_ENGINE_VERDICT.md`** — the post-mortem of the collapse/cortex
  experiment: what was tried, what looked promising, what failed, what remains
  useful.
- **`pyproject.toml`** — package config (name, version, build, ruff/black/pytest
  config).
- **`requirements.txt`** — dependencies for **reproducing `results/`** only:
  `numpy, pandas, scikit-learn, scipy, pyarrow, gensim`. (The core needs nothing;
  `pytest` is in `pyproject`'s optional `[test]` extra; the experimental
  `collapse_retrain/` additionally needs `torch` + `tqdm`, and `cortex_v2/` needs
  `numpy` — these are not pinned here.)
- **`.github/`** — CI that runs tests + lint on every push, keeping the core safe.
- **`CONTRIBUTING.md`** — how to run tests, submit PRs, and what not to break.
- **`visualizer/index.html`** — interactive 3D view of the cube cells, exposure
  classes, and rotations. The best teaching tool — people understand Livnium faster
  when they can see it.

---

## The clean final meaning

Livnium right now is best described as **a reversible base-27 symbolic-geometry
system with a conserved cube/lattice structure, plus honest experiments showing
that this structure alone does not create language understanding.**

That is a strong position. The project has a real center (Livnium Core), and the
partial/failed ML attempts are not trash — they are evidence that maps the boundary.
