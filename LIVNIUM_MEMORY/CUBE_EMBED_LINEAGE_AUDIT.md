# Cube-Embedding and Holonomy Lineage Audit

Updated: 2026-07-26  
Recovery stage: S16  
Primary source: `/Users/chetanpatil/Desktop/test/cube_embed`  
Adjacent memory/source: `session_summary.md`, `ablation_study.py`,
`architecture.md`, and their organized copies  
Handling: preserve as a historically important experimental lineage; do not
promote its semantic-holonomy claim without a redesigned representation and
clean controls

## Short verdict

This is a real, substantial prototype, not an empty folder. It contains a
clear sequence of attempts:

1. map word spelling and phonological heuristics to disturbances on a
   26-node cubie graph;
2. replace that graph response with a 27-position vector field;
3. inject PPMI-SVD semantic vectors into the field;
4. add lexical counter-fitting and a per-word spin/lens layer;
5. evolve fields sequentially for sentences;
6. test the resulting features on SimLex-style pairs and SNLI;
7. run a later ablation that correctly concluded the holonomy features were
   noisy.

The best part is the experimental self-correction: `ablation_study.py` and the
March 12 session summary explicitly withdraw the earlier single-run SNLI
interpretation. The reusable components are the compact PPMI-SVD pipeline, the
readable experiment decomposition, the negative ablation lesson, and the
general habit of comparing geometry to a raw-vector baseline.

The central claim does not survive the code and controls:

- the “holonomy” is an unsigned sum of ordinary pairwise angles around fixed
  loops, with no connection, parallel transport, group product, orientation,
  or path-dependent transported state;
- the final QR probes are mutually orthogonal, so the documented local
  Fourier correlation is removed;
- the raw cosine Fourier matrix has rank 14, and QR supplies 13 arbitrary
  completion directions;
- every within-field measurement depends only on squared probe coefficients,
  so a word vector and its negative have exactly the same signature and every
  probe coefficient can be sign-flipped independently;
- 40 of the claimed 94 dimensions are reconstructed from the first 54 edge
  phases, adding no independent measurement;
- the transform changes under a shared rotation of the underlying semantic
  basis even though ordinary cosine does not;
- the source’s SimLex counter-fitting learns from the same pair scores it then
  evaluates;
- a pair-held-out control collapses the apparent gain;
- on a clean SNLI test split, a standard 27D SVD sentence-pair representation
  beats the full cube feature set by 6.2 percentage points.

Current classification:

> **Historically valuable negative-result lineage with reusable evaluation
> pieces; not validated semantic holonomy and not an active model candidate in
> its present form.**

The historical 96% SNLI result is not part of this folder and remains
provisionally leaked/unusable under the existing recovery rule.

## What this reminds us you were doing

The March 11–12 work was a rapid sequence of increasingly grounded attempts to
turn the broader Livnium cube idea into a language representation.

You started with a literal cubie graph and asked whether a word could be a
disturbance rather than a coordinate. You then noticed the graph response was
too homogeneous, moved to angular fields on a 3×3×3 lattice, and tried to make
the field carry loop structure. When character-derived fields did not align
with semantic neighbors, you injected PPMI-SVD vectors. When raw distributional
vectors confused antonyms and synonyms, you added counter-fitting and a
spin/lens layer. You then extended the state update to sentences and SNLI.

The important intellectual turn came afterward: you wrote a direct
`ablation_study.py` asking what the holonomy contributed beyond one SVD cosine
and lexical overlap. That experiment answered “not enough; currently noise.”
The later session summary preserved that correction in unusually direct
language. That is the part worth carrying forward.

## Preservation and scope

No file in `cube_embed` was edited, moved, renamed, or deleted during this
audit. Fresh controls were written only to the recovery memory.

Primary folder inventory:

- 37 files total;
- 19 live Python source files;
- 13 historical CPython 3.10 bytecode files;
- 2 WikiText token files plus WikiText README and license;
- 1 `.DS_Store`;
- zero saved JSON/CSV/log/plot/checkpoint/model-result artifacts.

The source folder is untracked in the root Git worktree. Its historical
timestamps form one concentrated development sequence:

| Time | Generation |
|---|---|
| 2026-03-11 19:36–19:55 | v1 graph-response operators, demo, embedder, WikiText evaluation |
| 2026-03-11 20:14–20:24 | v2 3×3×3 field, fixed loops, angle embedder, character evaluation |
| 2026-03-11 21:25–23:47 | PPMI-SVD semantic injection |
| 2026-03-12 00:30–00:54 | counter-fitting, spin/lens layer, SimLex runner |
| 2026-03-12 03:15–03:32 | trigram fallback, sentence field, SNLI trainer |
| 2026-03-12 | session summary and later root-level ablation lineage |

Adjacent exact copies:

- root `session_summary.md` and
  `_ORGANIZED/00_START_HERE/session_summary.md` are byte-identical, SHA-256
  `292a02d3c80b05ea57e0bd5cd67d41269f71c3ef277153b02c09eda1d9af89e7`;
- root `ablation_study.py` and
  `_ORGANIZED/02_Experiments/Nova-and-Misc/ablation_study.py` are
  byte-identical, SHA-256
  `b3111e938249f9b4e8941f5f639d4c17347b7f92e031875eef3b736c1e1a4b37`.

The summary also discusses KarmicBasin and `nli_v8_nova.py`. Those are adjacent
NLI lineages, not evidence for this cube representation. The DTW/lexical-memory
mechanisms are already incorporated under the NLI-ALL audit and should not be
credited to cube holonomy.

## Data identity and a major historical mismatch

The folder does not contain `wiki.train.tokens`.

It contains:

- `wiki.valid.tokens`: 1,142,309 bytes, 177,028 regex tokens, 15,619 types;
- `wiki.test.tokens`: 1,281,077 bytes.

Every source runner that requests the missing local train file silently falls
back to `wiki.valid.tokens`. With the default minimum frequency of 30, the SNLI
pipeline obtains only 636 vectors despite declaring a vocabulary cap of 8,000.
Fresh source replay measured 84.0% token OOV on its first 200 development
examples.

A separate 539,209,157-byte WikiText train file exists at
`/Users/chetanpatil/Desktop/test/data-bank/wiki.train.tokens`, but the archived
source does not auto-detect or use it.

Therefore the March session summary’s statements that this run used roughly
100M WikiText tokens, 8,000 vectors, and 48.5% SNLI OOV do not describe the
currently preserved default execution. They may describe an earlier external
data arrangement, but no command log, result artifact, or copied train file
binds those numbers to the surviving source.

The official SNLI train, dev, and test JSONL files all survive in
`/Users/chetanpatil/Desktop/test/data`. The source uses train and dev but ignores
the available test split.

No official `SimLex-999.txt` survives in the archive search. The default runner
uses a manually assembled, partly approximate built-in list.

## Generation 1 — graph-response CubeEmbed

### Intended mechanism

`graph.py` defines 26 cubie nodes:

- 6 face centers;
- 12 edge pieces;
- 8 corner pieces.

`word_ops.py` maps spelling/phonological clusters, vowel ratio, word length,
and small hardcoded lexical lists to push, pull, bind, invert, or diffuse
operations. `embed.py` records 50 response features over graph diffusion.

This is a readable toy mechanism for studying deterministic graph response.
It is not corpus-learned semantics.

### Topology discrepancy

The documentation specifies physical cubie incidence:

| Node type | Documented degree | Fresh actual degree |
|---|---:|---:|
| face center | 8 | 8 |
| edge | 4 | 14 |
| corner | 6 | 18 |

The implementation connects every pair of nodes sharing any face. That makes
each edge adjacent to nearly every other node on either of its faces and each
corner adjacent to nearly every node on any of its three faces. It is a
face-membership clique graph, not the documented cubie-incidence graph.

### Zero-state behavior

Every word begins from all-zero activation:

- `pull(primary)` suppresses nothing and only the later opposite-face push has
  an effect;
- `invert(primary)` computes the maximum zero and changes nothing, leaving only
  the later secondary push;
- `diffuse(primary)` diffuses zeros and then applies the later small push;
- `push` and `bind` are the only immediately expressive disturbances.

Thus operator names such as negation/inversion do not express their named
operation on the initial state.

### Fresh WikiText replay

Command:

```bash
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
python3 cube_embed/wikitext_eval.py
```

Result:

- vocabulary: 600;
- source’s “known-pair agreement”: 8/8;
- every displayed cube similarity: 0.808–0.988;
- top-5 nearest-neighbor overlap with co-occurrence: 0/5 for all ten query
  words;
- polarity check: 1/2.

The 8/8 agreement is a weak threshold agreement: the collapsed cube
similarities all exceed 0.5 and the compared co-occurrences all exceed 0.05.
It does not establish ranked or lexical-semantic agreement.

Examples included `king → took/take/oregon/black/working` and
`war → way/army/many/maya/major`, all with cube similarities near 1. This
generation is best understood as spelling/operator response with severe
similarity collapse.

## Generation 2 — AngleCubeEmbed

### Intended mechanism

Each word begins with a 27D unit vector `b`. A 27×27 probe matrix supplies one
probe `e_p` per lattice position. With injection strength `alpha=2.5`:

```text
a_p = b · e_p
u_p = normalize(b + alpha a_p e_p)
```

The field signature concatenates:

- 54 normalized angles on face-adjacent lattice edges;
- 27 per-node mean neighbor cosines;
- 13 normalized sums of edge angles around fixed loops.

The similarity function uses Pearson correlation within the three blocks,
weighted 0.3 edge, 0.2 neighbor, and 0.5 loop.

### Fourier matrix rank and QR completion

The raw probe matrix contains only cosines over the 3D frequency grid. Because
cosine identifies opposite frequencies, it has rank 14, not 27.

Fresh singular-value reconstruction gives fourteen singular values equal to
`sqrt(27)=5.196152...` and thirteen numerical zeros. The code then computes QR
on the transpose and returns all 27 rows of `Q.T`. That necessarily supplies
13 completion directions not determined as independent cosine Fourier
features.

The resulting rows are orthonormal to `7.77e-16`. Consequently:

- mean neighbor probe dot product is approximately zero;
- mean non-neighbor probe dot product is also approximately zero;
- neighbor probes are no more correlated than non-neighbor probes.

This directly contradicts the source comment that QR preserves correlated,
locally smooth neighboring probes.

### Exact sign ambiguity

Let `e_p` be the orthonormal probe rows and `a_p=b·e_p`. Before row
normalization, for two different positions:

```text
u_p · u_q = 1 + alpha(a_p² + a_q²)

||u_p||² = 1 + (2alpha + alpha²)a_p²
```

Therefore every edge cosine depends only on `a_p²` and `a_q²`. The full
signature has the following exact ambiguity in real arithmetic:

```text
signature(b) = signature(-b)
```

More strongly, each of the 27 probe coefficients may be sign-flipped
independently without changing the signature. There are up to `2^27` semantic
directions mapped to one signature, before considering other collisions.

Fresh numerical checks:

- `signature(b)` versus `signature(-b)`: maximum difference `0.0`;
- independent coefficient sign flips: maximum float32 difference
  `2.84e-6`;
- analytic edge-cosine formula versus direct computation: maximum error
  `3.33e-16`.

The small independent-flip difference is float32 normalization noise, not a
distinct signal.

This matters because semantic opposition and direction live in the signs of
vector coordinates. The transform discards precisely that information before
claiming to recover opposition through loop geometry.

### The 94 dimensions contain only 54 measured edge values

Every loop winding is the arithmetic mean of the edge phases along that loop.
All loop edges are among the first 54 signature entries.

Every neighbor-cosine value is:

```text
mean_q cos(pi × normalized_edge_phase(p,q))
```

Thus the remaining 40 values are deterministic functions of the first 54.

Fresh reconstruction:

- loop incidence matrix rank: 13/13;
- loop block from the edge block: maximum difference `8.38e-9`;
- neighbor block from the edge block: maximum difference `6.06e-8`;
- entire 94D signature rebuilt from first 54D: maximum difference `6.06e-8`.

The signature may still be a nonlinear feature expansion for a classifier, but
it is not 94 independent geometric measurements. The 13 loops especially add
no information not already in the edge vector.

### Why this is not holonomy

`loop_winding` sums nonnegative `arccos` angles and divides by loop length and
pi.

It has:

- no oriented sign;
- no connection or gauge variable;
- no parallel-transport map;
- no composition/product of transports;
- no state transported around the loop;
- no comparison between initial and returned transported state;
- no path ordering;
- exact invariance to loop reversal.

Fresh reversal difference is `0.0`.

The implemented quantity is a closed-loop perimeter roughness or total
variation. Calling it a loop statistic is accurate. Calling it geometric or
gauge holonomy is not.

### Basis-orientation dependence

Ordinary cosine between semantic vectors is invariant when all vectors are
multiplied by the same orthogonal matrix. The cube transform fixes probes in
coordinate space and is not invariant.

Across 50 random vector pairs and one shared random rotation:

- maximum raw-cosine change: `2.36e-16`;
- mean cube-similarity change: `0.2888`;
- maximum cube-similarity change: `0.9459`.

The result therefore depends on the arbitrary orientation of the SVD basis.
SVD sign and degenerate-subspace conventions can change the cube score while
leaving the underlying semantic geometry equivalent.

### Character fallback reproducibility

`char_embed.py` says its trigram vectors are deterministic across all runs and
processes. Its table is fixed, but trigram row selection uses Python’s built-in
`hash()`, which is process-randomized unless `PYTHONHASHSEED` is fixed before
startup.

For `"running"`:

- seed-0 versus seed-1 vector cosine: `-0.020775`;
- seed-0 versus an unfixed random seed also changes;
- bytewise/vector equality across processes: false.

This explains why fresh reruns can differ from the remembered March numbers.
The archived three-seed ablation seeds NumPy and Python’s `random` module but
does not pin `PYTHONHASHSEED`; with 84–89% OOV, the unpinned fallback is
load-bearing.

### Character and semantic replays

Character-field replay:

```bash
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
python3 cube_embed/angle_eval.py
```

- known-pair agreement: 3/8;
- top-5 neighbor overlap: zero for all ten query words;
- polarity: 2/2, only two pairs available;
- examples include `city → opened/play/emperor` and
  `army → attack/area/running`.

PPMI-SVD injection replay:

```bash
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
python3 cube_embed/semantic_eval.py
```

- known-pair agreement: cube 5/8, raw SVD 6/8;
- mean top-5 overlap with co-occurrence: cube 0.12, raw SVD 0.34;
- polarity: cube 1/2, raw SVD 0/2;
- cube similarity has greater random spread, but spread is not correctness.

The semantic signal comes from the PPMI-SVD base. The cube transform degrades
the strongest retrieval check.

## Generation 3 — SimLex, counter-fitting, and spin/lens

### Built-in data boundary

The file says its built-in representative subset contains 150 pairs. It
contains 131. Comments mark many scores as approximate. This is neither a
surviving official SimLex-999 file nor an independently verifiable benchmark
artifact.

Under the preserved default corpus and `min_freq=30`:

- 207 unique listed words;
- 44 meet the vocabulary threshold;
- 15 of 131 pairs are scoreable;
- 116 are skipped.

### Default raw result

Fresh replay with `--no-cf`:

- raw SVD Spearman rho: `-0.4179`, p≈0.241, N=15;
- raw cube rho: `-0.3107`, p≈0.405, N=15;
- spin still trains from the same labels even under `--no-cf` and reaches
  `+0.0321`.

The tiny raw sample is negative and not statistically informative.

### Direct target leakage

Default `simlex_eval.py` extracts:

- synonym constraints from pairs with human score ≥8;
- antonym constraints from pairs with human score ≤2.

It counter-fits on those exact evaluation pairs, scores the same pairs, then
chooses a blend coefficient on those same human scores. The spin/lens layer is
again trained on the same target pair labels.

Fresh default replay on N=15:

| Condition | Spearman rho |
|---|---:|
| raw SVD | -0.4179 |
| raw cube | -0.3107 |
| SVD + same-pair counter-fit | +0.8143 |
| cube + same-pair counter-fit | +0.5571 |
| same-set tuned blend | +0.8893 |
| spin | +0.8143 |

Two synonym and seven antonym constraints directly cover 9 of the 15 target
pairs. This is resubstitution, not generalization.

### Expanded-coverage held-out control

The reusable audit probe lowered the corpus frequency threshold to 1 only to
increase coverage of the built-in list; it did not import an external dataset.
This produced 4,000 PPMI-SVD vectors and 100 scoreable pairs.

Raw:

| Condition | Spearman rho |
|---|---:|
| SVD | +0.0836 |
| cube | -0.1133 |

Same-pair fitting:

| Condition | Spearman rho |
|---|---:|
| SVD + counter-fit | +0.8459 |
| cube + counter-fit | +0.5961 |
| same-set tuned blend | +0.8471 |

Five-fold pair-held-out out-of-fold scoring:

| Condition | Out-of-fold rho |
|---|---:|
| SVD + counter-fit | +0.0337 |
| cube + counter-fit | -0.0620 |
| blend alpha selected only on each training fold | +0.0296 |

Fold-selected blend alphas were `[0.0, 0.0, 0.1, 0.1, 0.1]`, meaning the
training folds themselves preferred almost no cube contribution.

This control is still limited by the manually assembled pair list and small
WikiText validation corpus. Its purpose is not to establish a benchmark score;
it shows that the source’s dramatic increase is explained by fitting the
target pair labels.

### Cube-layout control

With the same semantic vectors and pairs, the source’s raw cube rho was
`-0.1133`. Across 20 random assignments of the same orthonormal probes to cube
positions:

- mean rho: `-0.0350`;
- standard deviation: `0.0533`;
- minimum: `-0.1024`;
- maximum: `+0.0652`;
- source layout rank: 21st of 21.

The chosen adjacency/layout shows no advantage in this control.

### Historical full-SimLex numbers

The March session summary records:

| Model | Remembered rho |
|---|---:|
| cube only | +0.441 |
| SVD only | +0.706 |
| cube + counter-fit | +0.688 |
| SVD + counter-fit | +0.775 |
| blend alpha=0.3 | +0.784 |

These numbers must be preserved as historical memory, not erased. They are not
currently reproducible evidence because:

- the official 999-pair file is absent;
- the command and exact data hash are absent;
- no result log is saved;
- the source trains counter-fitting and blend selection on target scores;
- the remembered corpus/vocabulary does not match the surviving default data.

Status: **historical unattested/resubstitution result, not a valid benchmark
claim**.

## Generation 4 — sentence field and SNLI

### What SentenceField actually computes

The state update is:

```text
S_(t+1) = row_normalize(alpha S_t + beta Delta(word_(t+1)))
```

This is order-sensitive. It is a fixed recency-weighted nonlinear pooling
scheme because normalization occurs after each addition. It does not alter a
word operator based on linguistic context, attention, syntax, or a learned
sense state.

Two exact degeneracies:

- multiplying alpha and beta by the same positive constant leaves every state
  and final signature unchanged; only their ratio matters;
- `alpha=0, beta=1` returns the last word’s field exactly, not a sentence mean.

Fresh checks:

- `(alpha,beta)=(0.7,0.3)` versus `(0.35,0.15)`: signature difference `0.0`;
- a full sentence at `alpha=0` versus its last word alone: difference `0.0`.

The class tracks a 13D sum called `cumulative_holonomy`, but `embed()` returns
the final 94D field signature instead. The claimed trajectory accumulator is
not part of the classifier input.

So the preserved narrow claim is:

> The sentence representation is deterministically order-sensitive through
> fixed recency/state blending.

The retired claim is:

> It performs context-dependent semantic transport or returns the holonomy of
> the sentence trajectory.

### Pair features

The SNLI model uses:

- 376 cube signature pair features `[u,v,u*v,|u-v|]`;
- 10 hand-engineered negation features, multiplied by 5;
- 1 raw mean-SVD sentence cosine.

Any accuracy from the 387D model cannot be attributed to cube geometry without
ablation.

### Source evaluation protocol problems

`snli_train.py`:

- scans the full SNLI training vocabulary;
- builds only 636 WikiText-valid vectors in the preserved environment;
- counter-fits using a small hardcoded lexicon;
- grid-searches alpha and beta on development examples 0–499, training the
  quick classifier on dev 0–299 and evaluating dev 300–499;
- trains the final classifier on 10,000 train examples;
- checks all 2,000 dev examples after every epoch for early stopping;
- tunes a contradiction-logit boost on those same 2,000 dev examples;
- reports those same 2,000 dev examples;
- tunes the “SVD baseline” thresholds and reports accuracy on the same first
  500 dev examples;
- never loads the surviving SNLI test split.

The declared alpha and beta are hyperparameters, not trainable parameters.
Because scale is redundant, the 3×3 grid is effectively a search over ratios,
not two independent degrees of freedom.

The development score and 500-example baseline are also not directly
comparable because they cover different sample sizes and both are selected on
their reported data.

### Fresh source replay

Command:

```bash
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
python3 cube_embed/snli_train.py
```

Key output:

- semantic vocabulary: 636;
- dev-200 token OOV: 84.0%;
- chosen `alpha=0.7, beta=0.5`;
- grid macro-recall: 0.338;
- train accuracy: 0.4152;
- dev accuracy: 0.4070;
- source SVD threshold baseline on dev-500: 0.4240;
- majority on dev-2000: 0.3385;
- contradiction recall: 0.206;
- explicit SVD cosine has mean absolute classifier weight 0.39731, far larger
  than any average cube block;
- reported “improvement over best baseline”: -0.0170.

The replay is a negative development result and does not recover the March
43.10% run.

### Replayed archived three-seed ablation

The root `ablation_study.py` was run with `PYTHONHASHSEED=0`. It uses the fixed
dev set for epoch selection and reporting, so it is still a development
ablation, not a held-out test result.

Fresh results:

| Condition | Dimensions | Mean dev accuracy | Std |
|---|---:|---:|---:|
| word overlap + negation | 13 | 48.92% | 0.18% |
| SVD cosine only | 1 | 40.77% | 0.13% |
| cube + negation | 386 | 35.82% | 1.47% |
| cube + negation + SVD cosine | 387 | 39.58% | 0.24% |

The full model is 1.18 points below the one-dimensional SVD cosine and its
neutral recall rounds to zero across the three runs.

The session summary remembers 41.40%, 36.25%, and 36.15% for the last three
conditions. Those exact values did not replay. The conclusion does replay:
cube features do not improve the SVD scalar and lexical overlap is much
stronger.

The script’s final two delta labels are swapped:

- `D-B` is printed under a “D vs C” label;
- `D-C` is printed under a “D vs B” label.

The table itself is the reliable part.

### Clean held-out SNLI ablation

The recovery probe used:

- first 10,000 valid train examples, shuffled with seed 42;
- first 2,000 dev examples only to select logistic regularization `C`;
- first 2,000 official test examples for one final report;
- the same 636-vector PPMI-SVD/lexicon-counter-fit base;
- frozen source-replay sentence ratio `alpha=0.7, beta=0.5`;
- standardized multinomial logistic regression;
- `PYTHONHASHSEED=0`.

Results:

| Condition | Dev | Held-out test |
|---|---:|---:|
| cube only, 376D | 44.80% | 40.40% |
| cube + negation, 386D | 44.20% | 40.60% |
| cube + SVD cosine, 377D | 43.95% | 41.20% |
| full source feature family, 387D | 44.70% | 41.65% |
| negation + SVD cosine, 11D | 39.70% | 39.35% |
| SVD sentence pair, 108D | 45.25% | **47.85%** |
| SVD sentence pair + negation, 118D | 45.40% | 47.15% |
| negation only, 10D | 34.25% | 35.10% |
| SVD cosine only, 1D | 39.70% | 39.70% |
| train-majority | 33.15% | 34.50% |

Key held-out differences:

- full cube family minus SVD sentence pair: `-6.20` points;
- full cube family minus SVD sentence pair plus negation: `-5.50` points;
- cube only minus majority: `+5.90` points;
- full cube family minus negation plus one SVD cosine: `+2.30` points.

The cube features carry some supervised SNLI signal above majority, but the
underlying raw 27D semantic vectors are more useful when presented directly.
The result supports “nontrivial but lossy transform,” not “geometric advantage.”

### Historical SNLI progression in the session summary

The March summary remembers seven successive development runs from 37.0% to
43.10%, with a 42.80% one-scalar baseline, then later calls that best run a
cherry-pick after the three-seed ablation.

Preserve both statements:

1. 43.10% was the remembered best development run during exploration.
2. The same historical summary later withdraws it as a stable finding.

There is no saved result log or model for the 43.10% run. Fresh source replay is
40.70%, fresh three-seed development ablation disfavors the cube block, and the
new held-out test favors direct SVD pair features.

Status: **historical exploratory run, explicitly superseded by its own
ablation and by fresh held-out control**.

## Internal summary/architecture inconsistencies

The March summary should be preserved as memory, but not treated as a verified
manifest:

1. It says the package used WikiText-103 at roughly 100M tokens and 8,000
   vectors; current default uses 177,028 validation tokens and 636 vectors.
2. It says the trigram fallback is deterministic across processes; Python
   `hash()` makes that false without an external seed.
3. It reports full SimLex-999 results, but no official file or log survives.
4. It describes 94 dimensions as distinct channels, while 40 are derived from
   the 54 edge phases.
5. Its early overview calls loop-angle sums holonomy without the required
   transport structure.
6. It describes the 486 symbolic-weight invariant and divergence law from
   broader `architecture.md`; neither is implemented as a correctness boundary
   in `cube_embed`.
7. The summary’s displayed divergence interpretation says positive divergence
   means entailment and negative means contradiction. `architecture.md` says
   the opposite: with `divergence=0.38-alignment`, negative is entailment and
   positive is contradiction.
8. `snli_train.py` does not use that divergence rule anyway; it trains a
   softmax classifier.

These are normal signs of fast exploratory work. The recovery action is to
separate remembered direction from executable evidence, not delete the memory.

## Claim ledger

| Claim | Current verdict |
|---|---|
| Words can be represented as deterministic disturbances on a graph | Preserved as a toy/reusable design pattern |
| v1 uses the documented physical cubie adjacency | Falsified: edge/corner degrees are 14/18 instead of 4/6 |
| v1 pull/invert/diffuse act meaningfully from the initial state | Narrowed: their primary operation is null on zero state; later pushes carry the response |
| v1 embeddings align with WikiText semantics | Falsified by collapsed similarities and zero top-5 neighbor overlap |
| The QR probes retain local Fourier correlation | Falsified: final probe rows are orthogonal for neighbors and non-neighbors alike |
| The cosine Fourier probes span 27 independent directions | Falsified: raw rank is 14; QR supplies 13 completion directions |
| The field preserves semantic direction | Falsified: independent sign flips of all 27 probe coefficients leave the signature invariant |
| The 94D signature contains 94 independent measurements | Falsified: all 40 neighbor/loop values are functions of the first 54 edge phases |
| Loop windings are holonomy | Retired terminology: they are unsigned loop perimeter/roughness statistics |
| Loop orientation matters | Falsified: reversing a loop changes nothing |
| The transform is intrinsic to semantic geometry | Falsified: shared semantic-basis rotations change scores by up to 0.946 |
| Character trigram fallback is process-deterministic | Falsified unless `PYTHONHASHSEED` is externally fixed |
| PPMI-SVD grounds the model in corpus semantics | Preserved: this is the actual source of semantic signal |
| Counter-fitting separates supplied antonym/synonym pairs | Preserved conditionally for its training constraints |
| Default SimLex gain generalizes | Falsified by direct same-pair leakage and pair-held-out collapse |
| The chosen cube layout helps SimLex | Negative control: source layout ranked last of 21 |
| Spin/lens adds value after counter-fitting | Null result: it remains effectively identity when margins are already satisfied |
| SentenceField is order-sensitive | Preserved narrowly as fixed recency/state blending |
| SentenceField returns trajectory holonomy | Falsified: it returns the final 94D field; cumulative 13D trace is tracked separately |
| Alpha and beta are independent trainable parameters | Falsified: they are grid-searched hyperparameters and common scale cancels |
| The full cube SNLI model beats raw semantic features | Falsified on held-out test: 41.65% versus 47.85% for direct SVD pair features |
| Cube features are pure noise | Too strong: cube-only reaches 40.40% versus 34.50% majority, but is a lossy transform and loses to direct semantic features |
| The remembered 43.10% is a stable result | Retired by the archive’s own later ablation and absent saved artifact |
| The historical 96% supports cube holonomy | No: it belongs to another lineage and remains leaked/unusable |

## What is genuinely reusable

### Keep

- `cooc_ops.py` as a small, readable PPMI-SVD teaching/prototyping pipeline;
- explicit raw-vector versus transformed-vector evaluation;
- `counter_fit.py` as a compact implementation whose constraints must be kept
  separate from evaluation pairs;
- the fixed-lattice edge and loop enumerations as generic graph-feature
  utilities, without calling angle sums holonomy;
- `ablation_study.py` as evidence of scientific self-correction, while fixing
  split handling, hash seeding, and swapped labels;
- the idea of preserving negative results and channel-level comparisons;
- held-out test ablations and random-layout controls.

### Do not port unchanged

- the 27D cosine-only Fourier probe followed by full QR;
- sign-destroying within-field angle signature as a semantic representation;
- the redundant 94D signature;
- Python built-in hash for reproducible feature identity;
- target-pair counter-fitting or hyperparameter selection on evaluation scores;
- dev-set early stopping, calibration, and final reporting without test;
- the physical cubie adjacency claim for the face-sharing clique graph;
- broader architecture invariants as evidence for a module that does not test
  or implement them.

## If this idea is ever restarted

Do not continue by tuning alpha, adding more loop sums, or training another
classifier on the same signature. The exact sign and redundancy failures are
architectural.

A legitimate restart would require:

1. define the mathematical object first: a connection, transport operators on
   directed edges, path-ordered composition, and gauge/basis behavior;
2. use orientation-sensitive transport, not `arccos` magnitudes;
3. prove what information the representation preserves and quotient only the
   invariances intentionally desired;
4. keep the semantic representation invariant to arbitrary SVD sign and basis
   conventions, or learn probes jointly under explicit constraints;
5. declare train/dev/test identities and hash all data;
6. keep lexical constraints completely outside the evaluation pairs;
7. compare with direct SVD/SGNS/fastText/transformer baselines at matched
   dimension and classifier capacity;
8. include random probe, random layout, no-loop, edge-only, and identity
   transforms;
9. report multiple seeds and the untouched test set once;
10. require improvement over direct semantic features, not merely over
    majority.

Until those conditions exist, the right next action is preservation, not more
training.

## Reproduction commands

Fast exact/algebra controls:

```bash
cd /Users/chetanpatil/Desktop/test
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
python3 .codex_memory_staging/CUBE_EMBED_AUDIT_PROBE.py --skip-semantic
```

Expanded SimLex leakage/layout controls:

```bash
cd /Users/chetanpatil/Desktop/test
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
python3 .codex_memory_staging/CUBE_EMBED_AUDIT_PROBE.py
```

Held-out SNLI ablation:

```bash
cd /Users/chetanpatil/Desktop/test
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
python3 .codex_memory_staging/CUBE_EMBED_SNLI_ABLATION_PROBE.py
```

Historical source replay:

```bash
cd /Users/chetanpatil/Desktop/test
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 python3 cube_embed/wikitext_eval.py
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 python3 cube_embed/angle_eval.py
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 python3 cube_embed/semantic_eval.py
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 python3 cube_embed/simlex_eval.py --no-cf
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 python3 cube_embed/simlex_eval.py
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 python3 cube_embed/snli_train.py
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 python3 ablation_study.py
```

## Source manifest

Live source SHA-256:

| File | SHA-256 |
|---|---|
| `__init__.py` | `0386d4e9677db389b5aad5ed21bf212b368a11cfde589e96689748ba5b6599e8` |
| `graph.py` | `dafc12c397597ae00683e43b828c4fddd967561be09d7c20d12ede7c06c2432e` |
| `word_ops.py` | `e548b29950e6ffae81a611cdb086ca182f89b480a34f20ecf15b2481b4bd07e5` |
| `embed.py` | `83b4a9a64f4a8da23507fb0570bef00a411516a709cc00e98cec8a077994f032` |
| `demo.py` | `81ad3e6ed626c868e4897b5454c74d782c491f33e0259da1a16a5f55b2347cd8` |
| `wikitext_eval.py` | `9a904e611b2468ef724feec492fb099c0c6ee332fa7b532f90027831e034ef6a` |
| `field.py` | `c41b1c726788d40753e1c421461af6deea8a6215ee2ade26eb01e651d0388ab4` |
| `loops.py` | `686f0f53a6d81a64ce926f9a8c8438646f17616bd18adbe4925aaec5bb869c85` |
| `angle_ops.py` | `46c2e963828fa6695f152913c4579f6fcd415e22976f14e5e7fe64b569b6a9af` |
| `angle_embed.py` | `6786f2e4c0bf2612f78910aee3114f033dd09d4d890a1b4c84de556ca0866e03` |
| `angle_eval.py` | `a1e9a797596566f198b45c7a3c1734e973b00ca4ba45a8f114b92f92f664aaa3` |
| `char_embed.py` | `ecf7b173bf73fbfa1c01ae17239959a93207addfe85e0202b6e58931b2950f5c` |
| `cooc_ops.py` | `b832114bc31ef3d4f651f7f7d9c13d1f0fc4ba38fa0c524c2136d7cf3f43ce3e` |
| `semantic_eval.py` | `77a0c7642bed8f8a65507c96d07e16840c89dff9fd61045376b116c5f6943244` |
| `counter_fit.py` | `b762047d6081ecf4647e855cf9b78153e8e4595e016fdcf89d8fd60347f298d0` |
| `spin_ops.py` | `11871c3e2938e7425be50a7b07fe8434996aa91159f90dc855a234166614e18a` |
| `simlex_eval.py` | `6f380838e8514eb1b894b56a041f33f73bff78eb563bd483f3e07ba9d438534b` |
| `sentence.py` | `32d2abc8689456a609879d1bb2c57a3df7316f6392fd5e79e926bf0bf5dab273` |
| `snli_train.py` | `ff0a7170b467a17132dc7411d9b3cdb88c72f291f0b14e82cbbc9cd8ceaf353d` |

Data/support hashes:

| File | SHA-256 |
|---|---|
| `wikitext-103/wiki.valid.tokens` | `9aee9f52d6e77de6c7751fc5e7c44287db3ec0eaf15477098acc6b93c1c4250b` |
| `wikitext-103/wiki.test.tokens` | `56db52eb157b1ab5c89646eef70510c1eca6e12a3bfdb99f0d6b05b4eacb75b9` |
| `wikitext-103/README.txt` | `1b50ff59fefaf57b1af4cd875ef0fde65f9bbf8db7ffb5ad64e43084ef3f66bf` |
| `wikitext-103/LICENSE.txt` | `bb21dc59e69079967d10f54aaa408c2ba548210aea16babf57b4d5f2e8f169c5` |

## Stop point

S16 is incorporated when this report, both probes, and all navigation ledgers
are copied and byte-verified in `/Users/chetanpatil/Desktop/LIVNIUM_MEMORY`.

Do not repeat:

- do not call unsigned loop-angle sums holonomy;
- do not call 94 derived features 94 independent dimensions;
- do not evaluate lexical constraints on the pairs that trained them;
- do not rely on Python `hash()` without fixing or replacing it;
- do not compare a selected 2,000-example dev result to a separately selected
  500-example baseline as if it were held out;
- do not lose the negative ablation or replace it with the remembered best
  seed;
- do not attach the historical 96% result to this folder.

