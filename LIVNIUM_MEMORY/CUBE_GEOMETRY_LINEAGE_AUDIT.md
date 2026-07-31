# Cube and Geometry Lineage Audit

**Session:** S13  
**Audit date:** 2026-07-26  
**Historical source tree:** `/Users/chetanpatil/Desktop/test`  
**Canonical reading copy:**  
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Cube-and-Geometry`  
**Historical root mirror:** `/Users/chetanpatil/Desktop/test`  
**Independent probes:** `CUBE_SOKOBAN_AUDIT_PROBE.py` and
`CUBE_GEOMETRY_AUDIT_PROBE.py` in this memory

## Executive verdict

The entire 22-file Cube-and-Geometry family is one lineage copied exactly between
the organized folder and the archive root. It contains:

- CubeSokoban rotation-canonical recognition;
- the odd-cube directional sign-block decomposition;
- a learned directional autoencoder;
- a partially rotation-tied autoencoder;
- robust Laplacian graph-signal denoising described as geometry-direct truth
  decoding; and
- 3-D Om/LO polarity features for SNLI.

Two mathematical/engineering components survive cleanly:

1. the generated 24 cube rotations are unique bijections and closed under all
   576 compositions; and
2. the directional sign-block partition satisfies the binomial identity
   `(2m+1)^3 = 1 + 6m + 12m² + 8m³` for odd cubes.

The major headlines require narrowing:

- **CubeSokoban's 100% is deterministic orbit normalization, not learned task
  generalization.** Every one of the 720 canonicalized “unseen rotation” rows is
  byte-identical to a canonicalized training row. A hash lookup or one-template
  nearest neighbor gets the same 100%. The labels are identities of the same 40
  random worlds in train and test. There is no player, crate, goal, move,
  reachability, or Sokoban solution.
- **The learned directional autoencoder supports learning local filters, not a
  decisive cube advantage.** Its saved linear directional score is 0.636 versus
  fixed block means at 0.596 and PCA at 0.718. A fresh crossed control over five
  model seeds and eight random partitions per seed gives directional 0.6357
  versus random mean 0.6150, but directional beats only 30/40 random layouts;
  some random layouts reach 0.6458 and win.
- **The alleged rotation-equivariant autoencoder is not equivariant as a whole.**
  Only encoder weights are tied. Nine independent encoder biases and the dense
  decoder/bias are untied. The advertised 13-versus-49 count excludes the shared
  decoder and biases; total parameters are 512 versus 548, only 6.57% fewer.
  Its saved result is negative anyway: tied is worse than untied on both
  oriented digits and isotropic fields.
- **Geometry-direct is robust graph smoothing, not a low-dimensional
  error-correcting code or truth oracle.** `I + 5L` is full rank 343, so the
  generated “smooth field” family spans all of `R^343`. The method strongly
  favors the source's smooth prior: clean smooth data decode with 0.016 relative
  error, while a clean checkerboard is distorted by 0.637. The claimed “social
  median” is constructed as exactly the naive corrupted report array.
- **Om/LO polarity is a small supervised feature diagnostic, not novel observer
  information.** All three LO features are algebraically determined by the four
  Om features; fresh numerical reconstruction is exact to `3.75e-15`. The saved
  40.39% SNLI result is above nominal chance but below bag-of-words at 62.44%,
  and it lacks hypothesis-only and matched lexical controls.

The strongest honest conclusion is not “the cube finally wins.” It is:

> Exact group normalization guarantees invariance when the task quotient is
> known. Directional locality can be a useful standard prior. Neither result
> establishes learned reasoning, Sokoban solving, universal truth recovery, or
> a uniquely Livnium advantage.

## Source identity and preservation

Every one of the 22 organized files has a same-named, byte-for-byte duplicate at
the root of `/Users/chetanpatil/Desktop/test`. These are two preservation copies,
not independent replications. The organized folder is the canonical reading copy
because it keeps the family together. Both historical copies remain untouched.

### Integrity manifest

| File | Bytes | SHA-256 |
|---|---:|---|
| `CUBE_SOKOBAN_VERDICT.md` | 3,622 | `f5f4cfc0bc1196ee090373976d05ea997f19f3e2fab3394014823b0360112222` |
| `DIRECTIONAL_DECOMPOSITION.md` | 2,161 | `d96c45c25f59516ec98e4c2c90355ee375c8d6357732d15ad84e273fdde965f7` |
| `LEARNED_CUBE_AE_VERDICT.md` | 3,268 | `3812051399f069fca6636e1fc09ed9437c3e7dda3636c4e8ab8e87749c480d83` |
| `LV_GEOMETRY_DIRECT_VERDICT.md` | 2,699 | `50fb3bd57c732f44580e35b707e10b15995450458ca6238e65f6fe31244e034a` |
| `OM_LO_3D_VERDICT.md` | 3,022 | `503fde5eeb1e9e4a063639abbf7d6db73af5ce1f867d14ad5f991a8f2125b2c5` |
| `ROTATION_VERDICT.md` | 3,441 | `bbb63dd0c8bf0b58aae402a9e5df5ec21ec0751770f5a342847e764a48c587c2` |
| `cube_sokoban_symmetry.py` | 6,443 | `d655387fe11f8dcdb39d03a79417903b25307054d4bbb3507637398c005b1946` |
| `cube_sokoban_symmetry.json` | 207 | `2eec93247ff21fc8b28fe4f4aab427ffdbc53a244fb3479596791bbe679c43f2` |
| `cube_sokoban_symmetry.png` | 37,806 | `74cce513c7a1e5beb8d48bf2e03af359f7fbfae7fea78a82728c756401dfff68` |
| `livnium_directional_numbers.py` | 2,218 | `00807658303fcefd722294cdd012de0b29e85bd44f04613cdcc7ae1c7ccdcc50` |
| `livnium_geometry_direct.py` | 8,787 | `2587bd7347c57b23df3c6e62e8b8547b9e0726c8164cb0c12712dcabcacb1654` |
| `livnium_geometry_direct.json` | 505 | `1418c9e64cc278d8683c6892c157c67161e7287e35b66f7935183edfb5adc46d` |
| `livnium_geometry_direct.png` | 71,436 | `b35f96d390f078c8fd1a49e034e906193544682eb81bf842fd213fdbea7af025` |
| `livnium_learned_cube_ae.py` | 7,641 | `62657cc7cc9ee5775d5d7f7b507573a2861fc90f5c334adb0813faefba2c690f` |
| `livnium_learned_cube_ae.json` | 292 | `81e605d1afd6d6060d7729659751e4a8f1a8117a595a7ef9ff083f1766a1a7eb` |
| `livnium_learned_cube_ae.png` | 45,988 | `e0a874c647b9003e96d5c420c7ee73810620e6527d40d875a74d7332dc6f582f` |
| `livnium_om_lo_3d.py` | 7,410 | `ba2b282e52c7b756c51e8bdb89e606476737e5326df598cf11fd3891c7791a6e` |
| `livnium_om_lo_3d.json` | 187 | `f28218e63b0c8ead4965d1e7ff6dd14b34ade4e1433327d6b1664fa3cc7a4c7c` |
| `livnium_om_lo_3d.png` | 40,672 | `1786776fe35ba9bd77455f1778e368eb453fdd05ff635b457d8c89e5f9eaf895` |
| `livnium_rotation_equivariant.py` | 9,243 | `a6764be6bc6401bf347b1156f66c5e6bc0bcbe28feeec5cda6fa165bf1893a5b` |
| `livnium_rotation_equivariant.json` | 1,396 | `ab0ca61850e95e10646fac83643a70609eccd88da7cfdf871914f0b419b808d8` |
| `livnium_rotation_equivariant.png` | 107,128 | `2ccf5bc7ad36be6ec7df75e0a6ccd6d1c2916e1c042682875003aa6fb4ca5772` |

No checkpoint, learned weight file, per-example prediction file, or independent
replication accompanies this family.

## Independent audit method

The two reusable probes:

- import source modules without calling `main`;
- use `PYTHONDONTWRITEBYTECODE=1`;
- never write into the historical folder;
- verify group uniqueness, bijection, and closure;
- measure exact transformed train/test overlap;
- compare MLP output with hash and nearest-neighbor controls;
- test literal one-template canonical recognition and raw augmentation;
- verify the odd-cube partition identity beyond the four saved examples;
- inspect Laplacian/operator rank and prior mismatch;
- compare clean smooth and clean high-frequency reconstruction;
- repeat learned directional versus random partitions under matched
  initialization seeds;
- test whether weight tying guarantees encoder and whole-model equivariance;
- count complete rather than selected parameter totals; and
- reconstruct LO features algebraically from Om features.

Representative commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 CUBE_SOKOBAN_AUDIT_PROBE.py \
  --source /Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Cube-and-Geometry/cube_sokoban_symmetry.py \
  --model-seeds 10

PYTHONDONTWRITEBYTECODE=1 python3 CUBE_GEOMETRY_AUDIT_PROBE.py \
  --directory /Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Cube-and-Geometry \
  --ae-seeds 5 \
  --ae-random-partitions 8
```

## Generation A — CubeSokoban symmetry recognition

### What the program actually builds

Forty binary vectors are sampled independently. Each has 60 occupied cells among
the 125 coordinates of a 5x5x5 cube. The class label is the identity of that
random vector. Each vector is permuted through the 24 proper cube rotations.

Training and test use the same 40 underlying random worlds:

- train: six rotations per world, 240 rows;
- test: the other 18 rotations per world, 720 rows.

Canonicalization applies all 24 permutations to a row and chooses one
lexicographically minimal byte representation.

This dataset contains:

- no player;
- no crates;
- no goals;
- no legal moves;
- no push dynamics;
- no reachability;
- no solution target; and
- no actual Sokoban state.

The source says a prior full-Sokoban attempt sat at chance, but no source, log,
result, or model from that predecessor survives.

### The rotation action is correct

| Boundary | Fresh result |
|---|---:|
| Generated permutations | 24 |
| Unique permutations | 24 |
| Every permutation bijective | yes |
| Closure failures among 576 products | 0 |

This is verified engineering and a reusable exact group action.

### Saved MLP result reproduces exactly

| Model | Seen rotations | Unseen rotations |
|---|---:|---:|
| Plain, one rotation | 20.0% | 6.3889% |
| Plain, six rotations | 100% | 17.6389% |
| Canonicalized, six rotations | 100% | 100% |

Across MLP random states 0–9, canonicalized unseen accuracy remains 100%.
Plain-six unseen accuracy ranges from 16.94% to 22.78%, mean 18.96%.

### Why 100% is deterministic, not learned generalization

The transformed-data boundary is exact:

| Measurement | Result |
|---|---:|
| Canonical training rows | 240 |
| Unique canonical training rows | 40 |
| Canonical test rows | 720 |
| Test rows byte-identical to a canonical training row | 100% |
| Canonical keys with conflicting labels | 0 |
| All 24 views produce one canonical form per world | yes |

A direct canonical-byte hash lookup obtains 100% unseen-view accuracy. A
one-template-per-class canonical nearest-neighbor lookup also obtains 100%.
Neither learns a classifier.

This is not answer-label leakage. It is an intentionally exact quotient:
canonicalization maps every orbit member to the same representative. The correct
interpretation is a unit test of deterministic invariance. Calling it held-out
task generalization obscures that the preprocessed test inputs are exact training
inputs and the same world identities occur on both sides.

The labels prevent an unseen-world test: a new random world would be a new class
with no learned semantic property. A useful generalization benchmark needs a
property shared across different worlds—connectivity, reachability, object
count, solvability, value, or policy—not identity of a memorized template.

### Baseline boundaries

- Raw six-rotation 1-nearest-neighbor obtains 12.92% on the 18 held-out
  rotations.
- A plain MLP trained on 23 rotations reaches 95% on the one withheld rotation.
  Therefore it does not literally need to see the exact angle, although far more
  augmentation is required than exact canonicalization.
- Plain-one has only 20% accuracy even on the rotation set called “seen.” The
  verdict phrase “recognises only views it saw” is therefore not accurate for
  that model.

### Classification

**Verified canonicalization; retired as learned Sokoban evidence.** Preserve the
rotation generator, canonical orbit representative, and invariance tests. Do not
cite 100% as learning, unseen-world generalization, structure understanding,
Sokoban solving, or a unique Livnium mechanism.

## Generation B — directional sign-block decomposition

For odd side length `N=2m+1`, each axis is divided into:

- negative coordinates: `m` cells;
- zero: one cell; and
- positive coordinates: `m` cells.

The Cartesian products form 27 sign regions:

- one core region of size 1;
- six face-direction regions of size `m`;
- twelve edge-direction regions of size `m²`; and
- eight corner-direction regions of size `m³`.

The total is the binomial identity:

```text
1 + 6m + 12m² + 8m³ = (2m+1)³ = N³
```

Fresh enumeration verifies all odd `N` from 3 through 21. This is exact but
elementary combinatorics, not an empirical discovery.

The script counts regions; it does not itself pool values, learn a filter, or
prove a compression advantage. Claims about fixed directional pooling versus
random/radial pooling come from the already incorporated attractor/cascade
experiments and retain their prior status: a partial locality result on
scale-matched smooth fields, not a universal filter theorem.

The prose sentence that an inner N3 cell “is the average” of its block describes
a possible downstream pooling operator, not something established by the
counting script.

### Classification

**Proven partition identity / standard spatial decomposition.** Preserve as a
clear indexing utility and locality prior.

## Generation C — learned directional autoencoder

### Mechanism

The program crops each sklearn digit image from 8x8 to its upper-left 7x7,
shuffles once, uses 1,300 train and 497 test images, and learns a nine-dimensional
autoencoder. Each encoder code reads only one directional sign block; the decoder
is dense.

Saved held-out variance explained:

| Method | Saved result |
|---|---:|
| PCA-9 | 0.71808 |
| Learned directional, linear | 0.63569 |
| Learned directional, tanh | 0.62215 |
| Learned random partitions, four-layout mean | 0.62411 ± 0.00667 |
| Fixed directional means + least-squares decoder | 0.59557 |

### What holds

- Learning a block-local linear filter improves over fixed means by about
  `0.0401`.
- PCA is the correct optimal rank-nine linear reconstruction reference and
  remains clearly better by about `0.0824`.
- The saved source itself does not claim its directional-versus-random difference
  passes its stringent threshold.

### Crossed fresh control

The source uses one sequential global RNG: directional, tanh, and random
partitions receive different initial states. The fresh audit resets matched
initialization for every layout and crosses five model seeds with eight random
partitions:

| Measurement | Result |
|---|---:|
| Directional mean | 0.635693 |
| Random-layout mean | 0.615045 |
| Mean difference | +0.020648 |
| Directional wins | 30/40, 75% |
| Random layout range | 0.592738–0.645806 |

Directional locality is a decent upper-quartile layout in this small sample, but
it is not uniquely optimal: two of the eight random partitions beat it under
every matched model seed. Model initialization changes little after 4,000
full-batch epochs; partition choice dominates.

### Additional protocol boundaries

- The test set is centered using its own test mean. This is a mild transductive
  preprocessing leak and should use the training mean.
- Only four random layouts are used in the saved artifact.
- Random-layout standard deviation is not a standard error over independent
  dataset splits or model seeds.
- No weights, reconstructed examples, per-image errors, or repeated split
  results survive.
- The dense decoder can compensate for a poor local partition, so this tests a
  locality-constrained encoder plus dense reconstruction, not an end-to-end
  cube-local architecture.

### Classification

**Measured partial locality result.** Learning local filters helps; the specific
directional partition is competitive but not decisive; unconstrained PCA wins.

## Generation D — rotation weight tying

### Saved negative result

The source compares a 2-D C4 tied encoder, an untied directional encoder, and
random partitions. It does not use the 24-element 3-D cube group.

| Data | N | Tied | Untied | Difference |
|---|---:|---:|---:|---:|
| Oriented digits | 60 | 0.444 | 0.510 | -0.066 |
| Oriented digits | 1,300 | 0.563 | 0.636 | -0.072 |
| Isotropic fields | 60 | 0.887 | 0.901 | -0.014 |
| Isotropic fields | 1,300 | 0.909 | 0.920 | -0.011 |

The negative conclusion is honest for the implemented model: tying does not
help and hurts oriented digits.

### The model is not equivariant as a whole

The canonical core/face/corner encoder weights alone satisfy the expected
quarter-turn relation; a random probe gives equivariance error `8.08e-16`.
However:

- `benc` has nine independent biases rather than one bias per orbit;
- `Wdec` is a free dense 49x9 decoder;
- `bdec` is a free 49-vector; and
- none of these are tied under C4.

An illustrative random structural probe gives:

| Mapping | Equivariance error |
|---|---:|
| Tied encoder weights only | `8.08e-16` |
| Encoder after independent biases | 4.276 |
| Full mapping relative error with dense decoder | 1.768 |

These random values are not measurements of the saved trained model. They are a
counterexample showing that the source architecture does not guarantee the
property its name asserts.

### Parameter count correction

The advertised 13-versus-49 comparison counts only masked encoder weights.

| Model | Encoder weights | Encoder bias | Dense decoder | Decoder bias | Total |
|---|---:|---:|---:|---:|---:|
| Tied | 13 | 9 | 441 | 49 | 512 |
| Untied | 49 | 9 | 441 | 49 | 548 |

The whole-model reduction is 6.57%, not a 73% reduction.

### Other boundaries

- Tied and untied use one training seed.
- The random baseline averages only three partitions.
- The PCA line uses all 1,300 training rows even for small-N comparisons.
- No trained weights survive, so actual trained equivariance cannot be measured.

### Classification

**Measured negative result from a partially tied encoder.** Preserve the orbit
map and weight-gradient tying idea. Do not call the whole autoencoder
rotation-equivariant.

## Generation E — geometry-direct graph denoising

### Mechanism

The source builds the ordinary six-neighbor graph Laplacian of a 7x7x7 grid and
defines source fields by:

```text
x = (I + 5L)^-1 g
```

It then fits reports with an iteratively reweighted approximation to:

```text
min_x Σ_i |x_i - report_i| + 5 x^T L x
```

This is standard robust graph-signal smoothing. It is useful engineering.

### Saved result

| Corruption | Naive relative error | Robust geometry relative error |
|---|---:|---:|
| 20% independent values replaced by `3N(0,1)` | 24.95 | 0.383 |
| 80% replacement | 49.21 | 1.168 |
| 27-cell corner block shifted by +4 | 20.78 | 1.669 |
| Alternative generated smooth field | — | 1.066 to original truth |

### Why this is not an error-correcting code

For the 343-cell graph:

| Boundary | Fresh result |
|---|---:|
| Laplacian rank | 342 |
| `I + 5L` rank | 343/343 |
| Generated family dimension | full |

Because `I + 5L` is invertible, every vector in `R^343` equals
`(I+5L)^-1 g` for some `g`. The generator is a full-dimensional distribution
with a smoothness bias, not a low-dimensional manifold, discrete codebook, or
redundant code with a minimum distance.

The decoder can prefer lower graph energy, but geometry alone does not identify
truth.

### Prior-match boundary

| Clean report | Relative decode error |
|---|---:|
| Source-distributed smooth field | 0.0163 |
| Unit-norm checkerboard field | 0.6373 |

No corruption is present in either check. The large checkerboard distortion
shows that performance depends on the truth matching the chosen smoothness
prior. The method does not “protect honest reports exactly.”

### Invalid social comparison

The alleged local “social median” is constructed as:

```python
np.where(block_mask, corrupted_report, true_value)
```

Outside the block, the report already equals truth; inside, the expression
chooses the corrupted report. The resulting array is byte-identical to the
naive report. No agents, votes, samples, region median, or social algorithm are
computed. Therefore the claim that geometry beats the social stack is not
tested.

The geometry result of 1.669 is much smaller than 20.78 but is still a 167%
relative error to the unit-norm truth, not accurate reconstruction.

### Global alternative

The decoder changes the alternative smooth field by 6.16% relative error. Its
distance to the original truth is 1.083; after decoding, error to original is
1.066. The qualitative identifiability lesson is valid: a prior cannot choose
between multiple prior-compatible explanations without external evidence. The
claim that only a social/anchor layer can supply that evidence is not established;
measurements, trusted sensors, temporal constraints, redundancy, or another
generative model are alternatives.

### Missing baselines

The experiment does not compare:

- ordinary L2/Tikhonov graph smoothing;
- Huber or total-variation graph denoising;
- median filtering;
- Gaussian filtering;
- robust splines/trend filtering;
- oracle corrupted-cell inpainting; or
- matched noise amplitudes relative to the unit-norm signal.

The replacement noise has standard deviation 3 while the entire true field has
norm 1, making naive relative errors of 25–49 intentionally extreme.

### Classification

**Verified robust graph-denoising prototype; truth/error-code narrative
retired.**

## Generation F — 3-D Om/LO polarity on SNLI

### Mechanism and saved result

The source:

- takes the first 60,000 official SNLI training rows;
- learns 1,500 PPMI-SVD word vectors in three dimensions;
- averages word vectors into premise `P` and hypothesis `H`;
- constructs four “Om” and three “LO” scalar features;
- trains logistic regression; and
- evaluates on the official SNLI test file.

Saved result:

| Representation | SNLI test accuracy |
|---|---:|
| Om features | 40.3196% |
| LO features | 40.5029% |
| Om + LO | 40.3909% |
| Separate premise/hypothesis bag-of-words | 62.4389% |

The source also computes a single shuffled-label control, but that value is not
stored in the JSON.

### LO is algebraically redundant

Om contains:

```text
cos(P,H), ||P||, ||H||, P.H
```

LO contains:

```text
||H-P||, cos(H-P,-P), cos(H,-P)
```

Every LO feature is exactly determined by Om:

```text
||H-P||² = ||P||² + ||H||² - 2 P.H
cos(H-P,-P) = (||P||² - P.H) / (||H-P|| ||P||)
cos(H,-P) = -cos(P,H)
```

Fresh numerical reconstruction across 1,000 random vector pairs has maximum
absolute error `3.75e-15`.

The combined classifier therefore cannot receive new information from the local
observer block. The near-identical 40.32/40.50/40.39 results are expected from
feature reparameterization, not evidence that a local observer was independently
tested.

### Evidence boundary

- About 40% shows predictive signal in norms, dot products, and cosine on this
  SNLI slice.
- It does not establish entailment understanding, causal observer-relative
  semantics, or a 3-D capacity law.
- Hypothesis-only, premise-only, length/norm-only, lexical-overlap, and standard
  cosine-feature baselines are absent.
- The historical hard-coded session path is invalid on the current machine,
  though matching local SNLI train/test files survive.
- No model, vocabulary, embedding, per-example prediction, configuration, or
  run log survives.
- The result is one seed and one 60,000-row prefix.

### Classification

**Historical diagnostic.** Preserve the compact feature formulation and its
negative LO boundary. Do not promote it as novel semantic geometry.

## Corrected claim ledger

| Historical claim | Status | Correct statement |
|---|---|---|
| CubeSokoban is full Sokoban solving | Falsified | It classifies identities of 40 random binary occupancy templates; no Sokoban mechanics exist |
| Canonicalized 100% proves learned unseen-rotation generalization | Retired as learning evidence | All canonical test inputs exactly match training inputs; hash/1NN controls also score 100% |
| The cube classifier understands structure rather than camera angle | Narrowed | Deterministic canonicalization removes the known rotation nuisance; no structural property generalizes across worlds |
| A plain model must see every exact angle | Falsified literally | Plain training on 23 rotations reaches 95% on the withheld rotation |
| The 24 cube rotations are correctly implemented | Verified engineering | 24 unique bijections and 0/576 closure failures |
| Directional region counts close exactly | Proven elementary identity | It is the binomial expansion of `(2m+1)^3` for odd cubes |
| Learned directional filters beat fixed means | Measured on one digit split | 0.6357 versus 0.5956 |
| Learned cube blocks decisively beat random blocks | Unsupported | Fresh matched control wins 30/40 but random layouts reach 0.6458 above directional 0.6357 |
| PCA is the optimal dense linear nine-code ceiling | Supported for squared reconstruction | Saved PCA 0.7181 exceeds every block-local run |
| Rotation-tied autoencoder is equivariant | Falsified structurally | Only encoder weights are tied; independent biases and dense decoder break the guarantee |
| Tying cuts the model from 49 to 13 parameters | Misleading partial count | Whole totals are 548 versus 512, a 6.57% reduction |
| Rotation tying helps on matching isotropic data | Negative saved result | Tied remains about 0.011–0.014 worse than untied |
| Smooth fields form a low-dimensional error-correcting manifold | Falsified | `I+5L` is full rank 343 and the generated family spans all `R^343` |
| Geometry protects honest reports exactly | Falsified generally | Clean smooth error is 0.016, while clean checkerboard error is 0.637 |
| Geometry beats a local social median | Not tested | The “social” array is exactly the naive corrupted report |
| Only an anchor can resolve a prior-compatible alternative | Overstated | External evidence is required; social anchors are only one possible source |
| Om/LO combined carries distinct local-observer information | Falsified algebraically | All three LO features are deterministic functions of the Om block |
| Om/LO 40.39% proves observer-relative meaning | Diagnostic only | Compact geometry features carry SNLI signal but trail bag-of-words by 22.05 points and lack artifact controls |

## What survives

### CG-01 — Exact group-orbit normalization

- **Preserve:** validated group actions, canonical representatives, orbit hashes,
  and transformation-aware nearest-neighbor lookup.
- **Boundary:** canonical test equality is invariance by construction, not model
  generalization.
- **Decision:** **use canonicalization when the nuisance group is exactly known
  and information lost on the quotient is irrelevant**.

### CG-02 — Odd-cube directional partition

- **Preserve:** negative/zero/positive axis groups and the exact region-size
  identity.
- **Boundary:** counting a partition does not establish optimal pooling.
- **Decision:** **treat sign blocks as one standard spatial locality option and
  benchmark them against grids, pyramids, wavelets, learned convolution, and
  random/optimized partitions**.

### CG-03 — Learned local filters

- **Preserve:** block-masked encoder with dense decoder and fixed-mean control.
- **Evidence:** learned directional 0.6357 versus fixed means 0.5956.
- **Boundary:** PCA wins; random layout distribution overlaps and sometimes
  exceeds directional.
- **Decision:** **cross layout, initialization, and data split; report the
  entire random-layout distribution**.

### CG-04 — Whole-map equivariance contract

- **Rule:** an equivariant autoencoder must constrain encoder weights, encoder
  biases, latent group action, decoder weights, and decoder biases.
- **Decision:** **test `f(gx)=g f(x)` numerically after training for every group
  element; weight tying in one layer is insufficient**.

### CG-05 — Robust graph denoising with prior-mismatch tests

- **Preserve:** graph Laplacian, IRLS robust residual, and smoothness logging.
- **Boundary:** call it denoising, not truth decoding or an error-correcting code.
- **Decision:** **benchmark against standard robust smoothers across corruption
  amplitude, topology, structured attacks, and out-of-prior clean signals**.

### CG-06 — Algebraic feature-independence preflight

- **Preserve:** derive whether a proposed observer feature is already a
  deterministic function of existing norms/dots/cosines.
- **Evidence:** the entire LO block reconstructs from Om to machine precision.
- **Decision:** **do symbolic/numerical redundancy checks before attributing a
  gain or null result to a new conceptual mechanism**.

## Minimum promotion experiments

### A real cube-symmetry learning test

1. Define a label shared across different worlds: connectivity, reachability,
   number of components, solvability, value, or action.
2. Split underlying world identities between train and test.
3. Cross both unseen worlds and unseen orientations.
4. Compare raw MLP/CNN, full augmentation, canonical hash/1NN, canonical learned
   model, group pooling, and a standard group-equivariant network.
5. Verify the full transformation contract and report compute/sample cost.
6. Promote only an advantage beyond exact orbit lookup on new task instances.

### A real Sokoban test

1. Represent walls, player, crates, goals, legal pushes, and deadlocks.
2. Rotate state and action consistently.
3. Use hashed level splits, not views of the same memorized level.
4. Compare BFS/A*, deadlock-aware search, GNN/value learning, augmentation, and
   equivariant variants.
5. Report solved levels, valid action rate, nodes, optimality gap, and time.

### A proper directional/equivariant autoencoder test

1. Use training-set centering for test data.
2. Cross at least five dataset splits, model seeds, and many layout seeds.
3. Match total parameter counts and optimizer budgets.
4. Tie the complete encoder/latent/decoder map.
5. Test actual equivariance error after training.
6. Compare convolution, wavelet/pyramid, PCA, dense AE, and group-CNN baselines.

### A proper graph-denoising test

1. Predeclare a signal class not defined by the same smoother.
2. Sweep signal bandwidth, corruption fraction, amplitude, and structure.
3. Add L2, Huber, TV, median, Gaussian, and oracle-mask baselines.
4. Separate reconstruction error, corruption detection, calibration, and
   identifiability.
5. Never call the output truth without external ground-truth evidence.

## Final classification

| Component | Classification | Keep active? |
|---|---|---|
| 24 cube rotation permutations | Verified engineering | Yes |
| Canonical orbit representative | Verified deterministic normalization | Yes |
| CubeSokoban 100% MLP headline | Tautological transformed-input overlap | Historical only |
| Sokoban solving | Missing / never evidenced | Open only if rebuilt as an actual puzzle task |
| Odd-cube directional counts | Proven elementary identity | Yes |
| Fixed directional pooling advantage | Prior S8 partial result | As a baseline |
| Learned directional autoencoder | Measured partial locality result | Yes, as scaffold |
| Cube-versus-random learned advantage | Unproven | Re-test |
| Rotation-tied autoencoder | Negative and not fully equivariant | Repair only if needed |
| Geometry-direct IRLS | Verified standard robust graph denoising | Yes, renamed |
| Error-correcting truth/anchor narrative | Retired | No |
| Om/LO SNLI geometry | Historical diagnostic with redundant LO | No active claim |

The Cube-and-Geometry family is now incorporated. No source, JSON, figure,
verdict, or root mirror was deleted, moved, renamed, or overwritten.
