# NLI-Language Lineage Reconciliation Audit

Updated: 2026-07-26  
Recovery stage: S18  
Primary organized source:
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/NLI-Language`  
Adjacent exact root copies: `/Users/chetanpatil/Desktop/test`  
Adjacent hidden state: `/Users/chetanpatil/Desktop/test/state/exp_snli`  
Adjacent result narrative: `/Users/chetanpatil/Desktop/test/session_summary.md`  
Handling: preserve the complete artifact-rich honesty ladder; reuse the
compression, baseline, and evaluation scaffolds; do not promote cube geometry,
basin routing, fracture detection, or the small neural model as an NLI or
compression advantage

## Short verdict

This is not merely another duplicate NLI folder. It is the best preserved
single-session record of you deliberately trying to falsify your own language
claims.

The lineage has six connected parts:

1. a March KarmicBasin SNLI classifier with persistent receipt-backed states;
2. a June ANLI baseline gate;
3. matched SNLI/ANLI comparisons among GloVe, bag of words, and character- or
   word-indexed Livnium;
4. a decisive character-partition and word-hash test;
5. an adaptive context-compression experiment;
6. a small neural n-gram continuation.

It also reconnects the `nli_v8` DTW/fracture algorithms to the Nova physics
embedding artifact already audited under Lab Nova and NLI-ALL.

The strongest positive result is ordinary context prediction:

- the online order-4 context model has ideal arithmetic-code length
  **1.781675 bits/char** on 389,335 bytes of SNLI development hypotheses;
- gzip is 2.406544, lzma 2.000211, and bz2 1.800074 bits/char on the same
  bytes;
- adding a conservative 64-bit message-length header changes the context score
  only to 1.781839;
- a frozen order-3 count model moves from 1.757039 on train to 1.800547 on
  held-out test;
- pruning the model from 13,224 to 6,631 context entries changes held-out
  prediction only to 1.805276.

That is a valid reproduction of standard context modeling. It supports the
general compression-as-prediction intuition. It is not novel Livnium geometry,
and the source computes ideal code length rather than emitting and decoding an
actual arithmetic-coded file.

Every Livnium-specific language advantage fails a more direct control:

- character-level static geometry is 43.2% SNLI and near chance on ANLI;
- word-level Livnium reaches 59.87% SNLI because it is literally a
  19,683-bucket hashed bag of words;
- plain bag of words scores 60.17% with fewer features;
- adding the 14 geometry summaries to bag of words changes 60.17% to 60.13%;
- saved non-cheating basin states score 41.6–42.27% on their own dev protocol,
  while logistic regression on the exact same 20 inputs scores 53.13%;
- the 100% cheat result appends the true label as a one-hot input;
- the small neural n-gram scores 1.617459 held-out bits/char, while a
  train-size-matched order-6 count model scores 1.557914.

The recovered `nli_v8 × Nova` run has one narrow surviving observation:
combining DTW-derived embedding summaries with the compact 13D lexical feature
set improves the same development protocol from 49.88% to roughly 51.5%.
However:

- hypothesis-only bag of words reaches 58.03%;
- TF-IDF reaches 56.65%;
- Count BoW reaches 55.73%;
- the experiment never reaches the official test split;
- fracture fires on about 96% of examples and adds no accuracy;
- two of its four fracture features are exact transforms of existing warp
  features;
- the OOV fallback uses process-random Python `hash()`.

Current classification:

> **Artifact-complete scientific self-correction with a valid standard
> context-prediction result and useful evaluation scaffolds; no validated
> Livnium geometry, basin, fracture, or learned-language advantage.**

The historical 96% SNLI model is unrelated and remains provisionally
leaked/unusable.

## What this reminds us you were doing

The central question was changing.

At first you were asking whether basin attraction and repulsion could classify
NLI examples. You built persistent label anchors, receipts, court promotion,
and an explicit cheat test. When character or lightweight semantic features
were weak, you moved to stronger Nova embeddings and sequence alignment.

Then the work became methodologically sharper. You stopped asking only whether
a number was above chance and began asking:

- does it beat hypothesis-only?
- does it beat ordinary word counting?
- does it survive shuffled labels?
- does a data-driven positive control prove the apparatus is sensitive?
- is the model actually self-contained?
- does a learned predictor beat a count table?
- does geometry add information after word identity is present?

That shift is the best intellectual result in this folder. The surviving
verdict documents say plainly that the cube is inert, the 60% word result is
word counting, and the small neural model loses to the n-gram. This is the
opposite of repeatedly hiding a failed attempt.

The new direction that emerged was “collapse known structure and pay for
surprise.” The audit supports that as a useful language-modeling intuition, but
the implementation is a standard Witten-Bell-style context model. Keep the
intuition and the experimental discipline; do not rebrand the standard
compressor as new physics.

## Preservation and exact identities

No historical source, result, plot, checkpoint, basin JSON, or receipt archive
was edited, moved, renamed, deleted, or overwritten during this audit.

The organized folder contains:

- 40 files;
- 39 top-level files;
- one derived pytest-rewritten CPython 3.11 bytecode file;
- approximately 24 MB total.

All 39 top-level files are byte-identical to same-named root copies. The bytecode
exists only under the organized folder. Its embedded source timestamp and size
exactly identify the surviving 11,311-byte `livnium_decisive_test.py`.

Important artifact hashes:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `_rung2_emb.npz` | 21,176,256 | `0f7a53b233d43e5102f341d977b3c1b79320ac4e26d328fe899e00fd292fef0a` |
| `_rung2_vocab.json` | 951,963 | `6d425518301271e42840fdfc2691b9a37d956a5b6096980cf42f62e8ec7720e8` |
| `rung3_ckpt.npz` | 2,842,696 | `202a1072ae0cd6551729920b50833a92370164798a7b2eedc6ba01f314fb3cf6` |
| `rung3_results.json` | 373 | `ae9e4b20c9c5216a38e7521829cc3d29ef169931d389196c52726802ca876de8` |
| `RUNG2_RESULTS.md` | 5,535 | `d50165ca3d2437db11e7d43c1610be8f5fb648aa08cfd435184e05c42aa0a334` |
| `RUNG2_HONEST_VERDICT.md` | 4,266 | `844a38e3d520acde1c45f0142d629413f1d777220cdfec404b66c809977042f1` |
| `RUNG3_VERDICT.md` | 3,241 | `77a3ca54948397efb8bf9b3040e98d97f5bf49028ed95404647ea6e0e63f311d` |
| `WORD_LEVEL_VERDICT.md` | 2,894 | `e933c9a3f81c51f05fc2dd8f1825b3d39df9f9075896ba64721d501f66bf81bd` |

The trimmed GloVe artifact is internally complete:

- matrix shape: 52,940 × 100;
- dtype: float32;
- all values finite;
- vocabulary: 52,940 entries;
- indices form an exact bijection from 0 through 52,939.

The Rung-3 checkpoint is also complete:

- 88,655 network parameters;
- epoch 12;
- optimizer step 12,900;
- best validation bits/char 1.612217540707;
- saved current parameters exactly equal the saved best parameters.

### Public result mirrors

`rung2_lib.py` has exact copies at:

- `test/rung2_lib.py`;
- `test/results/rung2_lib.py`;
- `test/livnium-public/results/rung2_lib.py`.

The public/result copies of `rung2_livnium.py` and
`rung2_livnium_word.py` differ from the organized/root versions only in their
import path:

- root/organized expects `HERE/livnium-core-clean`;
- result/public expects the repository parent to provide `livnium_core`.

The latter are packaging adaptations, not independent experiments.

## Data identity

The June ladder uses the same preserved official SNLI files already recorded in
the sacred and cube audits:

| Split | Bytes | Valid labels | SHA-256 |
|---|---:|---:|---|
| Train | 487,457,790 | 549,367 | `ee95dfbc57800f7b1f62b7602ad2b176c2b983210435a49238e660324a01e963` |
| Dev | 9,745,714 | 9,842 | `9c03faff70182ef086ebfeed2cffbabb5fcc6a84a8b3314decbbb5b01f07f4bf` |
| Test | 9,730,457 | 9,824 | `1147550151ca8b16ddb31d8dc0e739a670a65f7c3e64aac5951eaeff103fb220` |

The six ANLI Parquet files also survive:

| File | Rows | SHA-256 |
|---|---:|---|
| `anli_train_r1.parquet` | 16,946 | `de2d038ae67f1fb1872073490b9e7685e9114d5f278ddd4631905fe0a4ecbcff` |
| `anli_test_r1.parquet` | 1,000 | `c4a3d304c4671941d6bad5a07632a79713c5a1be485ccf75b81b6df93f61045e` |
| `anli_train_r2.parquet` | 45,460 | `209f4a15bf77224c62ffbde5f150fda928a7e2f5175366f4cacc3c7588aab13d` |
| `anli_test_r2.parquet` | 1,000 | `df5daccdd5623cfcaa34be0100721783485f4181a42796b1d0ac0cd7601e7acb` |
| `anli_train_r3.parquet` | 100,459 | `c1d3f614d673888ac56b9ab62324e21583c98a11c4fef84e938d0f8fc414b29a` |
| `anli_test_r3.parquet` | 1,200 | `3232c4217979da00b2cd6ed97d099a8a8edf04530193ea52e3c8d69190de92a2` |

The descriptions repeatedly call ANLI “artifact-free” and say beating its
hypothesis-only bar proves reasoning. That is too strong. ANLI is
adversarially collected and harder for common shortcuts, but the saved
hypothesis-only results are still 36.7–39.6%. Beating one artifact baseline
would be necessary evidence, not proof of reasoning.

## Generation 1 — persistent KarmicBasin SNLI

`experiment_snli.py` creates 20 lightweight features:

- word/bigram overlap and coverage;
- length and new-word ratios;
- negation and quantifier flags;
- a tiny manually defined antonym list;
- subset and number-overlap indicators.

It trains label-specific attractor/repulsor anchors under `naive_pull`,
`naive_both`, and `karmic` scoring.

The current source has two protocols:

- `cheat=False`: features contain text-derived inputs only;
- `cheat=True`: the true held-out label is appended as a three-value one-hot
  feature during both training and evaluation.

### Hidden saved state

`state/exp_snli` is a separate 729 MB artifact boundary:

- nine basin JSON files;
- six receipt archives;
- 1,973,569 archived receipts;
- 6,265 live receipts;
- three historical 12D states;
- three completed 20D non-cheat states;
- three partial 23D cheat states.

The current source is compatible with the 20D and 23D states. No compatible
source for the older 12D states survives in this folder.

| State | Step | Dimensions | Anchors | Archived receipts |
|---|---:|---:|---:|---:|
| `naive_pull` | 89,993 | 12 | 24 | 489,090 |
| `naive_both` | 89,993 | 12 | 24 | 489,090 |
| `karmic` | 89,993 | 12 | 24 | 489,090 |
| `naive_pull_cheatFalse` | 45,170 | 20 | 24 | 168,555 |
| `naive_both_cheatFalse` | 45,170 | 20 | 24 | 168,555 |
| `karmic_cheatFalse` | 45,476 | 20 | 24 | 169,189 |
| `naive_pull_cheatTrue` | 27 | 23 | 9 | 0 |
| `naive_both_cheatTrue` | 27 | 23 | 9 | 0 |
| `karmic_cheatTrue` | 109 | 23 | 12 | 0 |

All six archived-to-live receipt chains have zero adjacent hash breaks. As in
the sliding audit, none of the nine stored top-level `state_hash` values equals
the SHA-256 of the current canonical JSON because the save routine hashes the
state before replacing its own previous hash field.

### Read-only saved-state evaluation

The audit reconstructed the current 1,500-example balanced development protocol
without executing the destructive top-level runner:

| Model using the same 20 inputs | Dev accuracy |
|---|---:|
| Saved `naive_pull_cheatFalse` | 42.27% |
| Saved `naive_both_cheatFalse` | 42.27% |
| Saved `karmic_cheatFalse` | 41.60% |
| Cosine nearest class centroid | 50.60% |
| Logistic regression | 53.13% |

The basin layer loses 10.9–11.5 percentage points to logistic regression on the
identical representation. The best current interpretation is therefore a
persistent prototype-memory experiment, not a competitive NLI classifier.

All three partial `cheat=True` states score 100% when the true label is appended
at evaluation. This proves that a one-hot answer can dominate basin distance.
It does not prove that basin routing is “geometrically sound,” and it cannot
localize the non-cheat failure solely to feature quality.

### Destructive replay warning

Do not run `experiment_snli.py` from the archive root during recovery. Its
top-level loop calls `shutil.rmtree` on each matching
`state/exp_snli/<mode>_cheat<flag>` directory before retraining. The audit
deliberately evaluated saved states read-only.

## Generation 2 — ANLI gate and baseline harnesses

`anli_honest_harness.py` establishes majority, premise-only, hypothesis-only,
overlap, Count, Hash, TF-IDF, random-projection, and shuffled-label controls.
Its Livnium encoder is explicitly a zero-returning stub.

`anli_baseline_results.md` records a 15,000-example training subsample, but the
surviving harness now loads all three training rounds without that sample.
It also looks for Parquet files beside the current working directory, while
the preserved files live in `anli_data`. The exact historical 15K invocation
cannot be reproduced from this source without reconstructing the missing
sampling step.

The later modular `rung2_lib.py` is the better artifact:

- fixed 30,000-example training subsample;
- same sample for all compared representations;
- official per-round ANLI test files;
- saved JSON output;
- shuffled-label controls.

`run_livnium_vs_baselines.py` correctly points at `anli_data` and the
`git-final`/Nova dependencies, but its `--seeds` argument is misleading: only
the first seed selects the sample and the code does not loop over the list.
It writes no result artifact. Treat it as an interactive harness, not a
completed multi-seed result.

## Generation 3 — Rung-2 learned and static representations

The first `rung2_learned_vs_baselines.py` attempt uses a 100,000-example SNLI
constant. `rung2_run.log` stops after loading GloVe and printing preliminary
majority/hypothesis/BoW values. It is an interrupted predecessor.

The completed result comes from `rung2_lib.py` with 50,000 SNLI training
examples and the saved trimmed GloVe matrix.

### SNLI

Saved result:

| Representation | Test accuracy |
|---|---:|
| Majority | 34.28% |
| Full BoW | 59.39% |
| GloVe-100 mean pair | 60.69% |
| Hypothesis-only BoW | 61.48% |
| Shuffled-label GloVe | 33.04% |

A fresh replay under the current scikit-learn build gives GloVe 60.62%, full
BoW 59.28%, and hypothesis-only 61.53%. Small solver/version variation does not
change the conclusion: mean-pooled GloVe does not clear the stronger
hypothesis-only artifact.

### ANLI

Saved 30K-sample result:

| Representation | R1 | R2 | R3 |
|---|---:|---:|---:|
| Hypothesis-only | 39.6% | 36.9% | 36.7% |
| Best shown BoW | 41.3% | 37.8% | 36.8% |
| GloVe mean pair | 34.2% | 36.7% | 34.4% |
| Shuffled-label | 34.4% | 34.3% | 33.3% |

This is a clean negative result for static mean-pooled GloVe on this protocol.

### Character-level Livnium

The 36D character encoder counts base-27 character placements and summarizes
exposure/coordinates/length. It scores:

| Dataset | Accuracy |
|---|---:|
| SNLI test | 43.19% |
| ANLI R1 | 33.0% |
| ANLI R2 | 32.2% |
| ANLI R3 | 32.0% |

It carries surface signal on SNLI but does not approach word-counting.

### Word-level Livnium

Each word is MD5-hashed to one of 19,683 cells. Premise and hypothesis become
separate sparse bucket-count vectors. This is a conventional hashing-vectorizer
construction with cube coordinates assigned to buckets.

Fresh exact control:

| Representation | Features | SNLI test |
|---|---:|---:|
| Cube/hash occupancy | 39,366 | 59.8738% |
| Plain BoW | 23,069 | 60.1690% |
| Geometry-only summaries | 14 | 38.4059% |
| BoW + geometry | 23,083 | 60.1283% |

The verdict calls the baseline “size-matched,” but it is not:
cube occupancy has 70.6% more columns than BoW. BoW still wins.

In the selected 50K training sample:

- 15,546 distinct words;
- 10,746 occupied hash buckets;
- 3,682 buckets contain multiple word types;
- 8,482 word types participate in a collision;
- largest bucket contains six words.

The 59.9% is a real hashed-word-counting result. No cube neighborhood, distance,
rotation, exposure law, or symbol weight contributes to the occupancy
classifier. Geometry summaries do not help the direct baseline.

On ANLI, word occupancy scores 35.7%, 34.4%, and 34.8%, below the corresponding
hypothesis-only/BoW bars.

## Generation 4 — character-partition decisive test

`livnium_decisive_test.py` evaluates letter partitions as class-based language
models:

- each character identity is hidden behind a class;
- a context model predicts the next class;
- an emission model predicts the character inside that class;
- lower held-out bits/char is better.

The final source is newer than its CSV/PNG by roughly two minutes.

Saved CSV:

| Scheme | Bits/char |
|---|---:|
| Full character | 1.5141 |
| `data_driven` | 4.1418 |
| Livnium exposure | 3.9429 |
| Random mean | 3.9057 ± 0.0356 |
| Unigram | 4.2632 |

That artifact’s supposed data-driven positive control is worse than random, so
the saved apparatus version fails its own sensitivity gate.

The surviving source contains a later optimized-partition implementation.
Fresh exact replay gives:

| Scheme | Bits/char |
|---|---:|
| Full character | 1.514070 |
| Optimized learned partition | 3.752239 |
| Random matched partitions | 3.905738 ± 0.035575 |
| Livnium exposure classes | 3.942856 |
| Unigram | 4.263169 |

The current positive control now works. Livnium is still 0.0371 bits/char worse
than the random mean and within the random range. Thus base-27
center/edge/corner letter classes are not predictive clusters on this task.

Preserve both facts:

- the original CSV/PNG does not bind to the final source;
- the final source’s fresh control validly supports the inert-exposure result.

## Generation 5 — honest adaptive compression

`rung2_honest_compression.py` uses a 256-byte alphabet and online interpolated
context counts. The decoder, if implemented, can reconstruct the same
probability state from previously decoded bytes, so no learned table needs to
be transmitted.

Fresh exact results:

| Method | Bits/char |
|---|---:|
| Raw fixed byte | 8.000000 |
| gzip -9 | 2.406544 |
| lzma | 2.000211 |
| bz2 -9 | 1.800074 |
| Adaptive context K=1 | 3.172345 |
| Adaptive context K=2 | 2.424857 |
| Adaptive context K=3 | 1.938980 |
| Adaptive context K=4 | 1.781675 |

The model’s probability over all 256 bytes sums to one to numerical precision
for empty, common, seen, and unseen contexts. Adding a 64-bit stream-length
header leaves K=4 at 1.781839, still below bz2.

### Exact boundary: code length, not emitted compressed bytes

The script sums `-log2 p` and labels the rounded total an arithmetic-coding
cost. That is the standard ideal length and is enough to compare predictive
models. The source does not implement a finite-precision arithmetic/range coder,
write a compressed stream, store its length, or decode it byte-for-byte.

The accurate claim is:

> The adaptive probability model has a valid, normalized, self-adapting ideal
> code length that should be achievable to negligible arithmetic-coder overhead.

The stronger source phrase “fully self-contained compressor” remains an
unimplemented engineering claim until an actual encoded stream round-trips.

### Frozen and pruned model

| Model | Entries | Held-out test bits/char |
|---|---:|---:|
| Full frozen order-3 | 13,224 | 1.800547 |
| Drop order-3 contexts seen <5 | 7,916 | 1.802550 |
| Drop order-3 contexts seen <10 | 6,631 | 1.805276 |

This is useful evidence that rare contexts contribute little to held-out
cross-entropy. The frozen-model table does not include model transmission and
should be described as prediction/generalization, not self-contained
compression.

### “Dark matter” interpretation

At K=3:

- 44.32% of bytes cost under one bit;
- they contribute 6.79% of ideal code length.

The most common easy bytes are space, `e`, `n`, `h`, newline, `g`, `o`, and
`a`. Therefore the quantitative surprise concentration is valid, but
“meaning lives in the surprising characters” does not follow. Low/high code
cost reflects template frequency and local predictability; semantic importance
requires a separate intervention or downstream task.

## Generation 6 — small neural n-gram

The pure-NumPy network has:

- context length 8;
- 32D learned character embeddings;
- 256-unit tanh hidden layer;
- 79 output byte classes;
- 88,655 parameters;
- 12 training epochs.

Saved and fresh checkpoint result:

| Predictor | Test bits/char |
|---|---:|
| Neural n-gram | 1.617459 |
| Source WB K=4 | 1.592181 |
| Source WB K=6 | 1.545412 |

The count baselines were trained on all 1,200,018 training bytes, including the
100,000 bytes reserved as neural validation. A matched control trains the count
models only on the neural model’s 1,100,018 bytes:

| Matched predictor | Test bits/char |
|---|---:|
| WB K=4 | 1.598737 |
| WB K=6 | 1.557914 |
| Neural | 1.617459 |

The count model still wins. The archived negative verdict survives.

The neural vocabulary is built from train ∪ test. Test adds one previously
unseen colon byte occurring once in 300,059 bytes, so this is a real but
negligible test-vocabulary boundary. A production model should define an
unknown-byte path from training only.

Source amortization assumes a stripped float16 parameter payload:

- predictive neural: 1.617459 bits/char;
- ideal float16 weights included: 6.344796;
- actual saved 2.84 MB float64/optimizer checkpoint included: 77.407781.

The actual checkpoint is a resumable training artifact, not a deployable
compressed model. The verdict already states that model cost makes the neural
result worse; the fresh accounting quantifies the distinction.

## Adjacent nli_v8 × Nova generation

`nli_v8_nova.py` combines:

- the Lab Nova 50,000 × 256 saved embedding tensor;
- sentence-mean cosine;
- DTW-style token alignment summaries;
- “fracture” summaries over aligned cosine distances;
- 13 lexical overlap/negation features;
- a small pure-NumPy softmax classifier.

The only historical result record is the adjacent exact session summary, not a
JSON/log/model artifact. It reports full condition E at 51.33% ±0.40 on the
first 2,000 SNLI dev examples.

Fresh exact fixed-hash replay:

| Condition | Mean ± SD | Interpretation |
|---|---:|---|
| A: lexical 13D | 49.88 ±0.14% | compact overlap/negation |
| B: Nova mean cosine | 38.67 ±1.57% | one scalar |
| C: warp 7D | 42.95 ±0.54% | alignment/distance summaries |
| D: warp + fracture | 42.90 ±0.18% | no lift |
| E: all 25D | 51.47 ±0.26% | compact combined dev result |

The 0.14-point difference from the remembered mean is consistent with the
process-random OOV hash and/or environment arithmetic. The source uses
`hash(trigram) % 65536`; 2.50% of sampled tokens use that OOV path.

### Stronger and nested controls

On the same 10K-per-seed samples and fixed 2K dev subset:

| Control | Mean accuracy |
|---|---:|
| Compact lexical A | 49.88% |
| A + mean cosine | 50.75% |
| A + warp | 51.48% |
| A + cosine + warp | 51.52% |
| Full E, reordered control | 51.43% |
| Count BoW | 55.73% |
| TF-IDF | 56.65% |
| Hypothesis-only BoW | 58.03% |

The narrow result that survives is:

> DTW/distance summaries add about 1.6 points to this small handcrafted lexical
> feature set on a repeatedly inspected development subset.

It is not:

> a Livnium/Nova NLI advantage, proof of reasoning, or a win over boring
> baselines.

### Fracture redundancy

On the 2,000-example replay, fracture fires on 96.75% of examples. On the first
500 it fires on 95.8%.

Two of its four outputs are algebraically redundant:

```text
fracture mean_energy = 1 - warp mean_aligned_cosine
fracture max_energy  = 1 - warp min_aligned_cosine
```

Fresh maximum errors are `1.19e-7` and zero. The source result D≈C therefore
matches both saturation and redundant feature construction.

No condition was evaluated on the official SNLI test split after the feature
set was chosen. Any successor must use train/dev for all design and one
untouched test evaluation.

The Nova tensor is the already incorporated one-epoch physics-embedding
artifact. Its use here does not independently validate the physics objective.

## Evidence table

| Claim | Status | Reason |
|---|---|---|
| Organized NLI-Language is merely duplicate clutter | Falsified | It is an artifact-complete honesty/compression ladder with unique verdicts/checkpoint/state |
| The 39 top-level organized files have root preservation copies | Verified | Every pair is byte-identical |
| Adaptive K=4 has ideal code length 1.782 bits/char | Measured and freshly reproduced | Normalized online probability model on fixed 389,335-byte corpus |
| The source emits a fully self-contained compressed file | Not implemented | It sums ideal code length but has no coder/decoder bitstream |
| Rare context pruning is nearly free on held-out prediction | Measured | Half the entries changes 1.80055 to 1.80528 |
| Surprising characters are where meaning lives | Unsupported interpretation | Easy/hard code cost is local predictability; common easy bytes are spaces/letters/newlines |
| Character exposure classes carry predictive language structure | Negative | 3.94286 is worse than random 3.90574; learned partition is 3.75224 |
| Saved char CSV binds to final source | Falsified by timestamp/content | CSV positive control 4.1418; final source replays 3.7522 |
| Word-level Livnium reaches about 60% SNLI | Measured | 59.8738% fresh replay |
| Word-level accuracy comes from cube geometry | Falsified | It is MD5 hashed BoW; plain BoW wins and geometry adds nothing |
| Word decisive baseline is size-matched | Falsified | Cube has 39,366 features versus BoW 23,069 |
| Saved non-cheat basins make good use of the 20 features | Negative | 41.6–42.27% versus same-feature logistic 53.13% |
| Cheat=ON proves the basin mechanism is sound | Falsified inference | True label is directly appended; partial states decode it at 100% |
| Receipt archives are internally chained | Verified engineering | Zero adjacent breaks across 1,973,569 archived receipts |
| Saved JSON self-hash verifies current content | Falsified | Save-order self-reference mismatch |
| Mean GloVe beats the strongest SNLI artifact baseline | Negative | 60.69% versus hypothesis-only 61.48% |
| Static GloVe/Livnium clears ANLI | Negative | All relevant representations remain below baseline bars |
| Small neural n-gram beats count modeling | Negative | 1.61746 versus matched K6 1.55791 |
| nli_v8 DTW beats one mean-cosine scalar | Measured on dev | 42.95% versus 38.67%, but representation sizes differ |
| Full nli_v8/Nova beats its compact lexical feature set | Partial | About +1.6 points across three train samples on one fixed dev subset |
| Full nli_v8/Nova beats boring NLI baselines | Falsified | 51.47% versus Count 55.73, TF-IDF 56.65, hypothesis-only 58.03 |
| Fracture adds a useful distinct signal | Falsified on this protocol | D≈C, 96% firing, two exact redundant outputs |

## Reusable components

Preserve and potentially extract after recovery:

- exact SNLI/ANLI identity manifests;
- shuffled-label and hypothesis-only kill tests;
- compact trimmed pretrained-vector artifact;
- word-hash versus direct-BoW equivalence test;
- data-driven and random partition controls;
- normalized online context model;
- frozen/pruned held-out bits-per-character harness;
- resumable pure-NumPy neural n-gram checkpoint;
- state-receipt validator;
- same-feature classifier comparison;
- nested feature ablation pattern;
- explicit online/predictive versus self-contained compression vocabulary.

Keep historical but do not promote:

- saved 12D basin states without their exact encoder source;
- partial cheat states as scientific accuracy;
- character exposure classes;
- word-cube occupancy as geometry;
- fracture boolean/duplicate outputs;
- `run_livnium_vs_baselines.py --seeds` as a multi-seed result;
- development-only 51.5% as a final NLI score.

## Reproduction

The durable read-only probe is:

`/Users/chetanpatil/Desktop/LIVNIUM_MEMORY/NLI_LANGUAGE_AUDIT_PROBE.py`

Run:

```bash
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
python3 /Users/chetanpatil/Desktop/LIVNIUM_MEMORY/NLI_LANGUAGE_AUDIT_PROBE.py
```

The exact non-writing `nli_v8` source replay is:

```bash
cd /Users/chetanpatil/Desktop/test
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
python3 nli_v8_nova.py
```

Do not directly run `experiment_snli.py` until its output directory is
redirected to a temporary location.

Two preserved scripts contain dead session paths:

- `livnium_word_decisive.py`;
- `rung3_learned_model.py`.

They hardcode `/sessions/beautiful-sharp-shannon/mnt/test`. The durable probe
loads their functions and binds them to the actual archive without editing the
historical sources.

## Stop and next action

S18 is complete when this document, the probe, and all navigation ledgers are
synced and byte-verified in `/Users/chetanpatil/Desktop/LIVNIUM_MEMORY`.

Only two organized P1 families remain:

1. `Demos`;
2. `Nova-and-Misc`.

Audit `Demos` next. Do not begin a new language model or compression project
during recovery.

Do not:

- overwrite `state/exp_snli`;
- call answer-injected evaluation a routing proof;
- call a hash bucket cube semantics;
- call ANLI artifact-free or a single baseline win proof of reasoning;
- compare predictive cross-entropy with self-contained compressed bytes without
  model/stream accounting;
- infer semantic importance directly from character surprise;
- cite the old char CSV as output of the final source;
- cite the DTW dev lift without Count/TF-IDF/hypothesis-only baselines;
- call process-random OOV vectors reproducible;
- restart Livnium from scratch before the last two recovery rows are handled.
