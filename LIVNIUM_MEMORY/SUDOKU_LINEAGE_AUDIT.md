# Sudoku Lineage Audit

**Session:** S12  
**Audit date:** 2026-07-26  
**Historical source tree:** `/Users/chetanpatil/Desktop/test`  
**Canonical reading copy:**  
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Sudoku`  
**Historical root mirror:** `/Users/chetanpatil/Desktop/test`  
**Reusable independent probe:** `SUDOKU_AUDIT_PROBE.py` in this memory

## Executive verdict

The recovered Sudoku family is one lineage with five implementations:

1. an original 9x9 learned predictor wrapped in a hard Sudoku legality mask;
2. a stricter pure 9x9 rollout with no legality mask or search;
3. a conventional propagation/MRV/backtracking solver with a learned value
   ordering;
4. a tabular 4x4 Q-learner trained on three fixed puzzles; and
5. a 9x9 reward-trained linear policy.

The practical winner is the **hybrid solver**, because it returns independently
valid completions for every sampled board. Its success comes from conventional
Sudoku propagation and backtracking. The learned ordering is not a demonstrated
general advantage: on the exact 45-puzzle replay it is worse on easy and hard,
nearly tied on medium, and better in the expert mean, but no difficulty has a
significant paired win. A standard least-constraining-value ordering also beats
the learned expert mean.

The most scientifically interesting member is the **pure learner**, because it
really is tested on separate generated solution boards and does roll out without
a legality mask or search. Its best saved run reaches 43.36% unseen per-cell
accuracy and 62.5% exact-source completion on the easiest generated boards, but
only 14.5% on medium, 1.5% on hard, and 0% on expert. This is a partial local
prediction result, not a general Sudoku solver or evidence for new physics.

The original 85% result is **not fake**, but its narrative is overstated. Its
printed “held-out style” score is actually training-set resubstitution, its
claimed symmetry augmentation is never called, and inference explicitly uses a
hand-coded Sudoku legality mask. A fresh held-out replay still gives 53.22%
per-cell prediction, so a modest learned signal survives.

Both RL branches are retired as general-solving evidence:

- the 4x4 tabular learner memorizes three exact puzzles: a deterministic replay
  solves 3/3 training puzzles and 0/100 held-out puzzles;
- the 9x9 policy records zero solves throughout training and evaluation, and its
  saved example changes zero cells because a wrong greedy action leaves the
  state unchanged and is repeated until timeout.

The deepest benchmark problem across the whole family is puzzle construction.
Every “difficulty” is random clue deletion from a completed grid. Uniqueness is
not enforced, solver-based difficulty is not measured, and exact equality to the
generating completion can mark another valid completion wrong. In a fresh
50-puzzle check, multiple-solution rates rise from 20–24% at 51 givens to 100%
at 23 or 26 givens for both generators.

## Source identity and preservation

The organized Sudoku folder contains 26 files:

- 5 Python sources;
- 9 JSON result artifacts;
- 9 PNG figures; and
- 3 Markdown verdicts.

Every organized file has a same-named, byte-for-byte duplicate at the root of
`/Users/chetanpatil/Desktop/test`. This is one lineage copied twice, not two
independent replications. The organized folder is used as the canonical reading
copy because it keeps the family together; both historical copies remain
untouched.

### Integrity manifest

| File | Bytes | SHA-256 |
|---|---:|---|
| `SUDOKU_HYBRID_VERDICT.md` | 1,696 | `4e49acefc8aedfb5f93705e881ab5f096ff75ce1225f4ecf7d21b08867c908a3` |
| `SUDOKU_PURE_VERDICT.md` | 1,404 | `faddae57a21ef5d151199ec69d3dc3b046db5203f15fff994831ebfa2d0de158` |
| `SUDOKU_VERDICT.md` | 3,258 | `674a15a7d4677ceac7483d8d275b6eccbe9888581fcd6052f59978377859f853` |
| `livnium_sudoku.py` | 8,284 | `34dea88afb6d007c942a0d4e251c473293e3c397418fe0a7edd1e50303bdde39` |
| `livnium_sudoku.json` | 184 | `ab2cfb17248ab9d6bb66a3a49f7ead25377ae3c8034d864b5f238ac6c280d371` |
| `livnium_sudoku.png` | 67,665 | `e1ba3fc459ed219cfe04e98c4e0b5107959a1ea554408727d91cbc2062fa9bce` |
| `livnium_sudoku_pure.py` | 12,448 | `a14d89a91ad913bf78f4f1a74b4771d7f43dd48a654398bd3711c5dad3fa09e5` |
| `livnium_sudoku_pure.json` | 920 | `ea3e1909a4236284b8c352f9a4e78663117641b86447f3d2392da3526d94e257` |
| `livnium_sudoku_pure.png` | 61,312 | `8493c7ab8236e7033aef3f31f794e13307673bde40ffa11d999a9f914bf9a320` |
| `livnium_sudoku_pure_1200.json` | 1,125 | `7e9f3b7808faa9b784f1c1594592f4b6d67973e5d5a45ccfcb98fb16fe649836` |
| `livnium_sudoku_pure_1200.png` | 61,367 | `770f9ff8602f55a1550f35540b5955fbd15eb6483e084a36fb4c2f46e7ed5088` |
| `livnium_sudoku_pure_1200_pkg.json` | 1,156 | `d10993cac55ce9d687dd6c20437b546733850e765e1b96c28f1c7a4abbe7e8a3` |
| `livnium_sudoku_pure_1200_pkg.png` | 61,367 | `ba311a926502d2a5b5e662d71be387459f9240c258c99c165a8f216c012183e9` |
| `livnium_sudoku_pure_pkg_smoke.json` | 1,108 | `205eb80b1d20c69e3963cd490c099db260e0039ad2da674311ae369709dfc95f` |
| `livnium_sudoku_pure_pkg_smoke.png` | 61,246 | `3bda8db8a65d78704c813f5165107e8248a53874dc6551728c605e86cc99c029` |
| `livnium_sudoku_hybrid.py` | 7,545 | `f622e3bf75c18f476a14bb6dc6168c68f01e02e6b73eb56ae6957e2e81e30193` |
| `livnium_sudoku_hybrid.json` | 545 | `56f1d27795adc699422b34f02c09b0e07d07ef09dd502bdb7524faca38fadd3e` |
| `livnium_sudoku_hybrid.png` | 34,886 | `8fba44ad06bea8d7933b001b5cc341cedefdbda64bd5aba0aa4adc47af6891fb` |
| `livnium_sudoku_rl.py` | 5,634 | `e8cf5098b28f322252cd89cc09f6738762e023cdc005d3aac2a0729899f0acaa` |
| `livnium_sudoku_rl.json` | 1,024 | `fbd8aeb406b1597beb839fe14d0594f063c8bebf13771d73e615241445a06035` |
| `livnium_sudoku_rl.png` | 42,515 | `8eb88b0fe40f9570403c67a85d93d2d6e9c1c2317dac3ab2fdd5af0e9bcd5b49` |
| `livnium_sudoku_policy_rl.py` | 13,704 | `beb20df6039f7b1005ea2cff901fed7cd72549c4b9f494f9fe39a15a16c429ef` |
| `livnium_sudoku_policy_rl_g60_e5000.json` | 4,534 | `a8a985f1eace7374d486f7733c9a0459914cc77b44750607b9c09cdcfd90a16c` |
| `livnium_sudoku_policy_rl_g60_e5000.png` | 35,142 | `a9302413c6479c52428560401a00ed0813d53ebf4907694583eddcd954feed45` |
| `livnium_sudoku_policy_rl_smoke_small.json` | 4,449 | `9406fa68a1fcd16ee662a1355fba6d3d2f939685557584b918580bf8fe8ee18a` |
| `livnium_sudoku_policy_rl_smoke_small.png` | 34,722 | `f7ae37448135cebf1b2cac3084ec3837b2e2a2eceb42737c5088e25ce084ea17` |

The `py-sudoku` dependency is locally available at version 2.0.0.

## Independent audit method

The reusable probe:

- imports historical modules without calling their `main` functions;
- uses `PYTHONDONTWRITEBYTECODE=1`;
- never writes into the historical source folder;
- independently validates completed rows, columns, boxes, and givens;
- counts puzzle solutions with a separate MRV solver capped at two;
- reproduces the hybrid model, puzzle sequence, and exact saved averages;
- distinguishes candidate attempts from actual failed branches;
- adds least-constraining-value and one-draw random orderings;
- performs paired win/tie/loss and bootstrap/sign-test analysis;
- replays the full 30,000-episode tabular RL training and tests 100 new puzzles;
- checks the saved policy trace against its starting puzzle and solution; and
- parses the source AST to verify claimed feature use.

Representative full replay:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 SUDOKU_AUDIT_PROBE.py \
  --sudoku-dir /Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Sudoku \
  --generation-samples 50 \
  --hybrid-samples 45 \
  --rl-held-out 100 \
  --rl-episodes 30000
```

The original-model check was enabled separately with
`--include-original-model`.

## Shared benchmark boundary: random deletion is not a Sudoku benchmark

All five programs construct a completed board first, then remove a fixed number
of cells at random. This creates a clue-count bucket, not a recognized
difficulty level. It does not test:

- uniqueness;
- logical technique depth;
- search-tree difficulty;
- distribution shift to published puzzles;
- clue minimality;
- symmetry or canonical equivalence; or
- robustness across puzzle generators.

### Fresh uniqueness and baseline check

Fifty new puzzles per clue count were generated independently for each of the
built-in backtracker and `py-sudoku`. The solution counter stops after finding a
second completion, so “multiple” means at least two.

| Givens | Historical label use | Built-in multiple | `py-sudoku` multiple |
|---:|---|---:|---:|
| 51 | original easy: 30 holes | 20% | 24% |
| 40 | pure/hybrid easy | 70% | 56% |
| 36 | original medium: 45 holes | 72% | 84% |
| 32 | pure/hybrid medium | 98% | 96% |
| 26 | original hard; pure/hybrid hard | 100% | 100% |
| 23 | pure/hybrid expert | 100% | 100% |

All 50 generated full solutions were byte-distinct in every generator/clue
bucket. The problem is therefore not exact duplicate boards; it is that random
clue deletion usually leaves multiple valid completions.

Exact equality with the generating completion is consequently not a clean
“solved Sudoku” metric. A fair evaluator must separately report:

1. grid validity;
2. givens preservation;
3. exact equality with a known completion, if scientifically relevant;
4. whether the puzzle has a unique completion; and
5. objective/search effort under a fixed solver protocol.

### How much is already exposed by local legality?

Before any learned model, a uniformly random currently legal digit has substantial
expected accuracy because the row/column/box masks remove many alternatives.

| Givens | Built-in random-legal expected accuracy | `py-sudoku` random-legal expected accuracy | Built-in initial naked singles |
|---:|---:|---:|---:|
| 51 | 69.96% | 71.53% | 45.27% |
| 40 | 49.87% | 50.80% | 18.05% |
| 36 | 42.73% | 42.47% | 10.71% |
| 32 | 35.45% | 36.40% | 4.45% |
| 26 | 28.12% | 27.83% | 1.05% |
| 23 | 24.66% | 25.15% | 0.41% |

This does not erase learned signal, but it changes the baseline. A model that sees
row/column/box presence must beat legal-candidate and standard CSP heuristics,
not a blind 1/9 digit guess.

## Generation A — original learned rule plus legality mask

### Mechanism

`livnium_sudoku.py` generates 60 completed boards, applies 20 random masks to
each, and trains a one-hidden-layer MLP from a 27-bit feature:

- 9 digits present in the row;
- 9 digits present in the column; and
- 9 digits present in the 3x3 box.

At inference, the program evaluates every empty cell, explicitly zeros the
probability of any digit that violates Sudoku legality, and commits the most
confident remaining cell/digit pair.

### Saved result

| Random deletion bucket | Exact generating completion | Cells equal to generating completion |
|---|---:|---:|
| 30 holes | 85% | 99.26% |
| 45 holes | 45% | 93.49% |
| 55 holes | 0% | 60.22% |

### What reproduces and what does not

The model is trained on 44,240 empty-cell examples. The source prints
`model.score(X, Y)` and calls it “held-out style”; this is training
resubstitution:

| Measurement | Cell accuracy |
|---|---:|
| Training resubstitution | 58.61% |
| Fresh 20-solution, 80-mask replay | 53.22% |
| Fresh replay after legality-masked argmax | 53.22% |

There is genuine but modest out-of-board prediction signal. The full rollout's
high easy score is then assisted by the explicit legality mask and by the large
amount of local information already present at 51 givens.

### Retired or narrowed statements

- **“No hand-coded Sudoku logic” is false.** Inference explicitly checks
  row/column/box legality.
- **“Symmetry augmentation” is false for the saved source.**
  `symmetry_variant` is defined but has zero call sites.
- **The symmetry helper itself is incomplete relative to the verdict.** The
  verdict names digit relabeling, row/column/band/stack permutations, and
  transpose; no such training augmentation is invoked.
- **“Learns naked and hidden singles” is unsupported.** The targets are simply
  source-solution digits for masked cells; there are no naked/hidden-single
  labels or technique-specific tests.
- **“Held-out style” is mislabeled.** The printed figure is the training score.
- **Difficulty is only random hole count.**
- **The hard-coded output path**
  `/sessions/beautiful-sharp-shannon/mnt/test` is not the current archive and
  would make direct replay unsafe or fail outside the original session.

### Classification

**Partial / historical.** Preserve it as a learned local digit-ranker plus a
hard legality mask. Do not cite it as a pure learned constraint system,
symmetry-augmented learner, or general Sudoku result.

## Generation B — pure 9x9 learned rollout

### Mechanism

`livnium_sudoku_pure.py` improves the experimental separation:

- completed solution boards are split into train, validation, and test lists;
- the feature expands to 54 dimensions by adding row, column, and box location;
- no legality mask is applied at inference;
- no backtracking or rescue is used; and
- the model fills its globally most-confident prediction even when it is
  illegal.

This is the cleanest test in the family of whether a local learned predictor can
survive its own rollout errors.

### Saved runs

| Artifact | Generator | Train/val/test boards | Test cell accuracy | Easy exact | Medium exact | Hard exact | Expert exact |
|---|---|---:|---:|---:|---:|---:|---:|
| `livnium_sudoku_pure.json` | unrecorded historical default | 60/15/20 | 40.04% | 55.0% | 10.0% | 0% | 0% |
| `livnium_sudoku_pure_1200.json` | built-in implied; generator omitted | 1200/150/200 | 38.94% | 54.5% | 11.0% | 0% | 0% |
| `livnium_sudoku_pure_1200_pkg.json` | `py-sudoku` | 1200/150/200 | 43.36% | 62.5% | 14.5% | 1.5% | 0% |
| `livnium_sudoku_pure_pkg_smoke.json` | `py-sudoku` | 30/10/10 | 38.36% | 90.0% | 10.0% | 0% | 0% |

The 90% easy smoke number has only ten test boards and is not the preferred
result. The 1,200/150/200 `py-sudoku` run is the strongest preserved pure
artifact.

### Honest interpretation

The 43.36% unseen cell accuracy is above the roughly 25–51% random-legal
expectation across the clue buckets, but the exact comparison is not recorded
on the same generated masks. It therefore suggests a modest local signal rather
than establishing the size of the advantage.

The rollout result is sharply difficulty-sensitive. The best run's average first
mistake is:

- 37.34 placements on easy;
- 22.51 on medium;
- 7.36 on hard; and
- 5.01 on expert.

Once a wrong value is committed, later features are computed from a corrupted
grid and there is no correction mechanism. That is a real and useful failure
mode.

### Metric and protocol cautions

- “Solved” means exact equality to the generating completion, not independent
  validity. Multiple-solution puzzles can make this an undercount.
- “Legal rate” tests only whether the proposed digit duplicates the current
  row/column/box at that moment. It does not certify that the accumulated final
  grid is globally valid.
- Location one-hots can learn generator- or traversal-specific board biases.
- Random clue count is not standard difficulty.
- The source does not record puzzle identities, puzzle hashes, package version,
  or model weights, so the saved JSON cannot reproduce its exact predictions.
- The source writes results to its own historical folder if `main` is called.
  The archive must remain read-only.

### Classification

**Measured partial result.** This is the best Livnium-like research signal in
the Sudoku family: a local learned policy generalizes somewhat across generated
boards and exposes its own rollout ceiling. It is not a complete Sudoku learner.

## Generation C — hybrid propagation, search, and learned ordering

### Mechanism

`livnium_sudoku_hybrid.py` uses:

1. conventional candidate sets;
2. repeated naked-single propagation;
3. minimum-remaining-values cell selection;
4. recursive backtracking; and
5. either ascending digits or MLP probability as candidate ordering.

Search completeness comes from the ordinary CSP algorithm, not from Livnium.
The learned component can only change which candidate is tried first.

### Exact deterministic replay

The independent replay reproduces all four saved source averages exactly:

| Bucket | Ascending attempts | Learned attempts | Learned minus ascending |
|---|---:|---:|---:|
| Easy, 40 givens | 1.3556 | 1.4667 | +0.1111 worse |
| Medium, 32 | 7.4889 | 7.2889 | -0.2000 better |
| Hard, 26 | 20.5778 | 25.3111 | +4.7333 worse |
| Expert, 23 | 43.2444 | 25.6889 | -17.5556 better |

Every replayed completion from ascending, learned, least-constraining-value, and
one-draw random order passed an independent validity and givens-preservation
check.

The source calls its counter “backtracks,” but it increments for every candidate
attempt, including candidates on the successful path. Corrected actual failed
branch means are:

| Bucket | Ascending failed branches | Learned failed branches | LCV failed branches |
|---|---:|---:|---:|
| Easy | 0.0667 | 0.2222 | 0.0444 |
| Medium | 1.6444 | 1.4667 | 1.4000 |
| Hard | 9.2222 | 13.8889 | 7.4667 |
| Expert | 27.0222 | 10.2222 | 5.9333 |

### Paired evidence

| Bucket | Learned wins | Ties | Ascending wins | Exact sign-test p | Bootstrap 95% CI for mean attempt reduction |
|---|---:|---:|---:|---:|---|
| Easy | 7 | 34 | 4 | 0.549 | [-0.60, 0.22] |
| Medium | 16 | 12 | 17 | 1.000 | [-0.69, 1.20] |
| Hard | 19 | 5 | 21 | 0.875 | [-17.67, 6.27] |
| Expert | 22 | 4 | 19 | 0.755 | [-7.93, 61.27] |

No bucket demonstrates a reliable paired learned-ordering advantage at ordinary
significance levels. The expert average is driven by the magnitude of a few
search trees, not by a consistent win rate. Its interval crosses zero broadly.

The expert learned mean of 25.69 attempts is also worse than the standard LCV
mean of 23.49. On hard, learned uses 25.31 attempts versus LCV 19.51. These are
single-seed comparisons, so they are a reason for a proper benchmark, not a
final ranking.

### Puzzle identity problem in the exact replay

At least two solutions exist for:

- 62.22% of the 45 easy puzzles;
- 97.78% of medium;
- 100% of hard; and
- 100% of expert.

This makes value-ordering performance especially sensitive to which valid
completion an ordering reaches first. It is not a benchmark on standard,
uniquely solvable “expert Sudoku.”

### Classification

**Verified standard engineering; learned advantage unproven.** The solver works
and its returned grids are valid. Preserve it as a clean scaffold for paired
heuristic tests. Retire “1.7x less search” as a general claim; retain only the
literal single-seed expert mean with its failed statistical boundary.

## Generation D — tabular 4x4 Q-learning

### Mechanism

`livnium_sudoku_rl.py` creates only three 4x4 puzzles with six holes each and
trains one Q-table for 30,000 episodes. The state key is the exact grid bytes.
The environment:

- rejects row/column/box violations;
- gives +0.3 for a currently legal placement;
- gives +10 for exact equality to the hidden generating solution;
- penalizes invalid or occupied actions; and
- never supplies a held-out evaluation in the historical program.

This is memorization of exact states under a hand-coded Sudoku environment, not
general from-reward rule induction.

### Full independent replay

| Measurement | Result |
|---|---:|
| Training episodes | 30,000 |
| Q-table states | 225 |
| Last 500-episode training solve rate | 99.8% |
| Greedy training puzzles solved | 3/3 |
| Held-out six-hole puzzles | 100 |
| Held-out initial states already in Q | 0/100 |
| Held-out greedy puzzles solved | 0/100 |

The class docstring claims 80 actions—64 placements plus 16 erasures—but
`nA=64`, action decoding implements placements only, and erasure does not exist.

The statement that the observation contains no constraint hint is narrowly true
for the raw grid. The transition and reward still teach the rule directly by
rejecting illegal row/column/box moves. The statement “pure RL, no solution
labels” is also too broad because successful completion is defined by exact
equality to a hidden solution.

### Classification

**Retired as generalization evidence.** Preserve as a tiny demonstration that a
tabular agent can memorize three 4x4 trajectories.

## Generation E — 9x9 reward-trained linear policy

### Mechanism

`livnium_sudoku_policy_rl.py` creates a 63-dimensional feature for every
cell/digit action:

- row/column/box digit-presence masks;
- row/column/box location one-hots; and
- candidate digit one-hot.

The environment compares every attempted digit to the hidden source solution:

- correct digit: +1 and the digit is written;
- wrong digit: -1 and the state is unchanged;
- complete exact source solution: +20.

This is direct target supervision delivered as reward. The training loop updates
after each action using an immediate `(reward - baseline) * gradient`. Although
`discounted_returns` exists and `gamma` is recorded, the training loop does not
use episodic returns or `gamma`; it is closer to a contextual-bandit policy
update than REINFORCE.

### Saved results

The 200-train/100-test, 5,000-episode `py-sudoku` run records:

- all 20 training-curve checkpoints equal to zero solves;
- 0% solve rate for easy, medium, hard, and expert;
- first mistake at about 1.22–1.36 actions; and
- tiny correct-action rates after percentage conversion.

The saved easy example has 41 holes and 123 evaluation steps, but the final
trace changes **zero cells** from the starting puzzle. The mechanism is
deterministic:

1. greedy argmax selects a wrong action;
2. the environment leaves the grid unchanged;
3. identical features produce the identical argmax;
4. the same wrong action repeats until `3 × holes` steps.

No weights are saved, so the historical policy itself cannot be replayed from
the JSON.

### Classification

**Retired negative result.** Preserve the failure as a valuable design lesson:
rejected deterministic actions require masking, exploration, state change,
penalty memory, or another anti-loop mechanism. Do not call this episodic
REINFORCE or general Sudoku learning.

## Corrected claim ledger

| Historical claim | Status | Correct statement |
|---|---|---|
| Original model learned with Sudoku symmetry augmentation | Falsified | A helper is defined but never called |
| Original evaluation is “held-out style” | Falsified | The printed score is training resubstitution |
| Original uses no hand-coded Sudoku logic | Falsified | Inference explicitly masks illegal digits |
| Original 85% easy result | Narrow measured artifact | 85% exact generating-completion match on 40 random 51-given boards, assisted by legality; puzzle uniqueness not enforced |
| Pure learner generalizes to unseen boards | Supported narrowly | Separate generated full-board splits exist; best saved unseen cell accuracy is 43.36% |
| Pure learner solves general Sudoku | Falsified | Best saved exact completion is 62.5/14.5/1.5/0% from easy to expert |
| Hybrid is a complete Sudoku solver | Verified as standard algorithm | All sampled completions are independently valid; completeness comes from exhaustive backtracking |
| Learned ordering cuts expert search 1.7x | Narrow literal artifact only | Exact single-seed means reproduce, but paired p=0.755, bootstrap CI crosses zero, and LCV has a lower expert mean |
| Hybrid counter measures backtracks | Falsified label | It counts candidate attempts; actual failed branches are lower |
| Tabular RL learned Sudoku from reward | Falsified as generalization | It memorizes three exact 4x4 puzzles and solves 0/100 unseen |
| Tabular agent can erase cells | Falsified | Only 64 placement actions exist |
| Policy RL is REINFORCE with discounted returns | Falsified | Training uses immediate reward-baseline updates; `gamma` is unused |
| Policy RL learns 9x9 Sudoku | Falsified | Training and test solve rates are zero; the saved example never changes a cell |
| Clue count equals easy/medium/hard/expert | Unsupported | It is only a random deletion bucket |

## What survives

### SUD-01 — Local candidate-context learner

- **Preserve:** row/column/box presence features and honest unseen-board split.
- **Evidence:** original fresh cell accuracy is 53.22%; strongest pure artifact
  is 43.36% on a much larger split.
- **Boundary:** compare against legal-candidate, frequency, location-only, and
  generator-bias baselines on the same masks.

### SUD-02 — Error-propagation experiment

- **Preserve:** the pure no-rescue rollout and first-mistake metric.
- **Evidence:** first mistake moves from about 37 on easy to 5 on expert, after
  which self-generated corruption compounds.
- **Boundary:** independently validate the final grid and distinguish target
  equality from any valid completion.

### SUD-03 — Learned value-ordering scaffold

- **Preserve:** paired puzzles, MRV search, independent validity check, and
  candidate-order injection point.
- **Evidence:** exact saved replay works.
- **Boundary:** compare against LCV, randomized ordering, activity-based search,
  DLX/Algorithm X, and multiple trained seeds on unique standard puzzles.

### SUD-04 — Rejected-action loop guard

- **Preserve:** the policy failure as a regression test.
- **Rule:** any environment where an invalid action leaves state unchanged must
  prevent deterministic reselection of the same action or terminate explicitly.

### SUD-05 — Puzzle identity contract

Every future Sudoku record must include:

- puzzle text or canonical grid;
- SHA-256;
- source corpus and license/provenance;
- uniqueness;
- recognized difficulty or solver-based difficulty features;
- seed and generator version;
- givens count;
- solver completion validity;
- givens preservation;
- exact-source equality only as a separate diagnostic;
- search nodes, candidate attempts, and actual failed branches with unambiguous
  names; and
- paired baseline results on the identical puzzles.

## Minimum promotion experiment

Do not rebuild another core. Reuse the hybrid injection point and run one clean
test:

1. select a fixed, hashed corpus of unique puzzles with recognized difficulty;
2. split puzzle identities, not just completed solution boards;
3. train the value-ordering model only on the training identities;
4. compare ascending, descending, random, LCV, and learned ordering on the exact
   same test puzzles;
5. validate every returned grid independently;
6. report attempts, failed branches, wall time, win/tie/loss, median, mean,
   bootstrap interval, and exact paired test;
7. repeat model training across at least five seeds; and
8. promote only if learned ordering wins consistently and beats LCV, not merely
   one weak ascending-digit baseline.

For the pure learner, keep a separate track:

1. predeclare cell-prediction and whole-board metrics;
2. add legal-random, smallest-legal, candidate-frequency, location-only, and
   generator-only controls;
3. evaluate valid completion independently of source-solution equality; and
4. measure calibration and first-error recovery, not just average cell equality.

## Final classification

| Component | Classification | Keep active? |
|---|---|---|
| Original MLP + legality | Partial historical learned heuristic | As provenance and baseline |
| Pure MLP rollout | Measured partial research result | Yes, as the clean learning question |
| Hybrid CSP solver | Verified standard engineering | Yes, as evaluation scaffold |
| Learned hybrid ordering claim | Unproven | Re-test before promotion |
| 4x4 tabular RL | Memorization / retired general claim | Keep only as negative control |
| 9x9 policy RL | Failed negative result | Keep as anti-loop lesson |
| Random-deletion “difficulty” benchmark | Invalid for general claims | Replace |

The Sudoku family is now incorporated. No source, result, figure, or verdict was
deleted, moved, renamed, or overwritten.
