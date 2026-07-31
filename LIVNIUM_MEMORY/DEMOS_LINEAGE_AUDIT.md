# Demos Lineage Reconciliation Audit

Updated: 2026-07-26  
Recovery stage: S19  
Primary organized source:
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Demos`  
Adjacent exact root copies: `/Users/chetanpatil/Desktop/test`  
Adjacent saved state:
`/Users/chetanpatil/Desktop/test/state/basin_memory.json`  
Adjacent bridge dependency:
`/Users/chetanpatil/Desktop/test/nova_basin_store.py`  
Handling: preserve the six-script teaching and self-debugging sequence; retain
the standard base-27, prototype-learning, stream-memory, and receipt scaffolds;
do not promote the selected toy seed, random-walk puzzle, online game scores,
karmic law narrative, court gate, or receipt hash as stronger than their direct
controls and actual contracts

## Short verdict

This folder is not six hidden projects. It is a compact presentation layer over
mechanisms already spread across Livnium Core, Games, NLI-Language, and the
Nova bridge:

1. map the 27 lattice locations to a base-27 alphabet;
2. show how `BasinField` prototypes move;
3. print every simulated-annealing puzzle decision;
4. compare attraction and repulsion during online tic-tac-toe;
5. add authority, bad reputation, freshness, and independent pull/push caps;
6. persist those anchors with receipts and a nominal court.

The strongest new positive result is narrower than the prose but still useful:
**saved anchors improve later prequential stream performance over a matched
cold start on average**.

Across five paired protocols, each with a 300-game first session followed by a
300-game second session:

| Second-session condition | Wins | Losses | Draws | Win rate | Non-loss rate |
|---|---:|---:|---:|---:|---:|
| Warm, loaded prior anchors | 474 | 649 | 377 | 31.60% | 56.73% |
| Cold, same second-session seed | 279 | 810 | 411 | 18.60% | 46.00% |

Warm start raises wins by 13.0 percentage points and lowers losses by 10.73
points in aggregate. It wins the paired win-count comparison in four of five
seeds; one seed regresses from 93/116/91 cold to 66/234/0 warm.

This is a valid **prequential continuation** result:

- both conditions continue learning on the 300 games whose outcomes are
  reported;
- there is no separately frozen held-out game set in the source;
- the first-stage seed changes the learned policy;
- the source's single headline compares warm seed 99 with cold seed 42, rather
  than a same-seed cold control.

The exact source headline freshly replays in temporary storage:

| Run | W/L/D | Win rate | Draw rate |
|---|---:|---:|---:|
| Session 1, cold seed 42 | 83/130/87 | 27.7% | 29.0% |
| Session 2, warm seed 99 | 107/94/99 | 35.7% | 33.0% |
| Matched cold seed 99 | 34/220/46 | 11.3% | 15.3% |

So persistence carries task-relevant stream state. It does not establish a
strong frozen policy: the saved bridge state, evaluated without updates across
2,500 new games, scores 850 wins, 838 losses, and 812 draws. A direct symbolic
win/block/center/corner policy loses zero of 2,500 games and draws all of them.

The rest of the folder is mostly valuable as a set of failure boundaries:

- the base-27 integer codec is standard positional arithmetic and loses leading
  zero glyphs, including the lattice core marker;
- the selected “random start” clustering demo is already 100% accurate before
  training;
- the 25-move puzzle start is exactly three moves from solved, and the annealer
  still fails after 5,001 proposals;
- all reported game scores train on their evaluation stream;
- the Karmic demo never advances its freshness clock;
- `O_Win` bad karma, which is supposed to govern push, remains exactly zero;
- promotion has no effect on scoring, and promoted anchors can never later be
  quarantined;
- one nearest-anchor harm event is charged to every anchor in the label by the
  persistent wrapper;
- receipt hashes form an adjacent chain but cover anchor centers only, not most
  mutated metadata;
- the saved file reports 1,557 “total” receipts although 1,557 archived plus
  1,000 live receipts physically exist.

Current classification:

> **Useful teaching, persistence, and scientific-debugging sequence with a real
> prequential warm-start effect; no evidence for a superior game policy,
> autonomous law/court governance, reversible full lattice codec, or
> cryptographically complete mutation ledger.**

The historical 96% SNLI model is unrelated and remains provisionally
leaked/unusable.

## What this reminds us you were doing

These scripts capture the moment when several recurring Livnium ideas were
first placed beside observable task metrics.

The progression matters:

- `demo_learning.py` asks whether a basin visibly learns rather than merely
  routing.
- `demo_feedback.py` stops looking at win rate alone and logs drift, score
  jumps, path diversity, pull, and push.
- `demo_karma.py` separates positive authority from negative reputation and
  tries to prevent one signal from consuming the other.
- `demo_nova_bridge.py` makes memory survive a process boundary and attaches
  an inspectable receipt to mutation.

The conceptual ideas are respectable:

- positive evidence and negative evidence should not be one overloaded scalar;
- stale evidence should decay;
- harmful observations should be attributed locally rather than globally;
- pull and push need separate budgets;
- persistence should be tested against a matched cold start;
- a bounded live log should archive old receipts rather than grow forever.

The current implementation does not fulfill all of those intentions, but the
intentions are worth preserving. Later NLI and sliding-puzzle experiments reuse
the same Karmic and Nova-store machinery. Their failures make more sense after
reading these demos: the mechanics were born in an online game stream, never
established as a frozen decision advantage, and already contained the
authority/reputation mismatch and incomplete receipt contract.

## Preservation and exact identities

No historical source or state file was edited, moved, renamed, deleted, or
overwritten during this audit.

The organized folder contains six Python files totaling approximately 68 KB.
Every file is byte-identical to a same-named archive-root copy:

| File | Bytes | SHA-256 |
|---|---:|---|
| `demo_base27.py` | 4,033 | `b1494c778b61cff5c7af6fd81c98938cfba45a85a1210d00cc65d32b4914b38f` |
| `demo_feedback.py` | 14,440 | `8371994ff545205689be3cca8064c61053d30fce0df84c82ac46d954c37fddaa` |
| `demo_inside.py` | 4,748 | `13efe3567e6af14cc01bf72f6c8b3b16848d9af583d6c41684eeed3e1eebe56a` |
| `demo_karma.py` | 19,527 | `eb0d238cd154fc06a07853c231bf8af682b1483bb0e6d8de2f7c08ce1eca8e8d` |
| `demo_learning.py` | 4,285 | `5e3609e5e7e6df96c7eb6fc8adfae656c89b0f0df0fc41d5c2d04766a2b8073c` |
| `demo_nova_bridge.py` | 9,728 | `b79dab16304f685699cf3bef019ce9bf0ca6969c9be10f1fe059d5c918dffccc` |

The bridge depends on `nova_basin_store.py`, organized under the final
`Nova-and-Misc` family. Its root and organized copies are exact:

- bytes: 20,568;
- SHA-256:
  `07c535c5e81b9895e59c905120e0f292b2459a8b66c78524afa601efdb2e7d11`.

The adjacent saved bridge artifacts are unique:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `state/basin_memory.json` | 364,178 | `5b33cea4867762b0ad2981ec02f2d052169b40e485fae0396c865da7dda684eb` |
| `state/basin_memory.json.ledger_archive.jsonl` | 594,649 | `634840740d14886a2c7be96020ab794a17ecb3e73bd924fe20d2c20106beb71a` |

### Source/result chronology boundary

The saved state predates all three current bridge components:

| Artifact | Modification time |
|---|---|
| Saved bridge state | 2026-03-10 14:59:57 |
| `demo_nova_bridge.py` | 2026-03-10 15:11:47 |
| `nova_basin_store.py` | 2026-03-10 16:57:23 |
| `demo_karma.py` | 2026-03-10 17:05:52 |

Therefore the state cannot be claimed as output from the exact surviving source
hashes. The current source freshly reproduces the same deterministic W/L/D
headline and structural counts when run in a new temporary directory, but UUID,
timestamps, state hashes, and bytes differ. Preserve the saved state as an
earlier compatible artifact, not an exact final-source result.

All files in this lineage are untracked in the archive Git repository. There is
no Git history from which to recover the exact earlier producer versions.

## Demo 1 — base-27 lattice language

`demo_base27.py` assigns:

- `0` to the fixed center coordinate;
- `a` through `z` to the 26 outer coordinates;
- ordinary positional base-27 values to strings.

The valid part is straightforward:

- all 27 glyphs map bijectively to digits 0–26;
- a canonical cubic rotation fixes the center;
- the rotation permutes the 26 outer symbols;
- the rotation group contains the expected 24 proper cube rotations;
- the word `livnium`, which has no leading zero digit, round-trips through its
  integer and binary value.

The stronger “reversible language” wording needs a boundary. Positional integer
conversion identifies all leading-zero forms:

```text
""          -> 0 -> "0"
"00"        -> 0 -> "0"
"0a"        -> 1 -> "a"
"00livnium" -> 4790160436 -> "livnium"
```

The 27-character identity lattice begins with the core glyph:

```text
0abcdefghijklmnopqrstuvwxyz
```

Converting that complete state string to an integer and back returns only:

```text
abcdefghijklmnopqrstuvwxyz
```

The core position is lost. The script never performs a full state-string
integer round-trip, and it defines no string-to-`CanonicalState` decoder.
`int_to_string(-1)` also returns an empty string rather than rejecting a
negative input.

The correct interpretation is:

> A conventional base-27 numeral demonstration plus a readable fixed ordering
> of lattice symbols; digit-list/string conversion is reversible, but bare
> integer conversion is not a lossless codec for fixed-width lattice states.

The more complete `livnium/arith27.py` already separates digit/string operations
from canonicalized integer arithmetic. No new arithmetic project is needed.

## Demo 2 — prototype learning

`demo_learning.py` generates two easy 2D Gaussian clusters:

- A around `(0.6, 0.6)`;
- B around `(-0.6, -0.6)`;
- noise standard deviation 0.15.

All 50 A samples have both coordinates positive, and all 50 B samples have both
coordinates negative in the fixed draw. The direct rule
`A iff x + y > 0` scores 100%.

### Adversarial anchors

The source deliberately starts the two class anchors in the opposite clusters:

| Condition | Accuracy |
|---|---:|
| Before training | 0.0% |
| After eight epochs | 54.0% |

Correct anchors spawn, but wrong anchors persist and split routing. This is a
useful failure demonstration for prototype cleanup and initialization
sensitivity.

### “Random start” headline

The selected source seed reports 100% throughout, but its first printed row is
measured before any update:

| Condition | Accuracy |
|---|---:|
| Selected random anchors, epoch 0 | 100.0% |
| Selected random anchors, final | 100.0% |

It therefore does not demonstrate convergence over “3–4 epochs.” The random
labeled prototypes happen already to separate this extremely easy antipodal
dataset.

A fresh 100-initialization control is more informative:

| Metric | Before training | After eight epochs |
|---|---:|---:|
| Mean accuracy | 43.32% | 95.91% |
| Minimum | 0.0% | 72.0% |
| Maximum | 100.0% | 100.0% |
| Perfect runs | 32/100 | 61/100 |
| Runs below 90% | — | 17/100 |

Thus `BasinField` genuinely improves this toy prototype task on average, but it
does not converge reliably and it has no advantage over the trivial sign rule.
Preserve it as a prototype-mechanics teaching example, not a learning
breakthrough.

## Demo 3 — inside the annealing engine

`demo_inside.py` is a transparent trace of the same target-supplied 8-puzzle
annealing family incorporated under Games.

It claims a 25-step shuffle. Immediate reverse moves are allowed. The resulting
start is:

```text
1 2 3
4 8 5
7 _ 6
```

Exact control:

| Quantity | Result |
|---|---:|
| Nominal shuffle moves | 25 |
| Initial Manhattan distance | 3 |
| Exact shortest-path depth | 3 |
| Greedy Manhattan moves | 3 |
| Annealing proposals | 5,001 |
| Annealing accepted/rejected | 3,139 / 1,862 |
| Solved | No |
| Final Manhattan distance | 4 |

The printed trace is useful for explaining Metropolis acceptance, temperature,
energy differences, and rejection. It is negative algorithm evidence: the
generic stochastic search fails on a state solved exactly or greedily in three
moves.

The source exits after the loop without an explicit “FAILED” message. It prints
only totals, so a casual reader can miss that no solution occurred.

## Demo 4 — feedback coupling

`demo_feedback.py` compares random play, attraction to `X_Win`, and attraction
minus repulsion from `O_Win` against a heuristic O opponent. It trains after
every reported game, so all numbers are online/prequential.

Fresh exact seed-42 replay:

| Mode | Beta | W/L/D | Win rate | Logical interpretation |
|---|---:|---:|---:|---|
| Off | 0.00 | 8/423/69 | 1.6% | random X |
| Pull | 0.00 | 236/260/4 | 47.2% | aggressive, still loses 52% |
| Pull+push | 0.10 | 85/401/14 | 17.0% | worse |
| Pull+push | 0.25 | 135/142/223 | 27.0% | mixed |
| Pull+push | 0.50 | 1/26/473 | 0.2% | mostly draws |
| Pull+push | 1.00 | 0/9/491 | 0.0% | defensive collapse |
| Pull+push | 2.00 | 0/8/492 | 0.0% | defensive collapse |

This is not one monotonic “more feedback is better” curve. Beta changes the
implicit objective from trying to win to avoiding loss. The source ranks only
win rate in prose even though its high-beta policy obtains a much lower loss
rate.

The diagnostic expansion is useful but some metrics need care:

- `AnchorDrift` measures only anchors present at the same list index before and
  after an update;
- `EnergyJump` is a change in an ad hoc action score, not physical energy;
- `Diversity` hashes the complete token-identity state, not only the logical
  X/O board, so identity permutations can inflate it;
- the dead predecessor `run_eval` contains placeholder learning and is replaced
  by `run_proper`;
- the module executes the full 3,500-game sweep on import because it has no main
  guard.

The central lesson survives: attraction and repulsion must be evaluated on
separate outcome axes, not collapsed into an exciting single score.

## Demo 5 — Karmic field and LawController

`demo_karma.py` introduces:

- per-anchor positive authority;
- per-anchor bad karma;
- freshness decay for unused authority;
- nearest-anchor harm;
- partial redemption after good outcomes;
- separately capped pull and push;
- tiered push scaling.

These are conceptually distinct from the earlier governance/economy metaphors.
They are local memory-policy heuristics.

Fresh seed-42 source replay:

| Mode | W/L/D | Win rate | Draw rate |
|---|---:|---:|---:|
| Off | 8/423/69 | 1.6% | 13.8% |
| Naive pull | 236/260/4 | 47.2% | 0.8% |
| Naive both | 1/26/473 | 0.2% | 94.6% |
| Karmic | 149/203/148 | 29.8% | 29.6% |

The Karmic condition is a compromise between aggressive pull and defensive
push on this seed. It is not the best win or non-loss policy.

### Five-seed online control

| Mode | Wins | Losses | Draws | Win rate | Non-loss rate |
|---|---:|---:|---:|---:|---:|
| Off | 41 | 2,096 | 363 | 1.64% | 16.16% |
| Naive pull | 933 | 1,470 | 97 | 37.32% | 41.20% |
| Naive both | 465 | 1,082 | 953 | 18.60% | 56.72% |
| Karmic | 672 | 1,293 | 535 | 26.88% | 48.28% |

The naive-both result is extremely seed-sensitive: its five W/L/D rows range
from 0/114/386 and 1/26/473 to 232/263/5. Karmic is somewhat less extreme but
still ranges from 17.0% to 32.0% wins.

### Frozen policy control

Each field was trained for 500 games, then evaluated for 500 additional games
without any anchor update:

| Mode | Wins | Losses | Draws |
|---|---:|---:|---:|
| Off | 0 | 2,500 | 0 |
| Naive pull | 500 | 2,000 | 0 |
| Naive both | 0 | 1,000 | 1,500 |
| Karmic | 1,000 | 500 | 1,000 |
| Direct symbolic heuristic | 0 | 0 | 2,500 |

Karmic memory carries substantial frozen behavior, but the deterministic
win/block/center/corner policy still dominates the tested “do not lose”
objective.

### Mechanism/implementation mismatches

The intended laws are not all active:

1. `freshness_halflife` is never exercised in this demo because `field.tick()`
   is never called. The global freshness step is zero after all five runs.
2. On an X loss, `X_Win` receives bad karma and `O_Win` is reinforced.
   `LawController.push_scale()`, however, reads **`O_Win` bad karma**.
   That value remains exactly zero across all five fresh Karmic runs.
3. Push therefore stays at its hardcoded minimum scale 0.05 rather than being
   governed by earned bad reputation.
4. Bad karma on `X_Win`, often near one, is not read by the scorer.
5. The source's `energy_jump` compares every move score with the first move in
   the game, not with the previous move as its metric description states.
6. Naive modes use ordinary `update_correct`/`decay_incorrect`, so their
   authority and bad-karma metrics remain zero and are not comparable with the
   Karmic row.

The accurate surviving idea is:

> Separate positive evidence, negative evidence, age, and action budgets, but
> explicitly test that each state variable reaches the scorer it is intended
> to govern.

## Demo 6 — Nova persistence, receipts, and court

`demo_nova_bridge.py` wraps `KarmicBasinField` in `NovaBasinStore`, saves every
100 games, creates a second field/process-like session, reloads anchors, and
continues training.

### Persistence

Persistence is verified engineering:

- the current source reloads ten anchors;
- centers, authority, bad karma, counts, labels, IDs, and statuses are serialized;
- the second session begins with non-empty scoring state;
- paired warm continuation improves aggregate prequential performance.

It is not an exact process snapshot:

- the Karmic global freshness step is not stored;
- per-anchor last-touched freshness steps are not stored;
- authority and bad karma are rounded to six decimals;
- Python RNG state is not stored;
- UUIDs and timestamps are nondeterministic.

Immediate behavior is approximately preserved, but long-term continuation can
diverge from an uninterrupted in-memory run.

### Saved state

The state contains:

- 10 current anchors;
- 8 `O_Win` and 2 `X_Win`;
- 6 promoted and 4 provisional;
- no currently quarantined anchor;
- 1,442 current-anchor support events;
- 1,108 current-anchor harm records;
- step 2,124.

The two promoted X anchors are:

| ID | Support | Harm | Bad karma | Status |
|---|---:|---:|---:|---|
| `dfd35abd` | 190 | 554 | 1.000000 | promoted |
| `ea10f8ee` | 570 | 554 | 0.952794 | promoted |

Both have identical harm count because the persistent wrapper charges each
decay operation to every anchor under the label, even though
`KarmicBasinField.karmic_decay()` changes only the nearest one.

### Receipt counts

The physical ledger contains:

| Operation | Count |
|---|---:|
| Reinforce | 1,442 |
| Decay | 1,108 |
| Promote | 6 |
| Quarantine | 1 |
| **Total** | **2,557** |

Storage is correctly bounded to 1,000 live entries with 1,557 archived JSONL
entries. The saved `ledger_total_count` is 1,557, not 2,557. `_Store.save()`
increments it by overflow only. The demo prints this archive count as “total
receipts,” which is false.

No explicit `spawn` receipt exists even though the schema promises that every
spawn produces one. A newly created anchor is represented by its first
`reinforce` receipt.

### Receipt hash boundary

All 2,557 receipts form one adjacent center-hash chain with zero breaks.

However `_hash_now()` hashes only:

```text
label -> list of anchor center vectors
```

It excludes:

- authority;
- bad karma;
- support and harm counts;
- status;
- step;
- receipts;
- timestamps.

Consequences in the saved ledger:

- all 1,108 decay receipts have identical before/after hashes even though bad
  karma and counts change;
- all six promotions and the quarantine have identical before/after hashes;
- 189 reinforcements also leave the center hash unchanged while other metadata
  changes.

The chain is useful for detecting center-order/center-value discontinuities. It
is not a hash-verifiable record of every mutation.

The top-level state self-hash also does not equal the SHA-256 of the current
canonical JSON. As in the NLI and sliding stores, `_Store.save()` hashes the
object while it still contains its previous `state_hash`, then inserts the new
hash and serializes again.

### Court gate boundary

The court has two possible behavioral states:

- provisional and promoted anchors both pull identically;
- quarantined anchors cannot pull.

Promotion therefore changes only metadata, not scoring. A direct control gives
the same score `(12.0, 0.0, 12.0)` before and after changing one anchor from
provisional to promoted.

More seriously, `_maybe_promote()` begins:

```python
if status == "promoted":
    return
```

Once promoted, an anchor can never be reconsidered for quarantine. A fresh
control promotes an anchor after six supports, applies 30 harms until bad karma
is 1.0 and harm/support is 5, and the final status remains promoted.

The saved X anchors show the real consequence: both are promoted despite bad
karma near one and hundreds of harms.

One quarantine receipt in the historical ledger indicates that a provisional
anchor temporarily crossed the rule. The final state has no quarantined anchor;
the status may later have been promoted as its support ratio changed.

### Destructive replay warning

Do not run `demo_nova_bridge.py` from the archive root during recovery. Its
top-level code:

- deletes `state/basin_memory.json`;
- deletes the lock file;
- does **not** delete the JSONL archive;
- immediately retrains two sessions.

A second run can therefore combine a fresh JSON state with an old archive and
make the receipt history misleading. The audit runs the exact source only
inside new temporary directories.

## Relationship to incorporated lineages

This family does not reopen earlier decisions:

- base-27 arithmetic belongs to Livnium Core/Realcore;
- cube rotation belongs to the verified canonical group math;
- the puzzle trace belongs to the Games search-demo/random-walk-depth boundary;
- online versus frozen tic-tac-toe belongs to Games;
- Karmic/Nova state reused by sliding memory is already negative against greedy
  Manhattan;
- Karmic/Nova state reused by SNLI is already negative against same-feature
  logistic regression;
- receipt-chain versus self-hash is already a recurring storage contract;
- `nova_basin_store.py`, `experiment_modes.py`, and the remaining saved
  comparison states will be reconciled fully in S20 `Nova-and-Misc`.

The Demos family adds the missing chronological and pedagogical bridge among
those lineages.

## Evidence table

| Claim | Status | Reason |
|---|---|---|
| The six organized demos are unique projects | Falsified | They are exact root mirrors and presentation scripts over existing Core/Games/Nova mechanisms |
| Base-27 word conversion works | Verified engineering | Ordinary positional numeral conversion on canonical strings |
| Integer/binary conversion reversibly stores a full fixed-width lattice string | Falsified | Leading zero/core glyph is discarded and no state decoder exists |
| Cubic rotation fixes the core and permutes outer symbols | Verified engineering | Fresh 24-rotation group check and multiset preservation |
| Random-start BasinField converges from scratch in the shown run | Falsified for the chosen run | It is already 100% before training |
| BasinField can improve the toy clusters across initialization | Measured | Mean 43.32% to 95.91% across 100 initializations |
| The toy shows a basin advantage | Falsified | Direct sign rule is 100%; 17/100 trained basin runs remain below 90% |
| A 25-step puzzle trace demonstrates deep search | Falsified | Exact depth and Manhattan are three |
| The traced annealer solves the puzzle | Falsified | It ends unsolved after 5,001 proposals; greedy solves in three |
| Pull improves online wins over random X | Measured on training stream | Seed 42: 47.2% versus 1.6%; five-seed mean 37.32% |
| Pull+push gives one stable improvement | Falsified | Beta changes objective and outcomes are strongly seed-sensitive |
| Karmic is best among source modes | Falsified | It is between naive pull and naive both depending win/non-loss objective |
| Karmic is a strong frozen policy | Negative | 500 losses/2,500; direct symbolic control loses zero |
| Freshness affects the Karmic demo | Falsified | Global freshness step remains zero |
| Earned O bad reputation governs push | Falsified | `O_Win` bad karma remains zero; minimum push scale is always used |
| Source `energy_jump` is consecutive score change | Falsified | It compares each score with the game's first score |
| Anchors persist between bridge sessions | Verified engineering | Ten anchors reload and influence the next stream |
| Warm persistence improves matched cold continuation | Partial positive | Aggregate paired lift, but one seed regresses and all evaluation is prequential |
| Saved bridge policy beats a direct tic-tac-toe policy | Falsified | Frozen state loses 838/2,500; symbolic heuristic loses zero |
| Every mutation has a hash-verifiable receipt | Narrowed | Adjacent center-hash chain is intact; most metadata mutations are outside the hash |
| Printed total receipt count is correct | Falsified | 1,557 is archive count; physical total is 2,557 |
| Harm is attributed only to the responsible nearest anchor | Falsified in wrapper | every label anchor receives harm count and receipt |
| Promotion gates pull | Falsified | provisional and promoted scores are identical |
| Promoted anchors can later be quarantined | Falsified | early return makes promotion permanent |
| Saved state binds to final source | Falsified by chronology | state predates bridge, store, and Karmic source revisions |

## Reusable components and lessons

Preserve and potentially extract after recovery:

- digit-list versus canonical-integer codec distinction;
- transparent Metropolis decision trace;
- 100-initialization prototype-control pattern;
- swapped-anchor adversarial initialization;
- separate win, loss, and draw reporting;
- separate positive evidence, negative evidence, freshness, and budgets;
- paired warm/cold continuation protocol;
- frozen-after-stream policy evaluation;
- bounded live ledger plus archive sidecar;
- explicit hash-coverage declaration;
- temporary-directory replay for destructive historical scripts.

Keep historical but do not promote:

- base-27 as a new mathematical discovery;
- integer encoding as a lossless state codec;
- selected 100%-before-training clustering run;
- nominal random-walk shuffle length as puzzle depth;
- target-supplied puzzle annealing as algorithm learning;
- online game outcomes as held-out accuracy;
- Karmic freshness or bad-reputation gating in this source;
- court promotion as earned scoring authority;
- full-mutation cryptographic receipt language;
- the saved state as exact output of final source.

## Reproduction

The durable read-only probe is:

`/Users/chetanpatil/Desktop/LIVNIUM_MEMORY/DEMOS_AUDIT_PROBE.py`

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 /Users/chetanpatil/Desktop/LIVNIUM_MEMORY/DEMOS_AUDIT_PROBE.py
```

The probe:

- reads all historical sources and the saved bridge state;
- runs non-persistent demos without writing;
- reconstructs unsafe import-time demos from their definition prefix;
- performs every bridge/persistence replay in `TemporaryDirectory`;
- never deletes or overwrites the archive state.

## Closure handoff

S19 is complete when this document, the probe, and all navigation ledgers are
synced and byte-verified in `/Users/chetanpatil/Desktop/LIVNIUM_MEMORY`.

At the time of this audit, `Nova-and-Misc` was the last open organized P1
family. It is now incorporated in `NOVA_MISC_AUDIT.md`, including the final
`nova_basin_store.py` ownership boundary, comparison states, improvement demo,
evaluation/gradient scripts, landscape plot, and displaced ablation files.

Do not:

- run `demo_nova_bridge.py` from the archive root;
- delete or regenerate `state/basin_memory.json`;
- call leading-zero-losing integer conversion a fixed-width state codec;
- cite the chosen learning seed without its 100% epoch-zero row;
- call nominal shuffle length puzzle depth;
- cite online tic-tac-toe as frozen generalization;
- call inactive freshness or zero-valued O bad karma adaptive law control;
- interpret promotion as a scoring gate;
- claim that center-only hashes protect authority, bad karma, counts, status,
  step, or ledger contents;
- restart Livnium before the final `Nova-and-Misc` recovery row is handled.
