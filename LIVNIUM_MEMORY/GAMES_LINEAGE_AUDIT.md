# Games Lineage Audit

Updated: 2026-07-26  
Recovery stage: S17  
Primary organized source:
`/Users/chetanpatil/Desktop/test/_ORGANIZED/02_Experiments/Games`  
Adjacent live project: `/Users/chetanpatil/Desktop/test/evaluate_chess`  
Adjacent saved state: `/Users/chetanpatil/Desktop/test/state/exp_sliding`  
Handling: preserve the complete lineage; reuse the chess state-transport
engineering and the evaluation lessons, but do not promote the game demos as
evidence that basin memory or Livnium search outperforms elementary game
algorithms

## Short verdict

This folder is four different kinds of work that happened to be grouped under
“Games”:

1. a careful bijective representation and move-transport layer for chess;
2. a hand-weighted chess mate-in-one ranker;
3. an online tic-tac-toe loss-memory experiment;
4. sliding-puzzle and sorting demonstrations driven by exact task objectives.

The best work is the improved chess transport project. It explicitly represents
pieces, side to move, castling rights, and en-passant state with conserved
symbols; it passes eight unit tests, 1,000 random transitions, 22 adversarial
transitions, and a new continuous-state replay with no state or symbol
conservation failures. That is real verified engineering.

Its boundary is equally important: Python chess still supplies legality and the
meaning of each move. The Livnium layer transports a move into a reversible
symbol state; it is not an independent chess rules engine. Move clocks are not
maintained, so the encoding is not a bijection over the complete FEN state.

The mate-in-one result is not evidence for basin intelligence. The ranker is a
fixed, manually weighted linear heuristic whose strongest inputs call ordinary
Python-chess rule functions. On the generated 100-puzzle set, the archived
ranker puts the intended mate first 84 times, while the elementary rule
“prefer check, then the fewest legal replies” finds the mate first in all 100.
Decoded and hybrid Livnium feature modes produce identical complete rankings.

The tic-tac-toe result is a useful online-learning observation, but the
evaluation games are also the training games. Across five seeds, freezing the
learned basins removes the dramatic near-perfect behavior: against the
heuristic opponent, the frozen agent wins 810, loses 729, and draws 961 of
2,500 games. It loses or draws every game against minimax. A simple symbolic
heuristic, with no basin memory, draws all 2,500 games against both the
heuristic and minimax opponents.

The sliding-memory experiment is a strong negative result. The “25-move”
starts have mean exact optimal depth 8.493 because the shuffle allows immediate
reversals. Exact search solves all 150 starts within the 300-step budget. In
the fresh source replay, greedy Manhattan without basin memory solves 74%,
while the three memory modes solve only 1–9%; carrying memory between sessions
makes every tested mode worse. The representation also aliases many puzzle
states, and cosine distance cannot attract any nonzero state to the all-zero
solved feature vector.

The sorting and one-off sliding demos show that `SwapSymbol`, `EnergyModel`,
and simulated annealing can optimize an explicitly supplied target. They do
not show learning. The sorting demo takes 1,345 annealing steps on a
permutation whose exact minimum is nine arbitrary swaps.

Current classification:

> **Preserve as verified chess state-transport engineering, instructive online
> and search experiments, and an unusually valuable negative result about
> state-only basin memory. Do not treat it as a validated game-playing,
> puzzle-solving, or learned-planning advantage.**

The historical 96% NLI model is unrelated to this family and remains
provisionally leaked/unusable.

## What this reminds us you were doing

On March 10 you were testing a concrete question: if Livnium’s primitive is a
bijective movement of named symbols, can a conventional discrete game be
expressed without destroying identity?

You began with chess board reconstruction, then quickly noticed that board
pieces alone were insufficient. The improved project moved side-to-move,
castling, and en-passant facts into the symbol state and added adversarial
sequences for the awkward moves. That was the rigorous part of the day.

You then asked a more ambitious question: could local geometric or “basin”
features rank tactical chess moves? The Level-2 experiment assembled features
such as check status, defender king mobility, attack coverage, checker support,
and mobility drop. The 14/15 handcrafted result looked exciting, but the
surviving code already contains the clue to its true nature: the ranker is a
fixed scorecard over chess-rule-derived facts, not a learned basin.

The tic-tac-toe and sliding experiments then tried to turn outcomes and local
improvements into persistent attractor/repulsor memory. The most important
thing to remember is that you did not merely repeat a demo: the saved sliding
state contains roughly 620,000 archived receipts across three modes. You were
testing persistence seriously. The fresh audit shows why it did not work:
state-only credit assignment and a lossy distance representation teach the
memory the wrong object. That failure is a design result, not wasted work.

## Preservation and exact identities

No historical source, result, receipt archive, or saved basin state was edited,
moved, renamed, or deleted during this audit. The fresh controls live only in
the separate recovery memory.

The six organized Games files have exact same-named root copies:

| File | Bytes | SHA-256 |
|---|---:|---|
| `LIVNIUM_CHESS_M1.md` | 1,497 | `9f38c8f0f273f93ba26d5b31c61e91e5f7c90db254e0c812dc645388c20c2a71` |
| `demo_sliding_puzzle.py` | 4,751 | `fbdd1a0e17b816f91c1cbc03c04d112bb0a492ec256fafd96b3110afaeb41b04` |
| `demo_sorting.py` | 3,679 | `714b8f0dd9cb43405c10a5d915ab2b25ae9acb0750aaba4561a6f2e804f67206` |
| `evaluate_chess_legacy.py` | 10,342 | `f421b1492e370395b259686de988dd7347d2770b8538f0a50325489baeb1c8e4` |
| `evaluate_tictactoe.py` | 11,160 | `d00c69de2fd7243d3bbfaf9d556ceb210bc5afdd4030960cc45ac92035727503` |
| `experiment_sliding.py` | 14,953 | `ea2e7051eabaac39e0147b424db7c5163e53a551ec51a2f714b9b6f0f018087a` |

The adjacent `evaluate_chess` project is a hidden project-inside-the-project,
not a duplicate of `evaluate_chess_legacy.py`. It contains 13 live modules:

| Module | SHA-256 |
|---|---|
| `__init__.py` | `b67cb45e2ff5627ef9094754904e70d87a8a233d4a71e50037420e439c4af86f` |
| `analysis.py` | `b6cc839f03eb6c44c81180709ea00dcb7097753166b3d316ca24ec166acce67b` |
| `basin_features.py` | `83738f68c710665ff31eca127515d627de937cf3acb4fdaecc15ba435798b170` |
| `basin_ranker.py` | `3b356ca55f0b772923d21725ac96a58f1eeebd313ca4e234483f0af7163cfd8c` |
| `decoding.py` | `f505752aeec6d494efdfe067d0a833ce669f18dad2ddf86d5a0bc764d5d867dd` |
| `encoding.py` | `cd350216a127091713f3a44734199ab4db24742687c0bd1420172a89fde73384` |
| `livnium_features.py` | `123b2231cd10e90a6b4c5b5e9abde34308eb11347dbf470a0dcbd01a82b82180` |
| `metadata.py` | `c2f35be889c805e965685ee9c051dcdebc698adf18333cfcae4104b54a958a53` |
| `move_apply.py` | `ec2d44f931201970cdb7673f640f8441035e6c27ac5a335ae76251ae5b000197` |
| `puzzle_dataset.py` | `49bdf2581a183fe387913052b36d01efe42ff5387b05e271b95d17c7efde6244` |
| `run_level15.py` | `2eacbd6d2f86b8c822f2d30211705d5248c5821bc9115146dbdfdb30527bf6a3` |
| `run_phase2.py` | `71e86699f17998918fe65e3bbe9ddb8fe11a4aac6abb6613531676f07eed2dd2` |
| `verification.py` | `3c46b66394e35debae40075b580790f4e9f2d26572e926fd04bc650319d0b7c9` |

Its eight-test suite is
`tests/test_evaluate_chess.py`, SHA-256
`00c0e46f916b646252f5fca2d414be4990b2f1aa9f9327fc939ad39ff36b8ca3`.

The three saved sliding memories and receipt archives are unique historical
artifacts:

| Mode | Basin JSON SHA-256 | Archive SHA-256 | Archived receipts |
|---|---|---|---:|
| `karmic_with_memory` | `6cc2fb387d2166f84a7b0789c9c4583a93f61f3ba77dced0bb07e78211d8b9c9` | `3658a27af9fe7672ceba373e8da50a632ecd6404f4205ee9f622c5f047ab7176` | 212,146 |
| `naive_both_with_memory` | `21940e03c5dbcb715902745339b074c6b5c6107ceae17cc5e240ac8cbe0d982e` | `117467fb6a0fa1bcec4fafa5daae8a682955d8c8f6b2581348093d76833f1f6f` | 195,651 |
| `naive_pull_with_memory` | `86b6418e9f7d17c1c3405e5380dd45ccf0894754c15bbfe69d36358de323d148` | `6c1ce99b3620dfd5230e165fa152624097190086df3bfec66468da11bbb6ccfa` | 212,530 |

Each JSON also retains 1,000 live ledger entries. The archived-to-live receipt
chains have zero adjacent hash breaks. The stored top-level `state_hash`,
however, is not the hash of the current canonical JSON because the save
routine computes the hash before replacing the previous `state_hash` field.
The receipts are internally chained; the JSON’s self-hash claim must be
narrowed.

## Chess generation 1 — board reconstruction

`evaluate_chess_legacy.py` places the 32 chess pieces into a Livnium lattice and
uses symbol swaps to represent ordinary moves, captures, promotion, castling,
and en passant.

The fresh random replay passes:

```text
successes = 1000
failures  = 0
```

The tested 1,000 moves contain 98 captures, one castling move, nine promotions,
and zero en-passant captures. Therefore the source printout’s broad statement
that the random run spans every special move is stronger than the actual random
coverage. Special moves are better supported by the improved project’s
targeted adversarial tests.

The legacy verifier reconstructs only `board_fen()`. It copies metadata from
the already-updated Python-chess source board. Legality, special-move
classification, and the expected result all come from Python chess. This is a
correct spatial transport check, not an independent full chess-state model.

## Chess generation 2 — explicit state transport

The improved `evaluate_chess` package fixes the largest legacy omission.
Metadata becomes part of the symbol state:

- side to move;
- white and black kingside/queenside castling rights;
- en-passant target or absence;
- optional move-clock tokens in the initial encoding.

`apply_livnium_move` implements piece relocation, captures, rook movement
during castling, promotion-symbol replacement, and metadata-token updates.
The state is checked after each transition against Python chess.

Fresh source results:

| Check | Result |
|---|---:|
| Unit tests | 8/8 pass |
| Random harness | 1,000/1,000 pass |
| Adversarial sequences | 22/22 transitions pass |
| New continuous-state random harness | 1,000/1,000 pass |
| Continuous-state symbol multiset | 0 conservation failures |
| Continuous adversarial harness | 22/22 pass |

The continuous-state control matters because the archived random and
adversarial harnesses re-encode the current Python board before each move.
That source protocol proves one-step equivalence but can hide state drift. The
new control encodes once and carries the Livnium state forward for all moves;
it still passes.

### Exact boundary

This layer is not an autonomous chess engine:

- a Python-chess `Board` is required by `apply_livnium_move`;
- Python chess determines whether a move is legal;
- Python chess supplies capture, castling, en-passant, promotion, and
  post-move metadata semantics;
- the Livnium state transports those already-known semantics.

The halfmove and fullmove clocks are encoded initially but not updated by
`move_apply` and not returned by `decode_board`. After two ordinary plies, the
decoded clock pair remains `(0, 1)` while Python chess expects `(0, 2)`.
Position identity used for repetition also depends on historical context that
is not represented here.

The accurate claim is:

> Livnium provides a conserved-symbol representation of chess pieces and the
> main position metadata, with verified move transport against Python chess.

It does not yet provide:

> a complete bijection over full FEN/game history or an independent
> implementation of chess legality.

## Chess generation 3 — mate-in-one ranking

The milestone note records:

```text
Top-1: 14/15
Top-3: 15/15
Mean rank: 1.0667
```

Fresh replay reproduces those numbers in both decoded and hybrid modes.

The source defines 20 handcrafted entries but silently filters invalid
positions, retaining 15. The single apparent top-1 miss is named
`alternative_mate_preferred`: the ranker prefers a different legal mate over
the nominated answer. Measured as “any mate first,” the handcrafted score is
therefore 15/15.

### What the “basin ranker” actually is

`basin_score` is a fixed linear combination with manually selected weights. No
basin state is learned, updated, loaded, or queried. The most important decoded
features are computed with Python-chess semantics:

- whether the move gives check;
- defender king legal-move count;
- attack maps and safe adjacent squares;
- direct-slider status;
- checker support and mutual support;
- opponent mobility drop.

The hybrid mode replaces a small subset of attack/safety calculations with a
local Livnium pseudo-attack map. Its complete rankings are identical to the
decoded mode on the 15 handcrafted and 100 generated puzzles.

### Elementary controls

On 100 freshly generated mate-in-one puzzles:

| Ranker | Any mate at top 1 |
|---|---:|
| Check only | 35/100 |
| Archived weighted ranker | 84/100 |
| Check, then fewest legal replies | 100/100 |

The last baseline is simply the definition of checkmate: a checking move with
zero legal replies. It uses no learned feature, geometry, or basin.

The source adversarial control creates positions with some non-mating checking
moves and reports how often the top move is checking. On 100 such positions,
the top move is checking 80% of the time. That is not a false-positive test for
mate classification: the ranker never emits a calibrated mate/non-mate
decision. Its mate scores range from 14.5 to 50.5, while non-mate top scores
overlap from 6.0 to 37.5.

The reusable piece is a clean candidate-enumeration and feature-analysis
harness. The basin-intelligence interpretation is retired.

## Tic-tac-toe — online loss avoidance

The implementation preserves physical piece identity. Each X or O token begins
in an off-board “hand” and reaches the board through `SwapSymbol`. Standard
symbolic code decides wins, legal squares, heuristic moves, and minimax moves.

The X policy projects the board to a nine-value vector:

```text
X = +1, O = -1, empty = 0
```

It chooses moves using distance to fixed-label anchors:

- pull toward `X_Win`;
- repel from `O_Win`;
- add small random noise.

After every game, every pre-move X state in the history receives the eventual
game outcome:

- an X win reinforces `X_Win`;
- an O win repels `X_Win` and reinforces `O_Win`;
- a draw updates nothing.

This is online episodic outcome credit, not a frozen predictor. The same games
used to print the result mutate the model.

### Source single-run replay

| Agent and opponent | X wins | O wins | Draws |
|---|---:|---:|---:|
| Pull-only X vs random, 1,000 | 772 | 129 | 99 |
| Pull+repulsion X vs random, 1,000 | 880 | 38 | 82 |
| Random X vs heuristic O, 500 | 8 | 423 | 69 |
| Online basin X vs heuristic O, 500 | 0 | 9 | 491 |

The last line is a real within-stream adaptation result: after early losses,
repulsion from losing states largely avoids subsequent losses.

### Five-seed frozen control

Five independent fields were trained online for 500 games each against the
heuristic opponent, then frozen and evaluated for another 500 games per
opponent:

| Phase/opponent | X wins | O wins | Draws |
|---|---:|---:|---:|
| Online training vs heuristic | 687 | 655 | 1,158 |
| Frozen vs random | 1,551 | 621 | 328 |
| Frozen vs heuristic | 810 | 729 | 961 |
| Frozen vs minimax | 0 | 1,484 | 1,016 |

The learned basins are therefore not a stable near-perfect tic-tac-toe policy.
The result depends heavily on continuing adaptation to the exact stream.

Symbolic baselines over the same five seeds and 2,500 games per matchup:

| X policy | Heuristic O | Minimax O |
|---|---|---|
| Random | 41 W / 2,096 L / 363 D | 0 W / 2,030 L / 470 D |
| Heuristic | 0 W / 0 L / 2,500 D | 0 W / 0 L / 2,500 D |
| Minimax | 0 W / 0 L / 2,500 D | 0 W / 0 L / 2,500 D |

The simple heuristic already solves the tested policy problem. The basin
experiment remains useful as a compact demonstration of online repulsion from
bad trajectories, but not as a game-playing advantage.

## Sliding demo — objective-guided annealing

`demo_sliding_puzzle.py` correctly uses legal blank-adjacent swaps and supplies
Manhattan distance to the known goal as the exact energy. The puzzle is
guaranteed solvable because it begins at the goal and performs 40 random legal
moves.

Immediate reversals are allowed, so “40 moves shuffled” is not difficulty or
optimal depth. The deterministic seed-99 start has:

- Manhattan distance: 8;
- exact optimal depth: 8;
- fresh annealing solution: step 3,714;
- accepted moves at solution: 2,678.

The demo is valid simulated-annealing engineering. It is thousands of moves
slower than an exact eight-move solution because its purpose is to exercise the
generic search engine, not to be a competitive 8-puzzle solver.

## Sliding experiment — why memory fails

`experiment_sliding.py` compares four modes:

- `off`: Manhattan-greedy selection plus noise;
- `naive_pull`: attraction to reinforced state anchors;
- `naive_both`: attraction to positive and repulsion from negative anchors;
- `karmic`: the basin implementation with its configured update rules.

The name `no_memory` means no persistence between the five sessions. It does
not mean no learning: every mode except `off` still learns within each
30-attempt session. `with_memory` carries anchors across sessions.

### Fresh isolated source replay

The replay was run in a temporary directory so it could not overwrite the
historical `state/exp_sliding` memories.

| Mode | No-memory solve rate | With-memory solve rate |
|---|---:|---:|
| `off` | 74% | not applicable |
| `naive_pull` | 3% | 1% |
| `naive_both` | 9% | 3% |
| `karmic` | 3% | 1% |

The `off` control ends at mean Manhattan distance 1.31. The persistent karmic
condition ends at mean distance 8.27. Persistence worsens all three memory
modes.

### Start-depth control

The 150 source starts are created by 25-step random walks with reversals:

| Property | Fresh exact result |
|---|---:|
| Unique starts | 126/150 |
| Mean optimal depth | 8.493 |
| Median optimal depth | 9 |
| Minimum / maximum depth | 1 / 17 |
| Exact solver within 300 moves | 150/150 |

The source’s 300-step budget is therefore 17.6–300 times the optimal depth of
every tested start.

### Representation collision

The feature stores, at each current board position, the Manhattan distance of
the occupying tile from its goal; the blank receives zero.

Across all 181,440 reachable 8-puzzle states:

| Representation | Unique signatures | Largest collision |
|---|---:|---:|
| Exact feature tuple | 63,591 | 33 states |
| Cosine direction/ray | 63,383 | 35 states |

Different boards therefore share an indistinguishable basin input.

The solved board maps to the all-zero vector. Cosine distance from that vector
to every nonzero vector is defined by the basin code as 1.0. Consequently a
terminal solved anchor cannot exert graded attraction on any unsolved state.

### Credit-assignment mismatch

The history stores the feature of the state *before* an action:

- if the next state improves, the pre-action state is reinforced;
- if the next state worsens, the pre-action state is penalized;
- the selected action is not part of the memory key.

Later move selection scores candidate next states against anchors trained on
pre-move states. The memory is therefore asked to predict action quality from a
state-only label that was attached to a different temporal role. This is the
main design lesson.

A viable successor would need at least:

1. a state-action or transition representation;
2. reward attached to the resulting transition;
3. a nondegenerate goal representation/distance;
4. a frozen evaluation set;
5. exact BFS/A* and greedy-Manhattan baselines;
6. matched seeds for within-session versus cross-session memory.

Do not start that successor during recovery. Preserve this result first.

## Sorting demo — supplied target, not learned order

`demo_sorting.py` exposes all 45 pairwise swaps among ten active cells and
supplies the exact number of misplaced symbols as its energy.

For seed 7:

- initial permutation: `[8, 3, 1, 4, 7, 0, 9, 6, 2, 5]`;
- exact minimum arbitrary swaps: 9;
- direct place-by-symbol method: 9 swaps;
- source annealing solution: 1,345 steps.

Across 1,000 random ten-item permutations, direct placement solves 1,000/1,000
with a mean exact minimum of 7.087 swaps and a maximum of nine.

The demo successfully exercises generic energy-guided transforms. It does not
discover sorting, learn an algorithm, or beat a symbolic baseline.

## Evidence table

| Claim | Status | Reason |
|---|---|---|
| Chess pieces and main metadata can be represented with conserved Livnium symbols | Verified engineering | Unit, random, adversarial, and continuous-state controls pass |
| Chess move transport is bijective over the complete game state | Partial / narrowed | Clocks and repetition history are not maintained |
| Livnium independently enforces chess rules | Retired | Python chess supplies legality and move semantics |
| Random chess harness spans every special move | Retired as phrased | Zero en-passant events in the measured 1,000 moves |
| Handcrafted mate target is top-1 in 14/15 | Measured | Freshly reproduced |
| Any mate is top-1 in 15/15 handcrafted puzzles | Measured | The one nominated-answer miss chooses another mate |
| Basin ranking solves generated mate-in-one | Retired as basin claim | Fixed heuristic; elementary legal-reply baseline is 100/100 |
| Hybrid Livnium attack features improve mate ranking | Retired on tested sets | Decoded and hybrid rankings are exactly identical |
| Tic-tac-toe basin repulsion adapts online to repeated losses | Measured | Single-run and five-seed training streams show adaptation |
| Tic-tac-toe result is a frozen general policy | Retired | Frozen agent loses 729/2,500 to heuristic and 1,484/2,500 to minimax |
| Basin tic-tac-toe beats a simple symbolic policy | Retired | Heuristic draws every tested game |
| Sliding demo solves a legal puzzle with Livnium transforms | Verified engineering | Deterministic replay solves |
| “40-move” or “25-move” shuffle is exact difficulty | Retired | Reversals reduce exact depth to 1–17 in experiment starts |
| Persistent basin memory improves sliding-puzzle solving | Retired | Every persistent condition is worse |
| Saved sliding receipts are internally chained | Verified engineering | Zero adjacent receipt-chain breaks |
| Saved JSON top-level state hash verifies current JSON | Retired | Hash is computed before replacing its own field |
| Sorting demo reaches the exact target | Verified engineering | Deterministic replay reaches zero energy |
| Sorting demo learns or efficiently discovers sorting | Retired | Exact direct method needs 9 versus 1,345 annealing steps |

## Reusable components

Preserve and potentially extract later:

- chess symbol encoding and metadata-token layout;
- continuous state/multiset conservation test;
- adversarial special-move sequences;
- candidate-move analysis and explicit feature records;
- physical piece-reserve pattern for small board games;
- deterministic receipt-chain validation;
- exact BFS depth-map and representation-collision checks;
- the negative lesson that state-only anchors cannot replace action credit;
- the protocol distinction between online adaptation and frozen evaluation.

Do not extract yet:

- the fixed mate score as a reusable “basin”;
- tic-tac-toe anchors as a general policy;
- sliding saved anchors as active puzzle knowledge;
- annealing demos as game/search benchmarks.

## Reproduction

The durable read-only probe is:

`/Users/chetanpatil/Desktop/LIVNIUM_MEMORY/GAMES_AUDIT_PROBE.py`

Run from the archive root:

```bash
cd /Users/chetanpatil/Desktop/test
PYTHONDONTWRITEBYTECODE=1 \
python3 /Users/chetanpatil/Desktop/LIVNIUM_MEMORY/GAMES_AUDIT_PROBE.py
```

It checks:

- exact root/organized identities and source hashes;
- chess one-step and continuous-state transport;
- move-type coverage, metadata-clock boundary, and symbol conservation;
- handcrafted and generated mate rankings with elementary baselines;
- five-seed online versus frozen tic-tac-toe behavior;
- symbolic heuristic and minimax controls;
- all 181,440 reachable 8-puzzle states and exact depths;
- feature/cosine collisions and the zero-goal distance boundary;
- saved basin receipts and top-level hashes;
- exact sorting-swap baselines.

The full sliding source matrix was replayed from an isolated temporary working
directory. Do not run it from the archive root unless intentionally creating or
overwriting new `state/exp_sliding` outputs.

## Stop and next action

S17 is complete when this document, the probe, and all navigation ledgers are
synced and byte-verified in `/Users/chetanpatil/Desktop/LIVNIUM_MEMORY`.

The next recovery task is the next still-unincorporated P1 family, not another
game model. Audit the NLI/Language organized family or, if that row is already
fully covered by the existing NLI audits after exact hash reconciliation, move
to Demos and Nova bridges.

Do not:

- delete the six exact root/organized preservation pairs;
- overwrite the three saved sliding memories or their receipt archives;
- call Python-chess-backed move transport an independent chess engine;
- call the fixed mate score a learned basin;
- report online tic-tac-toe games as held-out evaluation;
- use random-walk length as puzzle depth;
- omit BFS/A*, greedy Manhattan, direct sorting, heuristic, or minimax
  baselines;
- turn this negative result into another restart before the remaining recovery
  queue is finished.
