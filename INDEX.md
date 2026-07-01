# Livnium — Project Index, Graded & Ranked

*Generated 2026-06-27. Updated 2026-07-02: `chat/` now also hosts the **chat-brain** — a personal char→word→sentence→context→reasoning ladder trained on a raw ChatGPT export (see the chat/ section and `chat/README.md` Part 1). Covers all real files — caches (`__pycache__`, `.pyc`, `.pytest_cache`, `.ruff_cache`), `.git`, and `.DS_Store` excluded.*

## How to read this

Every file/component gets a **blended grade** combining two axes:

- **Importance** — how central it is to the project (the proven core ranks highest; archived duplicates lowest).
- **Quality** — maturity, documentation, honesty, whether claims are verified.

Grades: **A+** keeper / load-bearing · **A** strong · **B** solid working code · **C** experiment or rough · **D** redundant/dead weight.

### The one structural fact that drives the ranking

The repo is **heavily duplicated**. `reached/pure/` is a near-complete mirror of the whole project (its own `livnium_core/`, `cortex_v2/`, `docs/`, `tests/`, `results/`, plus model weights). `reached/pure-cleaned/`, `chat/`, and `collapse_retrain/` ↔ `reached/code/` are cleaned/partial copies of the same experiments. Roughly **half the 344 files are duplicates**. Canonical copies live at the repo root and in the top-level package dirs; the `reached/` tree is an archived snapshot and is graded down for redundancy regardless of internal quality.

---

## Top-level ranking (by component)

| Rank | Component | Grade | What it is |
|---|---|---|---|
| 1 | `livnium_core/` | **A+** | The proven, zero-dependency mathematical core. The keeper. |
| 2 | `tests/` | **A** | Test suite that backs the core's correctness claims. |
| 3 | `docs/` | **A** | The honest written record — formulas, limits, findings, components. |
| 4 | `ramsey/` | **A** | Verified Ramsey results (Cayley-on-cube-group, COMPASS, sum-tree). |
| 5 | Root verdict/status docs | **A** | `README`, `STATE_OF_THE_CORE`, the two `*_VERDICT` files. |
| 6 | `chat/` | **B+** | The on-device 5.98M premise generator demo + the chat-brain ladder (2026-07-02). |
| 7 | `collapse_retrain/` | **B** | Supervised NLI collapse model v1 (~68.9% SNLI) + training harness. |
| 8 | `cortex_v2/` | **B** | Geometry/MPS experiments + validation scripts (audited, patched). |
| 9 | `results/` | **B** | SNLI benchmark ladder scripts + honest RESULTS.md. |
| 10 | Root experiment scripts | **B–** | `livnium.py`, `geometry_discriminator_test.py`, `char_typer_symbols.py`. |
| 11 | `reached/code`, `reached/models`, `reached/pure-cleaned` | **C** | Archived experiment snapshots; mostly duplicated elsewhere. |
| 12 | `reached/pure/` | **D** | Full mirror of the repo. Redundant by construction. |

---

## Tier 1 — The core (A+ / A)

### `livnium_core/` — the proven system *(grade: A+)*

Pure Python, zero dependencies, fully tested. This is the part safe to call "Livnium."

| File | Grade | Notes |
|---|---|---|
| `__init__.py` | A+ | Clean public API + honest docstring ("no ML claims"). |
| `lattice.py` | A+ | Cube geometry, exposure classes, symbolic-weight conservation law. Largest core module (7.2 KB). |
| `ping.py` | A | The "inward ping" / nested-frame descent — newest core addition (16.7 KB). |
| `base27.py` | A+ | Lossless base-27 codec (`z + a = a0`). |
| `rotations.py` | A+ | The 24 cube rotations (S₄), all reversible. |
| `moves.py` | A | Face permutations / move sequences. |
| `hierarchy.py` | A | Nested-cube bookkeeping, capacity, wreath group order. |
| `layer_language.py` | A | Small parser/evaluator for the layer language. |
| `__init__.py` (sub) | — | `ramsey/FINDINGS.md` lives under the package; see ramsey tier. |

### `tests/` — correctness evidence *(grade: A)*

Each maps to a core module. These are what make the core trustworthy.

`test_base27.py`, `test_lattice.py`, `test_rotations.py`, `test_moves.py`, `test_hierarchy.py`, `test_layer_language.py`, `test_moments.py`, `test_ping.py` — **all A**.

---

## Tier 2 — The written record & verified findings (A)

### Root docs

| File | Grade | Notes |
|---|---|---|
| `README.md` | A+ | Honest framing: "the math is real, the 'beats AI' claim isn't." Best entry point. |
| `STATE_OF_THE_CORE.md` | A+ | Every claim with a number + reproducing script. The status source of truth. |
| `COLLAPSE_ENGINE_VERDICT.md` | A | Forensic post-mortem of the cortex_v2 training bugs. Excellent engineering honesty. |
| `GEOMETRY_DISCRIMINATOR_VERDICT.md` | A | Classical-vs-amplitude verdict; clear scope. |
| `LICENSE` | B | PolyForm NC 1.0. |
| `COMMERCIAL.md`, `CONTRIBUTING.md` | B | Standard project docs. |
| `pyproject.toml`, `requirements.txt`, `.gitignore`, `.github/workflows/ci.yml` | B | Project config / CI. |

### `docs/` *(grade: A)*

| File | Grade | Notes |
|---|---|---|
| `COMPONENTS.md` | A+ | Plain-language tour of every part — what each is "good for / not." |
| `FORMULAS.md` | A | The closed-form weight/conservation math. |
| `FINDINGS.md`, `ORIGINS.md`, `LIMITS.md` | A | History, results, honest boundaries. |
| `ML_LADDER.md`, `BENCHMARKS.md` | A | The honest ML benchmark record. |
| `COLLAPSE_STRUCTURE_REPORT.md`, `REARRANGEMENT.md`, `COLLAPSE_VISUALIZATION.md`, `COMPRESSION_NOTE.md` | B+ | Supporting reports. |
| `structure_probe_data.json` | B | Probe data backing a report. |
| `images/*.png` (5) | B | Figures: flow field, grid warping, char trajectories, anchor maps. |

### `ramsey/` — verified math findings *(grade: A)*

| File | Grade | Notes |
|---|---|---|
| `FINDINGS.md`, `R45_RACE_FINDINGS.md` | A | Exhaustively-verified Ramsey results with honest scope. |
| `cayley_cube_ramsey.py` | A | Cube rotation group (order 24) → R(4,5)≥25 witness. The structural result. |
| `compass_solver.py` | A | COMPASS net-delta solver; 8/8 on R(4,4) n=17. |
| `recursive_sumtree_bench.py` | A | Conserved 27-tree benchmark vs Fenwick/prefix-sum. |
| `r45_race.py`, `compass`/`independent_check.py`, `chunk.py` | B | Race harness + checkers. |
| `master.csv`, `witness_n24.json`, `r45_race_table.md` | B | Result data/artifacts. |
| `cayley_cube_ramsey` siblings | B | Supporting solver scripts. |

> Note: `livnium_core/ramsey/FINDINGS.md` and root `livnium.py` are related Ramsey artifacts — `livnium.py` (43 KB) is the "pattern-replication operator" stress-test against diagonal Ramsey numbers; honest, but a large standalone monolith → **B–**.

---

## Tier 3 — Experiments (B)

### `chat/` — the on-device demo + the chat-brain *(grade: B+)*

Two packages. (a) The 5.98M-param SNLI premise generator with measured numbers
and explicit claim corrections. (b) **The chat-brain (added 2026-07-02):** a
personal ladder — char → word → sentence → context → reasoning — trained on a
raw ChatGPT export (`conversations.json`, canonical-path walk, single source of
truth; the flatten is retired). Same collapse engine at every rung.

**Chat-brain files (2026-07-02):**

| File | Grade | Notes |
|---|---|---|
| `char_typer_all.py` | B+ | Char rung on RAW lines — all ~2k chars incl ENTER; code/emoji intact. |
| `chat_typer.py` | B+ | Word rung, char-stage-free; `--max-vocab 0` = every word. 100% clean held-out (20k run). |
| `chat_typer_live.py`, `char_fingerprint.py` | B | Minting bridge: unseen words get spelling-derived wells live. |
| `prep_chat_context.py` | B+ | Session-aware prep: canonical tree walk, `<you>`/`<me>` turns, sealed sessions. |
| `chat_reply.py` | B+ | Reasoning rung: collapse-trajectory reader + growing self-attend memory; `--chat` multi-turn REPL with thinking traces. |
| `prep_chat_sentences.py` | C | Word-corpus prep — **still flatten-sourced, pending canonical rewrite**. |
| `prep_chat_pairs.py`, `data/chat_pairs.tsv` | D | Deprecated (flatten, context-free) — superseded by `prep_chat_context.py`. |
| `model/chat_typer.pt`, `model/chat_reply.pt`, `model/char_typer_all.pt` | B | Chat-brain checkpoints (word 20k / reasoning v1 / char — see `CLAIMS_CHECKPOINT_MAP.md`). |
| `data/chat_context.tsv`, `data/chat_sentences.txt` | B | Training data (17,962 context→reply examples; 497k sentences). |

**SNLI demo files:**

| File | Grade | Notes |
|---|---|---|
| `README.md` | A | Honest, with Reddit-post corrections + measured latency table (now also documents the chat-brain). |
| `CLAIMS_CHECKPOINT_MAP.md`, `LYAPUNOV_TEST.md`, `SNLI_BASELINES.md`, `BENCHMARKS.md` | A | The claim-to-evidence paper trail. |
| `premise_from_hyp.py` | B+ | Main generator/classifier (25.5 KB). |
| `sentence_typer.py` | B+ | Word wells + tokenizer. |
| `chat_premise.py`, `chat_bench*.py` | B | Interactive + benchmark scripts. |
| `gravity_embed.py`, `supervised_embed.py`, `supervised_gravity.py`, `ordered_sentence_embed.py`, `relational_sentence_embed.py`, `token_path_embed.py`, `ping_embed_probe.py`, `verify_lyapunov.py` | B | Embedding/probe experiments. |
| `char_collapse_pure.py` | B | Frozen char→word stage (dup of root concept). |
| `model/premise_from_hyp_align_53.pt`, `batch_crossover.png` | B | Shipped checkpoint + figure. |

### `collapse_retrain/` — supervised NLI v1 *(grade: B)*

The label-supervised collapse model (~68.9% SNLI test). Well-documented training/eval/ablation harness.

| File | Grade | Notes |
|---|---|---|
| `README.md`, `MODEL_CARD.md`, `LADDER.md` | A | Pipeline manual + model card. |
| `train_collapse_embeddings.py` | B+ | Main trainer (30.7 KB). |
| `vector_collapse.py` | B+ | The 4-layer VectorCollapseEngine. |
| `eval_nli.py`, `ablate_nli.py`, `predict.py` | B+ | Eval + ablation + single-prediction CLI. |
| `structure_probe.py`, `basin_field.py`, `text_encoder_collapse.py` | B | Probes + encoder. |
| `train_nli_meaning_forms*.py`, `train_nli_meaning_head.py`, `train_nli_from_pure.py`, `train_snli_typer*.py`, `train_char_collapse.py` | B | The "meaning forms around char structure" ladder. |
| `char_collapse.py`, `char_collapse_pure.py`, `char_typer_symbols.py`, `sentence_typer.py`, `word_typer*.py`, `word_to_char.py` | B | Char/word stages. |
| `save_failures.py`, `score_sentence_typer.py`, `eval_nli_meaning_forms_symbols.py`, `visualize_char_collapse.py`, `verify_char_collapse_numpy.py` | B | Utilities. |
| `*.pt` (8 checkpoints), `failed_examples.json`, `model_nli_v1/nli_epoch20.pt` | B | Trained weights (large binaries). |

### `cortex_v2/` — geometry/MPS experiments *(grade: B)*

Audited & patched per `COLLAPSE_ENGINE_VERDICT.md`. Strong validation coverage.

| File | Grade | Notes |
|---|---|---|
| `SESSION_FINDINGS.md` | A | Findings log. |
| `lattice.py`, `mps.py`, `mps_qudit.py` | B+ | Lattice sim + MPS + complex-amplitude qudit layer. |
| `train_core_nli.py`, `train_core_nli_scaled.py` | B | Training scripts. |
| `validate_*.py` (20 scripts) | B | Broad validation suite (cube, conservation, nesting, energy, survival, forward-cone, etc.). |
| `test_regressions.py` | B | Regression tests. |

### `results/` *(grade: B)*

`RESULTS.md` (A — honest SNLI verdict), `README.md` (B), `rung2_lib.py`, `rung2_livnium.py`, `rung2_livnium_word.py` (B — benchmark ladder).

### Other root experiment scripts

| File | Grade | Notes |
|---|---|---|
| `livnium.py` | B– | 43 KB Ramsey stress-test monolith; honest but large/standalone. |
| `geometry_discriminator_test.py` | B | Backs the geometry verdict (reproducible). |
| `char_typer_symbols.py` + `char_typer_symbols.pt` | B | Pure-geometry whole-symbol typer. |
| `visualizer/index.html` | B | Standalone visualizer. |

---

## Tier 4 — Archived duplicates (C / D)

These are graded **down for redundancy**, not for internal quality — several are byte-identical copies of canonical files above.

| Path | Grade | Notes |
|---|---|---|
| `reached/README.md` | B | Useful narrative of the char-collapse → meaning journey. |
| `reached/code/` (13 files) | C | Snapshot of the char/word/NLI experiment scripts — overlaps `collapse_retrain/`. |
| `reached/models/` (4 `.pt`) | C | Duplicate checkpoints (`char_typer`, `nli_meaning_*`). |
| `reached/pure-cleaned/` (9 files) | C | Cleaned premise-generator build; overlaps `chat/`. |
| `reached/pure/` (58+ files) | **D** | **Full mirror** of the repo: its own `livnium_core/`, `cortex_v2/`, `docs/`, `tests/`, `results/`, `model/`, `hf_upload/`, plus the gateN scripts and `train_nli_*` variants. Redundant by construction; keep only if needed as a frozen release snapshot. |

Notable unique-ish items inside `reached/pure/` (still archived): the `gate3b/gate4/gate5/gate6*` phase-gate scripts, `train_nli_joint_nested.py`, `born_dims_corpus.py`, `closed_loop.py`, `phase_*_test.py`, and `hf_upload/` (HuggingFace packaging: `modeling_collapsenli.py`, `config.json`, `eval_snli.py`, `UPLOAD.md`) — all **C**, interesting but parked.

---

## Large binaries & data (informational)

Not source, but they consume the bulk of disk. Grade reflects keep-value, not quality.

| File | Size | Grade | Notes |
|---|---|---|---|
| `reached/pure/data/snli_1.0_train.jsonl` ×2 | 487 MB each | C | SNLI training set, duplicated across `pure` and `pure-cleaned`. ~975 MB total — biggest cleanup target. |
| `snli_1.0_dev/test.jsonl` ×2 | ~9.7 MB each | C | Duplicated SNLI splits. |
| `collapse_retrain/model_nli_v1/nli_epoch20.pt` ×2 | 52.6 MB | B | The official 68.9% checkpoint (also mirrored in `reached/pure`). |
| Various `*.pt` (sentence_typer, premise_from_hyp_align, nli_meaning_*) | 21–51 MB | B/C | Many duplicated 2–6× across trees. |
| `reached/pure/embedding.npy` | 20.5 MB | C | Exported embeddings. |

---

## Bottom line

The **A-grade heart of the project is small and excellent**: `livnium_core/` + `tests/` + `docs/` + `ramsey/` + the honest status/verdict docs. That's the keeper, and it's genuinely well-documented and verified.

The **B-grade experimental layer** (`chat/`, `collapse_retrain/`, `cortex_v2/`, `results/`) is solid, honestly scoped research code.

The **biggest hygiene issue** is duplication: the entire `reached/pure/` mirror plus duplicated SNLI data (~1 GB) and repeated `.pt` checkpoints. Collapsing `reached/` to a single frozen snapshot (or removing it) and de-duplicating the SNLI `.jsonl` files would shrink the repo dramatically with zero loss of unique content.
