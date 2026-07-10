# Livnium — Website Plan

Every product below traces to a real file in this repo. No inflated claims — the site sells the honesty as the brand.

## Product catalog (derived from the repo)

| # | Product | Source | Status | The number |
|---|---------|--------|--------|------------|
| 1 | **Livnium Core SDK** — conserved 3×3×3 geometric state space: base-27 codec, exposure weights, 24 rotations, nested hierarchy. Zero dependencies, fully tested. | `packages/livnium-core/src/livnium_core/`, `tests/` | Proven | ΣSW conserved under all 24 rotations; full test suite green |
| 2 | **Vector Collapse Engine** — standalone, configurable attention-free representation engine (attractor dynamics, basin fields, YAML config). | `packages/vector-collapse/src/vector_collapse/` | Shipping | Empirically attractor-directed; monotone energy descent on 100% of 12k sampled steps (chord force is non-conservative — not a proven Lyapunov energy) |
| 3 | **NLI Collapse Classifier v1** — supervised SNLI model, no transformer attention, geometric classification against E/N/C anchors. | `research/nli/supervised-collapse/`, checkpoint `nli_epoch20.pt` | Measured | 68.87% SNLI test (beats 61.5% hypothesis-only baseline) |
| 4 | **Chat-Brain / Premise Generator** — 5.98M-param on-device sequence model; type a hypothesis, it types back a premise. | `research/generation/chat-brain/` | Measured | ~4.5 ms/reply on CPU; classifier sibling 74.4% SNLI test |
| 5 | **Noun-Collapse Embeddings** — word meaning from pure collapse: one well per word, two scalars, no MLP/attention. On Hugging Face. | `research/embeddings/noun-collapse/noun_collapse_pure.py`, HF `chetanxpatil/noun-collapse` | Measured | 94.75M occurrences of WordNet noun-eligible tokens, one pass, ~3.2h on a MacBook |
| 6 | **Ramsey Toolkit** — Cayley-on-cube-group witnesses, COMPASS solver, conserved 27-ary sum-tree. Exhaustively verified. | `ramsey/` | Proven | R(4,5)≥25 witness on 24 vertices via the cube rotation group |
| 7 | **Char-Collapse Typer** — pure-geometry character encoder/decoder; anchors + start + 2 scalars, nothing else. | `reached/code/char_collapse_pure.py` | Proven | 100% exact typing on trained and unseen words |
| 8 | **Vision Collapse** — every attractor is a pixel; images collapse through their own pixels. | `research/vision/` | Research | Smoke-tested at 64²; not yet trained at full scale |
| 9 | **Rule 30 Investigation** — can collapse predict Rule 30's center column? The honest answer, with the full journey. | `research/archive/rule30/` | Case study | Fair test + two written assessments |
| 10 | **Honest Benchmark Harness** — the rung ladder: shuffled-label leakage control, dumb-baseline gate, same-everything protocol. | `benchmarks/nli/` | Method | Leakage control always ~33% (chance) — pipeline never cheats |
| 11 | **The Written Record** — ORIGINS, FINDINGS (every claim graded proven/partial/falsified), LIMITS, verdict post-mortems. | `docs/`, root `*_VERDICT.md` | Content | A year of claims, each with a number and a script |
| 12 | **Commercial License** — free for noncommercial use (PolyForm NC); companies pay a simple annual fee. | `LICENSE`, `COMMERCIAL.md` | Live | The business model |

## Site architecture (single page, dark)

1. **Hero** — rotating CSS 3×3×3 cube, tagline: "A conserved geometric state space. Every claim has a number behind it."
2. **The Honesty Bar** — the repo's differentiator, quoted from the README: *"The mathematics is real and proven. The early claim that it 'beats AI' is not."*
3. **Products** — 12 cards, each with status badge (Proven / Measured / Research / Case study), the key number, and the source path.
4. **By the numbers** — strip of headline metrics.
5. **The Method** — the three rules from `benchmarks/nli/README.md`.
6. **Licensing** — noncommercial free / commercial paid, contact CTA.
7. **Footer** — GitHub, Hugging Face, license.

## Aesthetic ("Livnium level")

- Near-black `#07090c` base, panel `#0e1218`, hairline borders.
- **Om gold** `#e8b44a` primary accent (the center cell, weight 0), teal `#4ad4c0` secondary.
- Monospace for numbers and file paths; wide-tracked uppercase micro-labels.
- 27-cell grid motif, faint background lattice, slow-rotating wireframe cube.
- Status badges color-coded; no stock imagery, no gradients-for-gradients'-sake.

## Later phases (not built yet)

- Phase 2: hook in `apps/core-visualizer/index.html` as an interactive demo page.
- Phase 3: per-product deep-dive pages generated from each README.
- Phase 4: live demo of the premise generator (WASM/NumPy port).
