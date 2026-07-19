# Components and boundaries

Livnium has five layers. The boundary between them is intentional.

| layer | location | promise |
|---|---|---|
| mathematical core | `packages/livnium-core/` | proven invariants, zero runtime dependencies |
| dynamics engine | `packages/vector-collapse/` | reusable, tested experimental package |
| trained models | `models/` | checkpoint schema, evaluation, and runnable entry point |
| active research | `research/` | clear question; API may change |
| historical record | `archive/` | preserved, not maintained |

## Packages

`livnium-core` contains base-27 encoding, odd-cube exposure classes, symbolic
weight, the 24 cube rotations, face turns, hierarchy, multipole summaries, the
layer language, and ping/path geometry. Its public invariants are covered by 67
tests.

`vector-collapse` contains the configurable dynamics engine, direct and learned
variants, dynamic basin management, and the observability ledger. It is the only
reusable collapse implementation; model-local copies remain only where they are
needed to reproduce old checkpoints.

## Trained models

- `models/noun-collapse/` — pure-collapse noun embeddings and matched evaluation.
- `models/premise-generator/` — hypothesis+label to premise generation, including
  the sentence/character prerequisites, demo, benchmarks, and claim map.
- `models/collapse-nli/` — supervised three-class SNLI collapse model and ablations.

Each owns its checkpoint directory. The noun model no longer imports chat
preprocessing, and the premise generator no longer relies on a shim inside
chat-brain.

## Evidence

`benchmarks/embeddings/matched-corpus/` freezes one cleaned corpus and compares
collapse, SGNS, and PPMI-SVD under the same input. `benchmarks/nli/` contains the
character/word geometry ladder and shared external SNLI location.

`research/ramsey/` is active mathematical evidence: known Ramsey witnesses are
reconstructed and exhaustively verified; COMPASS is compared against WalkSAT and
simulated annealing with an independent checker and unsatisfiable control.

## Active research

`research/chat-brain/`, `exact-gradient/`, `language-probes/`, `vision/`,
`discrete-chat/`, and `qwen-hook/` each have one responsibility. Experimental
results can be useful without being promoted to package or model status.

## Archive

`archive/cortex-v2/`, `rule30/`, `experiments/`, `chat-legacy/`, and the local
gitignored `reached/` snapshot preserve old work. Archived paths should not be
used as dependencies by maintained code.

## Promotion rule

A research component moves to `models/` only when it has:

1. a defined task and architecture,
2. a canonical checkpoint path and hash,
3. a measured benchmark with an honest baseline,
4. a runnable demo or evaluation command,
5. a README that states limitations.

A reusable implementation moves to `packages/` only when its API and regressions
are tested independently of one checkpoint.
