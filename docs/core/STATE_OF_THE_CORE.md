# State of the core

Current checkout-verifiable status, updated 2026-07-15.

## 1. Livnium Core — maintained and reproducible

`packages/livnium-core/` is dependency-free and contains the base-27 codec,
odd-cube exposure geometry, 24 orientation-preserving rotations, face turns,
hierarchy bookkeeping, multipole summaries, layer language, and ping/path
geometry.

- 67 focused tests run on Python 3.8–3.12 in CI.
- Ping frames are validated as members of the cube rotation group.
- Public operations preserve their documented reversible/conserved invariants.

```bash
python3 -m pip install -e "packages/livnium-core[test]"
python3 -m pytest packages/livnium-core/tests -q
```

## 2. Vector Collapse Engine — maintained experimental package

`packages/vector-collapse/` separates three update laws:

| mode | claim |
|---|---|
| `gradient_collapse` | exact gradient of a cosine potential |
| `direct_collapse` | closed-form approximation; no iterative energy claim |
| `mlp_collapse` | learned residual; observations are empirical |

Dynamic basin routing and the ledger are regression-tested. Post-step alignment
is recorded beside the post-step state, and inactive basin slots are masked.

## 3. Trained models — measured, checkpoint-dependent

Promoted models now live under `models/`:

- `noun-collapse/` — strongest result, SimLex noun ρ = 0.3616.
- `premise-generator/` — working contextual generator, label control still weak.
- `collapse-nli/` — 68.87% SNLI, matched end-to-end ablation pending.

Weights are intentionally excluded from Git. `artifacts/checkpoints.md` records
canonical local paths, byte sizes, SHA-256 hashes, and publication status.

## 4. Ramsey and conserved sum trees — reproducible research

`research/ramsey/` is present and contains the scripts, witness, race table, and
independent checker. The R(4,5) coloring is a reconstruction of a known bound,
not a new Ramsey number. Its verification is nevertheless exhaustive: every red
K4 and blue K5 is checked.

```bash
cd research/ramsey
python3 cayley_cube_ramsey.py
python3 independent_check.py
python3 recursive_sumtree_bench.py
```

The COMPASS-vs-WalkSAT-vs-SA result is a measured heuristic comparison under a
fixed budget, with an n=25 unsatisfiable control. It is not a universal solver
claim.

## 5. Active research and archive

Active work lives in `research/`; superseded work lives in `archive/`. Active
scripts may depend on Torch, NumPy, datasets, or local checkpoints and do not
receive package-level API guarantees. Archived code is retained for auditability
and must not be a dependency of maintained code.

## 6. Withdrawn or narrowed claims

- “Amplitude-like computer / 500 qubits” was narrowed to a classical small-scale
  state-vector simulator.
- “Universal geometry engine” was dropped.
- “Attention-free” is used only where accurate; the shipped premise checkpoint
  has lightweight cross-attention but no transformer self-attention.
- The chord-directed v1 force is non-conservative. Its sampled descent is an
  empirical observation, not proof of a global Lyapunov potential.
- Geometry-only language claims were rejected by fair NLI benchmarks.

The maintenance rule is simple: a current claim needs current code, a runnable
command, and an identified artifact when weights or data are required.
