# vector_collapse

Standalone, configurable vector collapse engine. Extracted from
`collapse_retrain/` (which stays untouched) with the verdict fixes baked in:
attractive divergence law (`div = 1 - align`) and identity-init residual MLP.
See `COLLAPSE_ENGINE_VERDICT.md` at repo root for the derivations.

## Usage

```python
from vector_collapse import CollapseConfig, VectorCollapseEngine

# defaults (dim=256, 4 layers, labels E/C/N)
engine = VectorCollapseEngine()

# or from YAML
engine = VectorCollapseEngine.from_yaml("vector_collapse/config.yaml")

# or override in code
cfg = CollapseConfig(dim=512, strengths={"E": 0.2, "C": 0.2, "N": 0.1})
engine = VectorCollapseEngine(cfg)

# static collapse (three learned anchors)
h_final, _ = engine.collapse(h0)                    # h0: (B, dim)

# dynamic collapse with basins
field = engine.make_basin_field()
h_final, trace = engine.collapse_dynamic(h0, labels, field, global_step=step)
```

## Observability: DynamicsLedger

Pass a `DynamicsLedger` to either collapse call to record the full geometric
path: per-step norm, displacement, per-anchor alignment, force magnitude,
energy, basin seed/spawn/prune/merge events, routed basin slots, and the
convergence reason. Zero cost when omitted.

```python
from vector_collapse import DynamicsLedger

ledger = DynamicsLedger()
h_final, _ = engine.collapse(h0, ledger=ledger)

ledger.summary()            # dict: steps, convergence, energy check
ledger.to_json("run.json")  # full record
ledger.to_csv("run.csv")    # one row per step
ledger.to_markdown("run.md")  # human-readable report
```

Honesty rules: `energy_kind="exact"` is recorded only for
`gradient_collapse`, where the value is the real potential the dynamics
descend — `ledger.energy_monotone()` verifies the descent actually happened.
`direct_collapse` and `mlp_collapse` record empirical alignment/displacement
only, and `energy_monotone()` raises rather than letting an empirical run
claim proven descent. The terminal state is called `converged` in code;
"Moksha" is a visualization label only.

## Config

Everything lives in `CollapseConfig` (`config.py`) / `config.yaml`:
dim, num_layers, max_norm, label set + per-label strengths, and all basin
hyperparameters (spawn thresholds, anchor lr, prune/merge rules).

Label integer encoding follows `labels` order — default `[E, C, N]` = 0/1/2,
matching the trained NLI models.

## Differences from collapse_retrain/vector_collapse.py

- All hyperparameters come from `CollapseConfig` instead of constructor kwargs.
- Static anchors are one `(num_labels, dim)` parameter `anchors` instead of
  `anchor_entail/anchor_contra/anchor_neutral`. Old checkpoints load via
  `engine.load_legacy_state_dict(torch.load(path))`.
- Label count is configurable (not hardcoded to 3).
- Basin spawning stops early when the field is full instead of looping.

Numerics of `collapse()` and `collapse_dynamic()` are unchanged.
