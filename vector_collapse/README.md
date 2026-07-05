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
