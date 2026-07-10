"""DynamicsLedger: populated traces, exact-energy descent, exports.

Key claims under test:
- gradient_collapse records an EXACT potential and it actually decreases;
- direct/mlp modes record empirical observations only, never "proven energy";
- collapse_dynamic fills its trace (it used to return an empty one) and
  records seed/spawn/prune/merge basin events;
- ledger=None costs nothing and changes nothing.
"""

import json

import pytest
import torch

from vector_collapse import (
    CollapseConfig,
    DynamicsLedger,
    VectorCollapseEngine,
)

DIM = 32
B = 8


def make_engine(mode: str, **overrides) -> VectorCollapseEngine:
    cfg = CollapseConfig(dim=DIM, num_layers=6, mode=mode, **overrides)
    torch.manual_seed(0)
    return VectorCollapseEngine(cfg)


def batch(seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, DIM)


# ---- static collapse ----


def test_gradient_collapse_exact_energy_decreases():
    engine = make_engine("gradient_collapse")
    ledger = DynamicsLedger()
    with torch.no_grad():
        engine.collapse(batch(), ledger=ledger)

    # initial state + one record per layer
    assert len(ledger.steps) == engine.num_layers + 1
    assert all(s.energy is not None and s.energy_kind == "exact" for s in ledger.steps)
    assert ledger.energy_monotone(), "gradient collapse must descend its own potential"
    assert ledger.mode == "gradient_collapse"
    assert ledger.convergence_reason in ("converged", "max_steps")
    # per-anchor alignment recorded for every label
    assert all(len(s.align) == engine.cfg.num_labels for s in ledger.steps)
    # displacement is 0 at the initial record, > 0 once dynamics run
    assert ledger.steps[0].displacement == 0.0
    assert ledger.steps[1].displacement > 0.0


def test_direct_collapse_is_closed_form_and_empirical():
    engine = make_engine("direct_collapse")
    ledger = DynamicsLedger()
    with torch.no_grad():
        engine.collapse(batch(), ledger=ledger)

    assert ledger.convergence_reason == "closed_form"
    assert ledger.converged
    assert all(s.energy is None for s in ledger.steps)
    with pytest.raises(ValueError):
        ledger.energy_monotone()  # empirical runs can't claim proven descent


def test_mlp_collapse_records_force_but_no_exact_energy():
    engine = make_engine("mlp_collapse")
    ledger = DynamicsLedger()
    with torch.no_grad():
        engine.collapse(batch(), ledger=ledger)

    assert len(ledger.steps) == engine.num_layers + 1
    assert all(s.energy_kind == "empirical" for s in ledger.steps)
    assert all(s.force is not None for s in ledger.steps[1:])


def test_ledger_is_optional_and_non_invasive():
    engine = make_engine("gradient_collapse")
    x = batch()
    with torch.no_grad():
        h_plain, _ = engine.collapse(x)
        h_logged, _ = engine.collapse(x, ledger=DynamicsLedger())
    assert torch.allclose(h_plain, h_logged)


# ---- dynamic collapse ----


def dynamic_run(mode: str, ledger=None, steps: int = 1):
    engine = make_engine(mode)
    field = engine.make_basin_field()
    x = batch()
    labels = torch.randint(0, engine.cfg.num_labels, (B,))
    with torch.no_grad():
        for step in range(steps):
            h, trace = engine.collapse_dynamic(
                x, labels, field, global_step=step, ledger=ledger, prune_every=0
            )
    return h, trace, field, engine


def test_dynamic_trace_is_populated():
    h, trace, field, engine = dynamic_run("gradient_collapse")
    assert len(trace["align"]) == engine.num_layers
    assert len(trace["div"]) == engine.num_layers
    assert len(trace["tens"]) == engine.num_layers
    assert trace["align"][0].shape == (B,)
    # attractive law: div = 1 - align, tension = |div|
    a, d, t = trace["align"][-1], trace["div"][-1], trace["tens"][-1]
    assert torch.allclose(d, 1.0 - a, atol=1e-6)
    assert torch.allclose(t, d.abs(), atol=1e-6)


def test_dynamic_ledger_records_seed_and_selection():
    ledger = DynamicsLedger()
    h, trace, field, engine = dynamic_run("gradient_collapse", ledger=ledger)

    seeds = [e for e in ledger.events if e.kind == "seed"]
    assert seeds, "first routing of each label must seed a basin"
    assert ledger.basin_selection, "routed slot per label must be recorded"
    for label, slots in ledger.basin_selection.items():
        assert label in engine.cfg.labels
        assert all(0 <= s < engine.cfg.basin.max_basins_per_label for s in slots)
    assert all(s.energy_kind == "exact" for s in ledger.steps)
    assert ledger.meta["dynamic"] is True


def test_dynamic_direct_mode_closed_form():
    ledger = DynamicsLedger()
    dynamic_run("direct_collapse", ledger=ledger)
    assert ledger.convergence_reason == "closed_form"
    assert len(ledger.steps) == 1  # one closed-form jump, not num_layers


# ---- exports ----


def test_exports_roundtrip(tmp_path):
    engine = make_engine("gradient_collapse")
    ledger = DynamicsLedger()
    with torch.no_grad():
        engine.collapse(batch(), ledger=ledger)

    j, c, m = tmp_path / "r.json", tmp_path / "r.csv", tmp_path / "r.md"
    ledger.to_json(j)
    ledger.to_csv(c)
    ledger.to_markdown(m)

    data = json.loads(j.read_text())
    assert data["summary"]["energy_monotone"] is True
    assert len(data["steps"]) == len(ledger.steps)
    assert data["labels"] == list(engine.cfg.labels)

    header = c.read_text().splitlines()[0].split(",")
    assert header[:3] == ["step", "norm", "displacement"]
    assert f"align_{engine.cfg.labels[0]}" in header

    report = m.read_text()
    assert "# Collapse dynamics report" in report
    assert "monotone: yes" in report
    assert "Moksha" not in report  # converged, not mysticism


def test_markdown_flags_energy_violation():
    ledger = DynamicsLedger()
    h = torch.ones(2, DIM)
    for i, e in enumerate([1.0, 2.0]):  # energy goes UP
        ledger.log_step(i, h, h, torch.zeros(2, 1), energy=torch.tensor(e), energy_kind="exact")
    ledger.finish()
    assert not ledger.energy_monotone()
    assert "NO" in ledger.to_markdown()
