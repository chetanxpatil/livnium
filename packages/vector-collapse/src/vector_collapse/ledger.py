"""DynamicsLedger: step-by-step observability for vector collapse.

Records what actually happened during a collapse — norms, displacement,
alignment, force, energy, basin events, convergence — and exports it as
JSON, CSV, or a Markdown report.

Honesty rules (see COLLAPSE_ENGINE_VERDICT.md):
- energy_kind="exact" only when the recorded value is a real potential the
  dynamics descend (gradient_collapse). The ledger then checks monotonicity.
- energy_kind="empirical" for everything else (mlp_collapse): alignment and
  displacement are reported as observations, not as a proven Lyapunov value.
- The terminal state is called `converged`, never "Moksha"; that word is a
  visualization label, not a code concept.

Zero overhead when unused: engine methods only log `if ledger is not None`.

Usage:
    ledger = DynamicsLedger(labels=cfg.labels)
    h, _ = engine.collapse(h0, ledger=ledger)
    ledger.to_json("run.json"); ledger.to_markdown("run.md")
"""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

__all__ = ["DynamicsLedger", "StepRecord", "BasinEvent"]


@dataclass
class StepRecord:
    """One collapse step, batch-reduced."""

    step: int
    norm: float  # mean ||h||
    displacement: float  # mean ||h_t - h_{t-1}||
    align: List[float]  # mean alignment per anchor/target
    force: Optional[float] = None  # mean force/gradient magnitude
    energy: Optional[float] = None  # mean energy, when one exists
    energy_kind: str = "empirical"  # "exact" | "empirical"


@dataclass
class BasinEvent:
    """A spawn/merge/prune/seed event in the basin field."""

    step: int
    kind: str  # "spawn" | "merge" | "prune" | "seed"
    label: str
    count: int


class DynamicsLedger:
    """Records collapse dynamics step by step.

    labels: anchor/label names, in engine order. For dynamic collapse the
        single align value per step is alignment to the routed target.
    convergence_tol: mean displacement below this counts as converged.
    """

    def __init__(
        self,
        labels: Optional[Sequence[str]] = None,
        convergence_tol: float = 1e-4,
    ):
        self.labels: List[str] = list(labels) if labels else []
        self.convergence_tol = convergence_tol
        self.mode: str = ""
        self.steps: List[StepRecord] = []
        self.events: List[BasinEvent] = []
        self.basin_selection: Dict[str, List[int]] = {}  # label -> chosen slot per sample
        self.converged: bool = False
        self.convergence_reason: str = ""
        self.meta: Dict = {}

    # ---- recording ----

    @staticmethod
    def _mean(x: torch.Tensor) -> float:
        return float(x.detach().float().mean().item())

    def log_step(
        self,
        step: int,
        h: torch.Tensor,
        h_prev: torch.Tensor,
        align: torch.Tensor,
        force: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_kind: str = "empirical",
    ) -> None:
        """align: (B, L) per-anchor or (B,)/(B, 1) per-target alignment."""
        a = align.detach().float()
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        self.steps.append(
            StepRecord(
                step=step,
                norm=self._mean(h.norm(dim=-1)),
                displacement=self._mean((h - h_prev).norm(dim=-1)),
                align=[float(v) for v in a.mean(dim=0).tolist()],
                force=self._mean(force.norm(dim=-1)) if force is not None else None,
                energy=self._mean(energy) if energy is not None else None,
                energy_kind=energy_kind,
            )
        )

    def log_event(self, step: int, kind: str, label: str, count: int) -> None:
        if count > 0:
            self.events.append(BasinEvent(step=step, kind=kind, label=label, count=count))

    def log_basin_selection(self, label: str, slots: torch.Tensor) -> None:
        self.basin_selection[label] = [int(s) for s in slots.detach().tolist()]

    def finish(self, reason: Optional[str] = None) -> None:
        """Mark the run finished. Auto-detects convergence if reason is None."""
        if reason is None:
            if self.steps and self.steps[-1].displacement < self.convergence_tol:
                reason = "converged"
            else:
                reason = "max_steps"
        self.convergence_reason = reason
        self.converged = reason in ("converged", "closed_form")

    # ---- analysis ----

    def energy_monotone(self, atol: float = 1e-6) -> bool:
        """True iff every recorded exact energy is non-increasing.

        Only meaningful for energy_kind="exact"; raises if no exact energies
        were recorded, so an empirical run can't masquerade as proven descent.
        """
        e = [s.energy for s in self.steps if s.energy is not None and s.energy_kind == "exact"]
        if not e:
            raise ValueError("no exact energies recorded; nothing to check")
        return all(b <= a + atol for a, b in zip(e, e[1:]))

    def summary(self) -> Dict:
        d: Dict = {
            "mode": self.mode,
            "num_steps": len(self.steps),
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "num_events": len(self.events),
        }
        if self.steps:
            last = self.steps[-1]
            d["final_norm"] = last.norm
            d["final_displacement"] = last.displacement
            d["final_align"] = last.align
        exact = [s.energy for s in self.steps if s.energy is not None and s.energy_kind == "exact"]
        if exact:
            d["energy_start"] = exact[0]
            d["energy_end"] = exact[-1]
            d["energy_monotone"] = self.energy_monotone()
        return d

    # ---- export ----

    def to_dict(self) -> Dict:
        return {
            "meta": self.meta,
            "labels": self.labels,
            "summary": self.summary(),
            "steps": [asdict(s) for s in self.steps],
            "events": [asdict(e) for e in self.events],
            "basin_selection": self.basin_selection,
        }

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def to_csv(self, path) -> None:
        """One row per step; alignment flattened to align_<label> columns."""
        n_align = len(self.steps[0].align) if self.steps else 0
        names = self.labels or [str(i) for i in range(n_align)]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            align_cols = [f"align_{names[i] if i < len(names) else i}" for i in range(n_align)]
            w.writerow(
                ["step", "norm", "displacement", "force", "energy", "energy_kind", *align_cols]
            )
            for s in self.steps:
                w.writerow(
                    [s.step, s.norm, s.displacement, s.force, s.energy, s.energy_kind, *s.align]
                )

    def to_markdown(self, path=None) -> str:
        """Render a human-readable report; optionally write it to `path`."""
        sm = self.summary()
        lines = ["# Collapse dynamics report", ""]
        lines.append(f"- mode: `{sm.get('mode') or 'unknown'}`")
        lines.append(f"- steps: {sm['num_steps']}")
        lines.append(f"- converged: {sm['converged']} ({sm['convergence_reason']})")
        if "energy_monotone" in sm:
            verdict = (
                "yes" if sm["energy_monotone"] else "**NO — dynamics violated their own potential**"
            )
            lines.append(
                f"- exact energy: {sm['energy_start']:.6f} -> {sm['energy_end']:.6f}, "
                f"monotone: {verdict}"
            )
        else:
            lines.append("- energy: empirical only (no exact potential for this mode)")
        if self.meta:
            lines.append(f"- meta: {json.dumps(self.meta)}")

        if self.steps:
            names = self.labels or [str(i) for i in range(len(self.steps[0].align))]
            n_align = len(self.steps[0].align)
            align_hdr = " | ".join(
                f"align_{names[i] if i < len(names) else i}" for i in range(n_align)
            )
            lines += ["", "## Steps", "", f"| step | norm | disp | force | energy | {align_hdr} |"]
            lines.append("|" + "---|" * (5 + n_align))
            for s in self.steps:
                force = f"{s.force:.4f}" if s.force is not None else "-"
                energy = f"{s.energy:.6f}" if s.energy is not None else "-"
                aligns = " | ".join(f"{a:.4f}" for a in s.align)
                lines.append(
                    f"| {s.step} | {s.norm:.4f} | {s.displacement:.6f} "
                    f"| {force} | {energy} | {aligns} |"
                )

        if self.events:
            lines += ["", "## Basin events", ""]
            lines += ["| step | kind | label | count |", "|---|---|---|---|"]
            for e in self.events:
                lines.append(f"| {e.step} | {e.kind} | {e.label} | {e.count} |")

        if self.basin_selection:
            lines += ["", "## Basin selection", ""]
            for label, slots in self.basin_selection.items():
                used = sorted(set(slots))
                lines.append(f"- `{label}`: {len(slots)} samples over slots {used}")

        text = "\n".join(lines) + "\n"
        if path is not None:
            Path(path).write_text(text)
        return text
