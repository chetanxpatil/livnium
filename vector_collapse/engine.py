"""Vector collapse engine, driven by CollapseConfig.

Two modes:
- static collapse: one learned anchor per label (legacy behavior)
- collapse_dynamic: routes each sample to per-label basins (BasinField) and
  optionally spawns/prunes anchors during training.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basin_field import (
    BasinField,
    maybe_spawn_vectorized,
    prune_and_merge_vectorized,
    route_to_basin_vectorized,
)
from .config import CollapseConfig
from .ledger import DynamicsLedger


def divergence_from_alignment(align: torch.Tensor) -> torch.Tensor:
    """div = 1 - align: always attractive, vanishing exactly on the anchor.

    The collapse update applies -strength * div * normalize(h - anchor), so
    div >= 0 everywhere with div == 0 only at align == 1 makes the anchor a
    genuine point attractor. (The old law `0.38 - align` had its equilibrium
    on a shell ~68 degrees off the anchor and turned repulsive past it —
    see COLLAPSE_ENGINE_VERDICT.md.)
    """
    return 1.0 - align


def tension(divergence: torch.Tensor) -> torch.Tensor:
    return divergence.abs()


class VectorCollapseEngine(nn.Module):
    def __init__(self, config: Optional[CollapseConfig] = None):
        super().__init__()
        cfg = config or CollapseConfig()
        self.cfg = cfg
        self.dim = cfg.dim
        self.num_layers = cfg.num_layers

        # (L,) per-label collapse strengths, ordered like cfg.labels
        self.register_buffer("strengths", cfg.strength_tensor())

        self.update = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
            nn.Tanh(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        # Zero-init the LAST linear so the residual block starts as identity.
        # At default init the random MLP delta overpowered the collapse force
        # ~37x and detonated every state into the norm clamp on step one.
        # See COLLAPSE_ENGINE_VERDICT.md.
        nn.init.zeros_(self.update[2].weight)
        nn.init.zeros_(self.update[2].bias)

        # (L, dim) static anchors, one per label
        self.anchors = nn.Parameter(torch.randn(cfg.num_labels, cfg.dim))

    @classmethod
    def from_yaml(cls, path) -> "VectorCollapseEngine":
        return cls(CollapseConfig.from_yaml(path))

    def make_basin_field(self) -> BasinField:
        return BasinField(
            dim=self.cfg.dim,
            max_basins_per_label=self.cfg.basin.max_basins_per_label,
            num_labels=self.cfg.num_labels,
        )

    def _clamp_norm(self, h: torch.Tensor) -> torch.Tensor:
        h_norm = h.norm(p=2, dim=-1, keepdim=True)
        return torch.where(h_norm > self.cfg.max_norm, h * (self.cfg.max_norm / (h_norm + 1e-8)), h)

    # ---- static collapse ----

    def _static_energy(self, h_n: torch.Tensor, align: torch.Tensor) -> torch.Tensor:
        """Exact potential for static gradient_collapse.

        V(h) = -(1/beta) * logsumexp(beta * align): the softmax-weighted
        gradient in the loop below is exactly -dV/dh, so this value must be
        non-increasing along the trajectory. The ledger checks that.
        """
        return -torch.logsumexp(self.cfg.beta * align, dim=-1) / self.cfg.beta

    def collapse(
        self,
        h0: torch.Tensor,
        ledger: Optional[DynamicsLedger] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """Collapse under all per-label anchors simultaneously.

        Pass a DynamicsLedger to record norms, displacement, per-anchor
        alignment, force and (for gradient_collapse) exact energy per step.
        Logging costs nothing when ledger is None.
        """
        h = h0.clone()
        if h.dim() == 1:
            h = h.unsqueeze(0)

        anchor_dirs = F.normalize(self.anchors, dim=-1)  # (L, dim)

        if ledger is not None:
            ledger.mode = self.cfg.mode
            if not ledger.labels:
                ledger.labels = list(self.cfg.labels)

        def log(step, h_new, h_old, force=None, exact_energy=False):
            if ledger is None:
                return
            with torch.no_grad():  # observation must not touch the graph
                h_n = F.normalize(h_new, dim=-1)
                align = torch.matmul(h_n, anchor_dirs.t())
                energy = self._static_energy(h_n, align) if exact_energy else None
                ledger.log_step(
                    step,
                    h_new,
                    h_old,
                    align,
                    force=force,
                    energy=energy,
                    energy_kind="exact" if exact_energy else "empirical",
                )

        if self.cfg.mode == "direct_collapse":
            # O(1) closed-form direct collapse step
            h_n = F.normalize(h, dim=-1)
            align = torch.matmul(h_n, anchor_dirs.t())  # (B, L)
            w = F.softmax(self.cfg.beta * align, dim=-1)  # (B, L)
            h_infty = torch.matmul(w, anchor_dirs)  # (B, dim)
            out = self._clamp_norm(h_infty * self.cfg.max_norm)
            if ledger is not None:
                log(0, h, h)
                log(1, out, h)
                ledger.finish("closed_form")
            return out, {}

        elif self.cfg.mode == "gradient_collapse":
            # Pure analytical energy gradient descent collapse (No MLP)
            alpha = self.cfg.alpha
            beta = self.cfg.beta
            log(0, h, h, exact_energy=True)
            for i in range(self.num_layers):
                h_norm = h.norm(dim=-1, keepdim=True)
                h_n = h / (h_norm + 1e-8)
                align = torch.matmul(h_n, anchor_dirs.t())  # (B, L)
                w = F.softmax(beta * align, dim=-1)  # (B, L)

                term1 = w.unsqueeze(-1) * anchor_dirs.unsqueeze(0)  # (B, L, dim)
                term2 = w.unsqueeze(-1) * h_n.unsqueeze(1) * align.unsqueeze(-1)  # (B, L, dim)

                grad_V = -(term1 - term2).sum(dim=1) / (h_norm + 1e-8)  # (B, dim)
                h_new = self._clamp_norm(h - alpha * grad_V)
                log(i + 1, h_new, h, force=grad_V, exact_energy=True)
                h = h_new
            if ledger is not None:
                ledger.finish()
            return h, {}

        elif self.cfg.mode == "mlp_collapse":
            # mlp_collapse: old MLP-residual + hand-designed away force
            log(0, h, h)
            for i in range(self.num_layers):
                h_n = F.normalize(h, dim=-1)
                align = torch.matmul(h_n, anchor_dirs.t())  # (B, L)
                div = divergence_from_alignment(align)  # (B, L)

                delta = self.update(h)

                # (B, L, dim): direction from each anchor to h
                away = F.normalize(h.unsqueeze(1) - anchor_dirs.unsqueeze(0), dim=-1)
                # force summed over labels, weighted by per-label strength
                force = (self.strengths.view(1, -1, 1) * div.unsqueeze(-1) * away).sum(dim=1)

                h_new = self._clamp_norm(h + delta - force)
                log(i + 1, h_new, h, force=force)
                h = h_new
            if ledger is not None:
                ledger.finish()
            return h, {}
        else:
            raise ValueError(f"Unknown collapse mode: {self.cfg.mode}")

    def forward(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        return self.collapse(h0)

    # ---- dynamic collapse ----

    def collapse_dynamic(
        self,
        h0: torch.Tensor,
        labels: torch.Tensor,
        basin_field: BasinField,
        global_step: int = 0,
        spawn_new: bool = True,
        prune_every: int = 0,
        update_anchors: bool = True,
        ledger: Optional[DynamicsLedger] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """Collapse each sample toward its label's nearest basin.

        h0: (B, dim), labels: (B,) integer-encoded per cfg.labels order.
        Pass a DynamicsLedger to record dynamics, basin selection and
        spawn/seed/prune/merge events. The returned trace holds per-step
        (B,) tensors of alignment/divergence/tension to the routed target.

        Ledger honesty: alignment here is to a routed, moving target, so
        gradient-mode energy (-cos to target) is exact only per collapse
        call; across training steps targets move. Chord/mlp mode records
        empirical observations only.
        """
        h = h0.clone()
        device = h.device

        if basin_field.centers.device != device:
            basin_field.to(device)

        if ledger is not None:
            ledger.mode = self.cfg.mode
            if not ledger.labels:
                ledger.labels = ["target"]
            ledger.meta.setdefault("dynamic", True)
            ledger.meta.setdefault("global_step", global_step)

        strengths = self.strengths.to(device)[labels]  # (B,)
        target_centers = torch.zeros_like(h)

        # 1. Routing: fixed target anchor per sample for the whole collapse.
        for l_idx in range(self.cfg.num_labels):
            mask = labels == l_idx
            if not mask.any():
                continue
            sub_h = h[mask]

            centers_sub, align_sub, _, tens_sub, found = route_to_basin_vectorized(
                basin_field, sub_h, l_idx, global_step, training=update_anchors
            )
            if (~found).any():
                # No basin yet for this label: seed one and re-route.
                first = torch.nonzero(~found, as_tuple=True)[0][0]
                basin_field.add_basin(l_idx, sub_h[first].detach(), global_step)
                if ledger is not None:
                    ledger.log_event(global_step, "seed", self.cfg.labels[l_idx], 1)
                centers_sub, align_sub, _, tens_sub, found = route_to_basin_vectorized(
                    basin_field, sub_h, l_idx, global_step, training=update_anchors
                )

            target_centers[mask] = centers_sub

            if ledger is not None:
                # Which basin slot each sample routed to (recomputed only
                # when observing; costs one small matmul).
                with torch.no_grad():
                    sub_n = F.normalize(sub_h.detach(), dim=-1)
                    sims = torch.matmul(sub_n, basin_field.centers[l_idx].t())
                    sims[:, ~basin_field.active[l_idx]] = -2.0
                    ledger.log_basin_selection(self.cfg.labels[l_idx], sims.argmax(dim=1))

            if spawn_new:
                spawned = maybe_spawn_vectorized(
                    basin_field,
                    sub_h,
                    l_idx,
                    tens_sub,
                    align_sub,
                    global_step,
                    self.cfg.basin.tension_threshold,
                    self.cfg.basin.align_threshold,
                )
                if ledger is not None:
                    ledger.log_event(global_step, "spawn", self.cfg.labels[l_idx], spawned)

        # 2. Dynamics: attract each h toward its fixed target center.
        trace: Dict[str, List[torch.Tensor]] = {"align": [], "div": [], "tens": []}
        strengths = strengths.unsqueeze(-1)  # (B, 1)

        def record(step, h_new, h_old, align_col, force=None):
            """Fill the returned trace and, if present, the ledger."""
            a = align_col.squeeze(-1).detach()
            d = divergence_from_alignment(a)
            trace["align"].append(a)
            trace["div"].append(d)
            trace["tens"].append(tension(d))
            if ledger is not None:
                exact = self.cfg.mode == "gradient_collapse"
                ledger.log_step(
                    step,
                    h_new,
                    h_old,
                    a,
                    force=force,
                    energy=-a if exact else None,  # V(h) = -cos(h, target)
                    energy_kind="exact" if exact else "empirical",
                )

        closed_form = False
        for i in range(self.num_layers):
            h_norm = h.norm(dim=-1, keepdim=True)
            h_n = h / (h_norm + 1e-8)
            align = (h_n * target_centers).sum(dim=1, keepdim=True)  # (B, 1)

            if self.cfg.mode == "gradient_collapse":
                # Analytical gradient of V(h) = -cos(h, T)
                grad = -(target_centers - h_n * align) / (h_norm + 1e-8)
                h_new = self._clamp_norm(h - self.cfg.alpha * grad)
                record(i, h_new, h, align, force=grad)
                h = h_new
            elif self.cfg.mode == "direct_collapse":
                # O(1) direct collapse
                h_new = F.normalize(target_centers, dim=-1) * self.cfg.max_norm
                record(i, h_new, h, align)
                h = h_new
                closed_form = True
                break
            elif self.cfg.mode == "mlp_collapse":
                delta = self.update(h)
                away = F.normalize(h - target_centers, dim=-1)  # anchor -> h
                div = divergence_from_alignment(align.squeeze(-1))
                force = strengths * div.unsqueeze(-1) * away
                h_new = self._clamp_norm(h + delta - force)
                record(i, h_new, h, align, force=force)
                h = h_new
            else:
                raise ValueError(f"Unknown collapse mode: {self.cfg.mode}")

        # 3. Anchor update: moving average of final positions per basin.
        if update_anchors:
            lr = self.cfg.basin.anchor_lr
            for l_idx in torch.unique(labels):
                l_idx = int(l_idx.item())
                mask_l = labels == l_idx
                all_centers = basin_field.centers[l_idx]  # (K, dim)
                h_final_n = F.normalize(h[mask_l].detach(), dim=-1)
                best = torch.argmax(torch.matmul(h_final_n, all_centers.t()), dim=1)
                for k in torch.unique(best):
                    mean_vec = h_final_n[best == k].mean(dim=0)
                    all_centers[k] = F.normalize((1 - lr) * all_centers[k] + lr * mean_vec, dim=0)

        # 4. Pruning
        if prune_every > 0 and global_step > 0 and global_step % prune_every == 0:
            n_pruned, n_merged = prune_and_merge_vectorized(
                basin_field, self.cfg.basin.prune_min_count, self.cfg.basin.prune_merge_cos
            )
            if ledger is not None:
                ledger.log_event(global_step, "prune", "*", n_pruned)
                ledger.log_event(global_step, "merge", "*", n_merged)

        if ledger is not None:
            ledger.finish("closed_form" if closed_form else None)

        return h, trace

    # ---- legacy checkpoint support ----

    def load_legacy_state_dict(self, state_dict: dict) -> None:
        """Load weights saved by collapse_retrain/vector_collapse.py
        (anchor_entail/anchor_contra/anchor_neutral -> anchors rows E/C/N)."""
        sd = dict(state_dict)
        legacy = ("anchor_entail", "anchor_contra", "anchor_neutral")
        if all(k in sd for k in legacy):
            sd["anchors"] = torch.stack([sd.pop(k) for k in legacy])
        sd.setdefault("strengths", self.strengths.clone())
        self.load_state_dict(sd)
