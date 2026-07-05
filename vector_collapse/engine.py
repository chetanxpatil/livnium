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
    route_all_labels_fused,
    route_to_basin_vectorized,
)
from .config import CollapseConfig


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
        return torch.where(
            h_norm > self.cfg.max_norm, h * (self.cfg.max_norm / (h_norm + 1e-8)), h
        )

    # ---- static collapse ----

    def collapse(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """Static collapse under all per-label anchors simultaneously."""
        h = h0.clone()
        if h.dim() == 1:
            h = h.unsqueeze(0)

        anchor_dirs = F.normalize(self.anchors, dim=-1)  # (L, dim)

        for _ in range(self.num_layers):
            h_n = F.normalize(h, dim=-1)
            align = torch.matmul(h_n, anchor_dirs.t())          # (B, L)
            div = divergence_from_alignment(align)              # (B, L)

            delta = self.update(h)

            # (B, L, dim): direction from each anchor to h
            away = F.normalize(h.unsqueeze(1) - anchor_dirs.unsqueeze(0), dim=-1)
            # force summed over labels, weighted by per-label strength
            force = (self.strengths.view(1, -1, 1) * div.unsqueeze(-1) * away).sum(dim=1)

            h = self._clamp_norm(h + delta - force)

        return h, {}

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
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """Collapse each sample toward its label's nearest basin.

        h0: (B, dim), labels: (B,) integer-encoded per cfg.labels order.
        """
        h = h0.clone()
        device = h.device

        if basin_field.centers.device != device:
            basin_field.to(device)

        strengths = self.strengths.to(device)[labels]           # (B,)

        # 1a. Seed: any label present in the batch but without a basin gets
        # one, seeded from its first sample (cheap check, no matmul).
        for l_idx in torch.unique(labels):
            l_idx = int(l_idx.item())
            if not basin_field.active[l_idx].any():
                first = torch.nonzero(labels == l_idx, as_tuple=True)[0][0]
                basin_field.add_basin(l_idx, h[first].detach(), global_step)

        # 1b. Routing: ONE matmul for the whole mixed-label batch.
        # Fixed target anchor per sample for the whole collapse.
        target_centers, align_all, _, tens_all, _ = route_all_labels_fused(
            basin_field, h, labels, global_step, training=update_anchors
        )

        if spawn_new:
            for l_idx in torch.unique(labels):
                l_idx = int(l_idx.item())
                mask = labels == l_idx
                maybe_spawn_vectorized(
                    basin_field, h[mask], l_idx, tens_all[mask], align_all[mask],
                    global_step,
                    self.cfg.basin.tension_threshold, self.cfg.basin.align_threshold,
                )

        # 2. Dynamics: attract each h toward its fixed target center.
        trace: Dict[str, List[torch.Tensor]] = {"align": [], "div": [], "tens": []}
        strengths = strengths.unsqueeze(-1)                     # (B, 1)

        for _ in range(self.num_layers):
            h_n = F.normalize(h, dim=-1)
            align = (h_n * target_centers).sum(dim=1)           # (B,)
            div = divergence_from_alignment(align)

            delta = self.update(h)
            away = F.normalize(h - target_centers, dim=-1)      # anchor -> h
            h = self._clamp_norm(h + delta - strengths * div.unsqueeze(-1) * away)

        # 3. Anchor update: moving average of final positions per basin.
        # One matmul to re-match + index_add scatter — no loops over basins.
        if update_anchors:
            self._update_anchors_fused(h, labels, basin_field)

        # 4. Pruning
        if prune_every > 0 and global_step > 0 and global_step % prune_every == 0:
            prune_and_merge_vectorized(
                basin_field, self.cfg.basin.prune_min_count, self.cfg.basin.prune_merge_cos
            )

        return h, trace

    @torch.no_grad()
    def _update_anchors_fused(
        self, h: torch.Tensor, labels: torch.Tensor, field: BasinField
    ) -> None:
        """EMA-update every touched basin center in one matmul + scatter.

        Re-matches final states to their nearest same-label active basin
        ((B, L*K) matmul), then accumulates per-basin means with index_add —
        replaces the old Python double loop over labels x unique basins.
        """
        B = h.size(0)
        device = h.device
        L, K, D = field.centers.shape

        h_n = F.normalize(h.detach(), dim=-1)                       # (B, D)
        centers_flat = field.centers.view(L * K, D)
        sims = torch.matmul(h_n, centers_flat.t())                  # (B, L*K)

        basin_labels = torch.arange(L, device=device).repeat_interleave(K)
        valid = field.active.view(1, L * K) & (basin_labels.view(1, -1) == labels.view(B, 1))
        sims = sims.masked_fill(~valid, -2.0)
        best = sims.argmax(dim=1)                                   # (B,)
        found = valid.any(dim=1)
        best, h_n = best[found], h_n[found]

        # per-basin sum and count via scatter
        sums = torch.zeros(L * K, D, device=device).index_add_(0, best, h_n)
        cnts = torch.zeros(L * K, device=device).index_add_(
            0, best, torch.ones(best.size(0), device=device)
        )
        touched = cnts > 0
        means = sums[touched] / cnts[touched].unsqueeze(-1)

        lr = self.cfg.basin.anchor_lr
        centers_flat[touched] = F.normalize(
            (1 - lr) * centers_flat[touched] + lr * means, dim=-1
        )

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
