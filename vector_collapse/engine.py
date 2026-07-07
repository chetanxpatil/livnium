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
        """Collapse under all per-label anchors simultaneously."""
        h = h0.clone()
        if h.dim() == 1:
            h = h.unsqueeze(0)

        anchor_dirs = F.normalize(self.anchors, dim=-1)  # (L, dim)

        if self.cfg.mode == "attention_projection":
            # O(1) attention/Hopfield lookup step
            h_n = F.normalize(h, dim=-1)
            align = torch.matmul(h_n, anchor_dirs.t())  # (B, L)
            w = F.softmax(self.cfg.beta * align, dim=-1)  # (B, L)
            h_infty = torch.matmul(w, anchor_dirs)  # (B, dim)
            return self._clamp_norm(h_infty * self.cfg.max_norm), {}

        elif self.cfg.mode == "gradient_descent":
            # Pure analytical energy gradient descent (No MLP)
            alpha = self.cfg.alpha
            beta = self.cfg.beta
            for _ in range(self.num_layers):
                h_norm = h.norm(dim=-1, keepdim=True)
                h_n = h / (h_norm + 1e-8)
                align = torch.matmul(h_n, anchor_dirs.t())  # (B, L)
                w = F.softmax(beta * align, dim=-1)  # (B, L)
                
                term1 = w.unsqueeze(-1) * anchor_dirs.unsqueeze(0)  # (B, L, dim)
                term2 = w.unsqueeze(-1) * h_n.unsqueeze(1) * align.unsqueeze(-1)  # (B, L, dim)
                
                grad_V = -(term1 - term2).sum(dim=1) / (h_norm + 1e-8)  # (B, dim)
                h = self._clamp_norm(h - alpha * grad_V)
            return h, {}

        else:
            # mlp_legacy: old MLP-residual + hand-designed away force
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
                centers_sub, align_sub, _, tens_sub, found = route_to_basin_vectorized(
                    basin_field, sub_h, l_idx, global_step, training=update_anchors
                )

            target_centers[mask] = centers_sub

            if spawn_new:
                maybe_spawn_vectorized(
                    basin_field, sub_h, l_idx, tens_sub, align_sub, global_step,
                    self.cfg.basin.tension_threshold, self.cfg.basin.align_threshold,
                )

        # 2. Dynamics: attract each h toward its fixed target center.
        trace: Dict[str, List[torch.Tensor]] = {"align": [], "div": [], "tens": []}
        strengths = strengths.unsqueeze(-1)                     # (B, 1)

        for _ in range(self.num_layers):
            h_norm = h.norm(dim=-1, keepdim=True)
            h_n = h / (h_norm + 1e-8)
            align = (h_n * target_centers).sum(dim=1, keepdim=True)  # (B, 1)

            if self.cfg.mode == "gradient_descent":
                # Analytical gradient of V(h) = -cos(h, T)
                grad = -(target_centers - h_n * align) / (h_norm + 1e-8)
                h = self._clamp_norm(h - self.cfg.alpha * grad)
            elif self.cfg.mode == "attention_projection":
                # O(1) attention/projection
                h = F.normalize(target_centers, dim=-1) * self.cfg.max_norm
                break
            else:
                delta = self.update(h)
                away = F.normalize(h - target_centers, dim=-1)      # anchor -> h
                div = divergence_from_alignment(align.squeeze(-1))
                h = self._clamp_norm(h + delta - strengths * div.unsqueeze(-1) * away)

        # 3. Anchor update: moving average of final positions per basin.
        if update_anchors:
            lr = self.cfg.basin.anchor_lr
            for l_idx in torch.unique(labels):
                l_idx = int(l_idx.item())
                mask_l = labels == l_idx
                all_centers = basin_field.centers[l_idx]        # (K, dim)
                h_final_n = F.normalize(h[mask_l].detach(), dim=-1)
                best = torch.argmax(torch.matmul(h_final_n, all_centers.t()), dim=1)
                for k in torch.unique(best):
                    mean_vec = h_final_n[best == k].mean(dim=0)
                    all_centers[k] = F.normalize((1 - lr) * all_centers[k] + lr * mean_vec, dim=0)

        # 4. Pruning
        if prune_every > 0 and global_step > 0 and global_step % prune_every == 0:
            prune_and_merge_vectorized(
                basin_field, self.cfg.basin.prune_min_count, self.cfg.basin.prune_merge_cos
            )

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
