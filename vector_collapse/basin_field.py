"""Dynamic basin field: per-label micro-basins (vectorized).

Basins are routed to, updated, spawned, and pruned during training.
Centers live in registered buffers so they persist in state_dict but are
never touched by the optimizer.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasinField(nn.Module):
    """Pre-allocated per-label anchor slots for fully vectorized routing."""

    def __init__(self, dim: int = 256, max_basins_per_label: int = 64, num_labels: int = 3):
        super().__init__()
        self.dim = dim
        self.num_labels = num_labels
        self.max_basins_per_label = max_basins_per_label

        # (L, K, dim); label index order is defined by CollapseConfig.labels
        self.register_buffer("centers", torch.zeros(num_labels, max_basins_per_label, dim))
        self.register_buffer(
            "active", torch.zeros(num_labels, max_basins_per_label, dtype=torch.bool)
        )
        self.register_buffer(
            "counts", torch.zeros(num_labels, max_basins_per_label, dtype=torch.int32)
        )
        self.register_buffer(
            "last_used", torch.zeros(num_labels, max_basins_per_label, dtype=torch.int32)
        )

    def get_active_centers(self, label_idx: int) -> torch.Tensor:
        """(K_active, dim) centers for one label."""
        return self.centers[label_idx][self.active[label_idx]]

    def add_basin(self, label_idx: int, vector: torch.Tensor, step: int) -> bool:
        """Add a basin in the first free slot. Returns False if full."""
        inactive = torch.nonzero(~self.active[label_idx], as_tuple=True)[0]
        if len(inactive) == 0:
            return False
        idx = inactive[0].item()
        self.centers[label_idx, idx] = F.normalize(vector.detach(), dim=0)
        self.active[label_idx, idx] = True
        self.counts[label_idx, idx] = 0
        self.last_used[label_idx, idx] = step
        return True


def route_to_basin_vectorized(
    field: BasinField, h: torch.Tensor, label_idx: int, step: int, training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route a batch of same-label vectors to their nearest active basin.

    Returns (target_centers, align, divergence, tension, found_mask).
    """
    B = h.size(0)
    device = h.device
    mask = field.active[label_idx]

    h_n = F.normalize(h, dim=-1)

    if not mask.any():
        zeros = torch.zeros(B, device=device)
        return (
            torch.zeros_like(h_n),
            zeros,
            zeros,
            zeros,
            torch.zeros(B, dtype=torch.bool, device=device),
        )

    active_centers = field.centers[label_idx][mask]  # (K, dim)
    sims = torch.matmul(h_n, active_centers.t())  # (B, K)
    best_sims, best_local = torch.max(sims, dim=1)  # (B,)

    active_global = torch.nonzero(mask, as_tuple=True)[0]
    best_global = active_global[best_local]
    target_centers = field.centers[label_idx, best_global]  # (B, dim)

    # Attractive law: div = 1 - align, zero only on the anchor.
    # Must stay consistent with engine.divergence_from_alignment.
    divergence = 1.0 - best_sims
    tens = divergence.abs()

    if training:
        unique_idxs, counts = torch.unique(best_global, return_counts=True)
        field.counts[label_idx].index_add_(0, unique_idxs, counts.to(field.counts.dtype))
        field.last_used[label_idx][unique_idxs] = step

    return (
        target_centers,
        best_sims,
        divergence,
        tens,
        torch.ones(B, dtype=torch.bool, device=device),
    )


def maybe_spawn_vectorized(
    field: BasinField,
    h: torch.Tensor,
    label_idx: int,
    tens: torch.Tensor,
    align: torch.Tensor,
    step: int,
    tension_threshold: float,
    align_threshold: float,
) -> int:
    """Spawn new basins where tension is high and alignment poor.

    Returns the number of basins spawned (0 if none or field full).
    """
    spawn_mask = (tens > tension_threshold) & (align < align_threshold)
    if not spawn_mask.any():
        return 0
    spawned = 0
    for idx in torch.nonzero(spawn_mask, as_tuple=True)[0]:
        if not field.add_basin(label_idx, h[idx], step):
            break  # field is full; stop early instead of looping uselessly
        spawned += 1
    return spawned


def prune_and_merge_vectorized(
    field: BasinField, min_count: int = 10, merge_cos_threshold: float = 0.97
) -> Tuple[int, int]:
    """Prune rarely used basins, merge near-duplicate ones (count-weighted).

    Returns (num_pruned, num_merged).
    """
    n_pruned = 0
    n_merged = 0
    for l_idx in range(field.num_labels):
        mask = field.active[l_idx]
        if not mask.any():
            continue

        to_prune = (field.counts[l_idx] < min_count) & mask
        if to_prune.any():
            n_pruned += int(to_prune.sum().item())
            field.active[l_idx][to_prune] = False
            field.counts[l_idx][to_prune] = 0
            field.last_used[l_idx][to_prune] = 0
            field.centers[l_idx][to_prune] = 0.0

        active_idxs = torch.nonzero(field.active[l_idx], as_tuple=True)[0]
        if len(active_idxs) < 2:
            continue

        c_norm = F.normalize(field.centers[l_idx][active_idxs], dim=1)
        sims = torch.mm(c_norm, c_norm.t())
        merged = torch.zeros(len(active_idxs), dtype=torch.bool, device=field.centers.device)

        for i in range(len(active_idxs)):
            if merged[i]:
                continue
            for j in range(i + 1, len(active_idxs)):
                if merged[j] or sims[i, j] <= merge_cos_threshold:
                    continue
                idx_i, idx_j = active_idxs[i], active_idxs[j]
                total = field.counts[l_idx, idx_i] + field.counts[l_idx, idx_j]
                w_i = field.counts[l_idx, idx_i].float() / total.float()
                w_j = field.counts[l_idx, idx_j].float() / total.float()
                new_center = w_i * field.centers[l_idx, idx_i] + w_j * field.centers[l_idx, idx_j]
                field.centers[l_idx, idx_i] = F.normalize(new_center, dim=0).detach()
                field.counts[l_idx, idx_i] = total
                field.active[l_idx, idx_j] = False
                field.counts[l_idx, idx_j] = 0
                merged[j] = True
                n_merged += 1
    return n_pruned, n_merged
