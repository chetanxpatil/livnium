"""
Dynamic Basin Field for quantum_embed (Vectorized)

Matches nova_v3 semantics: per-label micro-basins that can be routed to,
updated, spawned, and pruned during training.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ("E", "N", "C")
LABEL_TO_IDX = {"E": 0, "N": 1, "C": 2}


class BasinField(nn.Module):
    """
    Collection of anchors per label using pre-allocated tensors for vectorization.
    """

    def __init__(self, dim: int = 256, max_basins_per_label: int = 64):
        super().__init__()
        self.dim = dim
        self.max_basins_per_label = max_basins_per_label
        
        # We store centers as buffers so they are part of state_dict but not optimized by Adam
        # Shape: (3, max_basins, dim) - 0=E, 1=N, 2=C
        self.register_buffer("centers", torch.zeros(3, max_basins_per_label, dim))
        
        # Track which basins are active (1=active, 0=inactive)
        self.register_buffer("active", torch.zeros(3, max_basins_per_label, dtype=torch.bool))
        
        # Track counts/usage
        self.register_buffer("counts", torch.zeros(3, max_basins_per_label, dtype=torch.int32))
        self.register_buffer("last_used", torch.zeros(3, max_basins_per_label, dtype=torch.int32))
        
    def to_device(self, device):
        self.to(device)

    def get_active_centers(self, label_idx: int) -> torch.Tensor:
        """Returns (K, dim) tensor of active centers for a label index."""
        mask = self.active[label_idx]
        return self.centers[label_idx][mask]

    def add_basin(self, label_idx: int, vector: torch.Tensor, step: int):
        """Adds a new basin if space permits. vector: (dim,)"""
        # CRITICAL FIX: Ensure vector is detached from graph before storage
        vector = vector.detach()
        
        # Find first inactive slot
        mask = self.active[label_idx]
        # ~mask is true for inactive
        inactive_indices = torch.nonzero(~mask, as_tuple=True)[0]
        
        if len(inactive_indices) > 0:
            idx = inactive_indices[0].item()
            self.centers[label_idx, idx] = F.normalize(vector, dim=0)
            self.active[label_idx, idx] = True
            self.counts[label_idx, idx] = 0
            self.last_used[label_idx, idx] = step


def route_to_basin_vectorized(
    field: BasinField, 
    h: torch.Tensor, 
    label_idx: int, 
    step: int,
    training: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized routing for a BATCH of vectors all belonging to the SAME label.
    """
    B = h.size(0)
    device = h.device
    mask = field.active[label_idx]
    
    num_active = mask.sum().item()
    
    h_n = F.normalize(h, dim=-1) # (B, dim)
    
    if num_active == 0:
        # No basins exist. Return dummies.
        dummy_centers = torch.zeros_like(h_n)
        dummy_vals = torch.zeros(B, device=device)
        return dummy_centers, dummy_vals, dummy_vals, dummy_vals, torch.zeros(B, dtype=torch.bool, device=device)

    # Get active centers: (K, dim)
    active_centers = field.centers[label_idx][mask] # (K, dim)
    
    # Compute dot products: (B, dim) @ (dim, K) -> (B, K)
    sims = torch.matmul(h_n, active_centers.t())
    
    # Find best match for each item in batch
    best_sims, best_indices_local = torch.max(sims, dim=1) # (B,)
    
    # Map local indices (0..K-1) back to global indices (0..Max-1)
    active_global_indices = torch.nonzero(mask, as_tuple=True)[0] # (K,)
    best_indices_global = active_global_indices[best_indices_local] # (B,)
    
    # Construct target centers
    target_centers = field.centers[label_idx, best_indices_global] # (B, dim)
    
    # Metrics
    best_align = best_sims
    divergence = 0.38 - best_align
    tens = divergence.abs()
    
    # Update stats if training
    if training:
        unique_idxs, counts = torch.unique(best_indices_global, return_counts=True)
        field.counts[label_idx].index_add_(0, unique_idxs, counts.to(field.counts.dtype))
        field.last_used[label_idx][unique_idxs] = step

    return target_centers, best_align, divergence, tens, torch.ones(B, dtype=torch.bool, device=device)


def maybe_spawn_vectorized(
    field: BasinField,
    h: torch.Tensor, # (B, dim)
    label_idx: int,
    tens: torch.Tensor, # (B,)
    align: torch.Tensor, # (B,)
    step: int,
    tension_threshold: float,
    align_threshold: float
):
    """
    Checks criteria and spawns new basins if needed.
    """
    # Criteria: tension > thresh AND align < thresh
    spawn_mask = (tens > tension_threshold) & (align < align_threshold)
    
    if not spawn_mask.any():
        return

    # Candidates for spawning
    candidate_indices = torch.nonzero(spawn_mask, as_tuple=True)[0]
    
    for idx in candidate_indices:
        vec = h[idx]
        # add_basin handles detach now
        field.add_basin(label_idx, vec, step)


def prune_and_merge_vectorized(
     field: BasinField,
     min_count: int = 10,
     merge_cos_threshold: float = 0.97
):
     for l_idx in range(3):
         mask = field.active[l_idx]
         if not mask.any():
             continue
             
         # 1. Prune
         to_prune = (field.counts[l_idx] < min_count) & mask
         if to_prune.any():
             field.active[l_idx][to_prune] = False
             field.counts[l_idx][to_prune] = 0
             field.last_used[l_idx][to_prune] = 0
             field.centers[l_idx][to_prune] = 0.0

         # 2. Merge
         active_idxs = torch.nonzero(field.active[l_idx], as_tuple=True)[0]
         if len(active_idxs) < 2:
             continue
             
         active_centers = field.centers[l_idx][active_idxs]
         c_norm = F.normalize(active_centers, dim=1)
         sims = torch.mm(c_norm, c_norm.t())
         
         merged_mask = torch.zeros(len(active_idxs), dtype=torch.bool, device=field.centers.device)
         
         for i in range(len(active_idxs)):
             if merged_mask[i]: continue
             for j in range(i + 1, len(active_idxs)):
                 if merged_mask[j]: continue
                 
                 if sims[i, j] > merge_cos_threshold:
                     idx_i = active_idxs[i]
                     idx_j = active_idxs[j]
                     
                     count_i = field.counts[l_idx, idx_i]
                     count_j = field.counts[l_idx, idx_j]
                     total = count_i + count_j
                     
                     w_i = count_i.float() / total.float()
                     w_j = count_j.float() / total.float()
                     
                     new_center = w_i * field.centers[l_idx, idx_i] + w_j * field.centers[l_idx, idx_j]
                     # Detach explicitly to be safe, though operations on leaf/buffer usually fine
                     field.centers[l_idx, idx_i] = F.normalize(new_center, dim=0).detach()
                     field.counts[l_idx, idx_i] = total
                     
                     field.active[l_idx, idx_j] = False
                     field.counts[l_idx, idx_j] = 0
                     merged_mask[j] = True