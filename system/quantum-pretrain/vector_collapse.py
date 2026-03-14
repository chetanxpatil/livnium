"""
Vector collapse dynamics for quantum_embed (Fast Vectorized)

Mirrors nova_v3 semantics with a lightweight implementation:
- legacy static collapse with three learned anchors
- collapse_dynamic that routes to per-label basins (E/C/N) and optionally
  spawns/prunes anchors during training.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from basin_field import (
    BasinField,
    route_to_basin_vectorized,
    maybe_spawn_vectorized,
    prune_and_merge_vectorized,
    LABEL_TO_IDX,
)


def divergence_from_alignment(align_value: torch.Tensor) -> torch.Tensor:
    return 0.38 - align_value


def tension(divergence: torch.Tensor) -> torch.Tensor:
    return divergence.abs()


class VectorCollapseEngine(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 4,
        strength_entail: float = 0.1,
        strength_contra: float = 0.1,
        strength_neutral: float = 0.05,
        basin_tension_threshold: float = 0.15,
        basin_align_threshold: float = 0.6,
        basin_anchor_lr: float = 0.05,
        basin_prune_min_count: int = 10,
        basin_prune_merge_cos: float = 0.97,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.strength_entail = strength_entail
        self.strength_contra = strength_contra
        self.strength_neutral = strength_neutral
        self.basin_tension_threshold = basin_tension_threshold
        self.basin_align_threshold = basin_align_threshold
        self.basin_anchor_lr = basin_anchor_lr
        self.basin_prune_min_count = basin_prune_min_count
        self.basin_prune_merge_cos = basin_prune_merge_cos

        self.update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.anchor_entail = nn.Parameter(torch.randn(dim))
        self.anchor_contra = nn.Parameter(torch.randn(dim))
        self.anchor_neutral = nn.Parameter(torch.randn(dim))

    def collapse(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """Legacy static collapse (three learned anchors)."""
        return self._collapse_static(h0)

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
        
        # h0: (B, dim)
        # labels: (B,)
        
        h = h0.clone()
        device = h.device
        
        # Ensure basin field is on the right device
        if basin_field.centers.device != device:
             basin_field.to(device)

        # Prepare strength tensor for batch (B,)
        # 0=E, 1=C, 2=N (Wait, original vector_collapse check: "0: E, 1: C, 2: N" in `label_to_char`)
        # Wait, original code:
        # label_to_char = {0: "E", 1: "C", 2: "N"}
        # label_strength = {0: ...entail, 1: ...contra, 2: ...neutral}
        
        strengths = torch.zeros_like(labels, dtype=torch.float32)
        strengths[labels == 0] = self.strength_entail
        strengths[labels == 1] = self.strength_contra
        strengths[labels == 2] = self.strength_neutral
        
        # We need to find the target anchor for each item in the batch
        # Since routing depends on the label, we do it per-label-group
        
        target_centers = torch.zeros_like(h)
        # alignment/divergence stats (optional, for tracing)
        # We'll just trace properly.
        
        # 1. Routing phase (determine WHERE each point wants to go)
        # We do this once at the start? Or every layer?
        # Original code routing was OUTSIDE the layer loop:
        # "anchor, ..., = route_to_basin(...); for _ in range(num_layers): ... h = h + ... * anchor_vec"
        # So yes, fixed anchor per collapse.
        
        batch_align = torch.zeros(h.size(0), device=device)
        batch_tens = torch.zeros(h.size(0), device=device)
        
        for l_idx in [0, 1, 2]:
            mask = (labels == l_idx)
            if not mask.any():
                continue
                
            sub_h = h[mask]
            
            # Vectorized route
            centers_sub, align_sub, div_sub, tens_sub, found_mask = route_to_basin_vectorized(
                basin_field, sub_h, l_idx, global_step, training=update_anchors
            )
            
            # If no basins found (found_mask=False), we need to seed the first one
            # The vectorized route returns zeros for centers.
            # We should forcibly spawn one if empty.
            if (~found_mask).any():
                # Take the first one that failed and add it
                # Realistically, `add_basin` and then re-route
                # For simplicity, we just add the first one of the batch as a seed
                bad_indices = torch.nonzero(~found_mask, as_tuple=True)[0]
                # It's sub_h relative index
                # Just add first one
                first_bad = bad_indices[0]
                basin_field.add_basin(l_idx, sub_h[first_bad].detach(), global_step)
                
                # Re-route this group now that we have at least one
                centers_sub, align_sub, div_sub, tens_sub, found_mask = route_to_basin_vectorized(
                     basin_field, sub_h, l_idx, global_step, training=update_anchors
                )
            
            target_centers[mask] = centers_sub
            batch_align[mask] = align_sub
            batch_tens[mask] = tens_sub
            
            # Spawning logic
            if spawn_new:
                 maybe_spawn_vectorized(
                     basin_field, sub_h, l_idx, tens_sub, align_sub, 
                     global_step, self.basin_tension_threshold, self.basin_align_threshold
                 )

        # 2. Dynamics loop
        # Now we have `target_centers` (B, dim) for the whole batch
        
        trace = {
             "align": [], "div": [], "tens": []
        }
        
        strengths_unsqueezed = strengths.unsqueeze(-1) # (B, 1)

        for _ in range(self.num_layers):
            # Recalculate alignment/div based on current h vs FIXED target
            h_n = F.normalize(h, dim=-1)
            # Alignment with target
            # target_centers is (B, dim), h_n is (B, dim)
            # Dot product per row
            align = (h_n * target_centers).sum(dim=1) # (B,)
            div = divergence_from_alignment(align)
            tens = tension(div)
            
            # Trace
            # (We skip detailed tracing per label group for speed, just global stats)
            # trace["align"].append(align.detach()) 
            
            delta = self.update(h)
            
            # Vector pointing from target to h (repulsion direction?)
            # Logic: h = h + delta - strength * div * (h - center)
            # (h - center) is vector AWAY from center.
            # So we move opposite to it? i.e. TOWARDS center?
            # "- (h-center)" = "center - h"
            # Yes, standard attraction if strength > 0 and div > 0
            
            anchor_vec = F.normalize(h - target_centers, dim=-1) # Direction FROM anchor TO h
            
            # Update
            force = strengths_unsqueezed * div.unsqueeze(-1) * anchor_vec
            h = h + delta - force
            
            # Clamp norm
            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            mask_norm = (h_norm > 10.0)
            if mask_norm.any():
                h = torch.where(mask_norm, h * (10.0 / (h_norm + 1e-8)), h)

        # 3. Update anchors (Moving average of where the points ended up?)
        # Legacy code: `update_basin_center(anchor, h[i])` AFTER the loop, using FINAL h.
        if update_anchors:
             # We need to update the centers in `basin_field` using `h`.
             # Since it's batched, we do:
             # For each active basin, find all h's that targeted it, average them, update.
             # This is slightly expensive loop over basins? 
             # No, 3 * 64 = 192 basins max. Creating a mask for each is fast.
             
             # Optimization: only compute for labels present in batch
             present_labels = torch.unique(labels)
             
             for l_idx in present_labels:
                 l_idx = int(l_idx.item())
                 # Indices in batch
                 mask_l = (labels == l_idx)
                 if not mask_l.any(): continue

                 h_sub = h[mask_l]
                 centers_sub = target_centers[mask_l]
                 
                 # Vectorized update approach:
                 # 1. Get centers (K, dim)
                 all_centers_l = basin_field.centers[l_idx]
                 # 2. Find closest center for each h_sub (re-match, cheap)
                 # normalized h_final (detach to avoid tracking in anchor buffers)
                 h_final_n = F.normalize(h_sub.detach(), dim=-1)
                 sims = torch.matmul(h_final_n, all_centers_l.t())
                 best_idxs = torch.argmax(sims, dim=1) # (B_sub,) indices into K
                 
                 active_ids = torch.unique(best_idxs)
                 lr = self.basin_anchor_lr
                 
                 for k_idx in active_ids:
                     mask_k = (best_idxs == k_idx)
                     mean_vec = h_final_n[mask_k].mean(dim=0)
                     old_c = all_centers_l[k_idx]
                     new_c = (1 - lr) * old_c + lr * mean_vec
                     all_centers_l[k_idx] = F.normalize(new_c, dim=0)

        # 4. Pruning
        if prune_every > 0 and global_step > 0 and global_step % prune_every == 0:
             prune_and_merge_vectorized(
                 basin_field, self.basin_prune_min_count, self.basin_prune_merge_cos
             )

        return h, trace

    def _collapse_static(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """Same as before but ensures batch dimension handling is efficient."""
        # This part was already mostly vectorized (except for the 3 explicit branches being handled together).
        # We can optimize it by treating 3 anchors as a "batch of 3" but current code is fine for static mode.
        # Just need to make sure we don't break anything.
        
        h = h0.clone()
        if h.dim() == 1: h = h.unsqueeze(0)
        
        # ... logic ...
        # For brevity, I'll copy the logic but trust it's fast enough as it uses tensor broadcasting.
        # The original code's _collapse_static was actually fine, it had no Python loops over batch.
        
        trace = {}
        e_dir = F.normalize(self.anchor_entail, dim=0)
        c_dir = F.normalize(self.anchor_contra, dim=0)
        n_dir = F.normalize(self.anchor_neutral, dim=0)

        for _ in range(self.num_layers):
            h_n = F.normalize(h, dim=-1)
            # ... broadcasted ops ...
            a_e = (h_n * e_dir).sum(dim=-1)
            a_c = (h_n * c_dir).sum(dim=-1)
            a_n = (h_n * n_dir).sum(dim=-1)
            
            d_e = divergence_from_alignment(a_e)
            d_c = divergence_from_alignment(a_c)
            d_n = divergence_from_alignment(a_n)
            
            delta = self.update(h)
            
            e_vec = F.normalize(h - e_dir, dim=-1)
            c_vec = F.normalize(h - c_dir, dim=-1)
            n_vec = F.normalize(h - n_dir, dim=-1)
            
            h = (
                h + delta
                - self.strength_entail * d_e.unsqueeze(-1) * e_vec
                - self.strength_contra * d_c.unsqueeze(-1) * c_vec
                - self.strength_neutral * d_n.unsqueeze(-1) * n_vec
            )
            
            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)

        return h, trace

    def forward(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        return self._collapse_static(h0)
