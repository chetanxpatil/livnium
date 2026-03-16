"""
Vector Collapse Engine: Multi-Basin Collapse Dynamics

Evolves a state vector h through L steps with multiple anchors (E/C/N) to
encourage three basins. Each anchor uses the Livnium divergence law.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .physics_laws import divergence_from_alignment, tension, boundary_proximity
from .basin_field import (
    BasinField,
    route_to_basin,
    update_basin_center,
    maybe_spawn_basin,
    spawn_neutral_at_boundary,
    prune_and_merge,
)


class VectorCollapseEngine(nn.Module):
    """
    Core collapse engine for Livnium.
    
    Takes an initial state h0 and evolves it through L collapse steps with
    multiple anchors (entailment/contradiction/neutral).
    At each step, it:
    1. Computes alignment/divergence/tension to each anchor
    2. Applies state update + anchor forces
    3. Logs trace
    
    The trace is what watchdogs inspect.
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 6,
        strength_entail: float = 0.1,
        strength_contra: float = 0.1,
        strength_neutral: float = 0.05,
        strength_neutral_boost: float = 0.05,
        # Dynamic basin defaults
        basin_tension_threshold: float = 0.15,
        basin_align_threshold: float = 0.6,
        basin_anchor_lr: float = 0.05,
        basin_prune_min_count: int = 10,
        basin_prune_merge_cos: float = 0.97,
        # Rotational dynamics (v6)
        rot_rank: int = 0,
        rot_strength: float = 0.01,
        # Locking zones (v7)
        lock_threshold: float = 0.0,
        lock_gain: float = 2.0,
        lock_temp: float = 0.1,
        # Virtual null endpoint (v8)
        strength_null: float = 0.0,
        # Adaptive metric (v9)
        adaptive_metric: bool = False,
    ):
        """
        Initialize collapse engine.
        
        Args:
            dim: Dimension of state vector
            num_layers: Number of collapse steps
            strength_entail: Force strength for entail anchor
            strength_contra: Force strength for contradiction anchor
            strength_neutral: Force strength for neutral anchor
            strength_neutral_boost: Extra neutral pull applied when state is near the E-C boundary.
                When the state is equidistant between E and C (boundary_proximity ≈ 1), this
                additional force kicks in to pull the state toward the neutral anchor, preventing
                it from randomly falling into E or C instead of resolving to neutral.
            basin_*: Defaults for dynamic basin behavior (spawn/update/prune)
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.strength_entail = strength_entail
        self.strength_contra = strength_contra
        self.strength_neutral = strength_neutral
        self.strength_neutral_boost = strength_neutral_boost
        self.basin_tension_threshold = basin_tension_threshold
        self.basin_align_threshold = basin_align_threshold
        self.basin_anchor_lr = basin_anchor_lr
        self.basin_prune_min_count = basin_prune_min_count
        self.basin_prune_merge_cos = basin_prune_merge_cos
        
        # State update network
        # This learns how to evolve the state based on current configuration
        self.update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

        # Three anchors to create multi-basin geometry
        self.anchor_entail = nn.Parameter(torch.randn(dim))
        self.anchor_contra = nn.Parameter(torch.randn(dim))
        self.anchor_neutral = nn.Parameter(torch.randn(dim))

        # Virtual null endpoint (v8): a 4th attractor with no label.
        # Gives genuinely ambiguous vectors somewhere to go that is NOT N.
        # Initialized orthogonal to the semantic anchors so it carves its own
        # region of the space — "50% more space, but empty."
        # No training loss attached — pure geometry shaper.
        # strength_null=0.0 disables it entirely.
        self.strength_null = strength_null
        if strength_null > 0.0:
            null_init = torch.randn(dim)
            self.anchor_null = nn.Parameter(null_init)

        # Adaptive metric (v9): learned diagonal metric tensor M = diag(s₁…s_dim).
        # Alignment becomes cos_M(h, anchor) = cos(h⊙s, anchor⊙s) where s = softplus(log_scale).
        # The model stretches dimensions that help separate classes and suppresses noise.
        # Space warps where resolution is needed — living geometry.
        # 256 learnable parameters. adaptive_metric=False disables entirely.
        self.adaptive_metric = adaptive_metric
        if adaptive_metric:
            # log-scale init=0 → softplus(0)≈0.69, all dims start near-uniform
            self.metric_log_scale = nn.Parameter(torch.zeros(dim))
        else:
            self.metric_log_scale = None

        # Rotational dynamics: low-rank skew-symmetric matrix
        # S = W_rot_U @ W_rot_V^T - W_rot_V @ W_rot_U^T  (antisymmetric, rank-2k)
        # At each step: h += rot_strength * (h @ S)
        # This adds a circulating component to trajectories — vectors spiral
        # toward attractors instead of falling straight in, preventing basins
        # from merging into blobs. Physics analog: angular momentum.
        self.rot_rank = rot_rank
        self.rot_strength = rot_strength
        if rot_rank > 0:
            self.W_rot_U = nn.Parameter(torch.randn(dim, rot_rank) * 0.01)
            self.W_rot_V = nn.Parameter(torch.randn(dim, rot_rank) * 0.01)
        else:
            self.W_rot_U = None
            self.W_rot_V = None

        # Locking zones: once alignment with a basin crosses lock_threshold,
        # the attractive force is multiplied by (1 + lock_gain * sigmoid gate).
        # This creates a capture region — inside it, the vector is held firmly;
        # outside it, normal fluid dynamics apply.
        # lock_threshold=0.0 disables locking (gate stays near 0.5 always).
        self.lock_threshold = lock_threshold
        self.lock_gain = lock_gain
        self.lock_temp = lock_temp
    
    def _metric_normalize(self, h: torch.Tensor) -> torch.Tensor:
        """
        Normalize h in the learned metric space.
        If adaptive_metric is enabled: warp h by per-dimension scales before
        normalizing, so alignment is measured in a space the model has learned
        to stretch. The result is used for all cos-similarity computations.
        The metric scale is softplus(log_scale) > 0 always.
        """
        if self.metric_log_scale is None:
            return F.normalize(h, dim=-1)
        scale = F.softplus(self.metric_log_scale)   # (dim,) positive
        return F.normalize(h * scale.unsqueeze(0), dim=-1)

    def _apply_rotation(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply low-rank skew-symmetric rotation to h.
        S = W_rot_U @ W_rot_V^T - W_rot_V @ W_rot_U^T  (antisymmetric)
        h += rot_strength * (h @ S)
        Adds a circulating component orthogonal to the gradient forces.
        """
        if self.W_rot_U is None:
            return h
        u = h @ self.W_rot_U          # (B, rank)
        v = h @ self.W_rot_V          # (B, rank)
        rot_h = u @ self.W_rot_V.T - v @ self.W_rot_U.T  # (B, dim)
        return h + self.rot_strength * rot_h

    def collapse(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Backward-compatible static collapse (three fixed anchors).
        """
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
        """
        Collapse with dynamic per-label basins.
        
        Args:
            h0: Initial state vector(s) [B, dim] or [dim]
            labels: Ground-truth labels as integers (0=E,1=C,2=N)
            basin_field: Shared BasinField instance
            global_step: Training step for bookkeeping
            spawn_new: Whether to allow spawning new basins
            prune_every: If >0, prune/merge every N steps
            update_anchors: Whether to adapt basin centers after collapse
        """
        squeeze = False
        h = h0
        if h.dim() == 1:
            h = h.unsqueeze(0)
            labels = labels.unsqueeze(0)
            squeeze = True
        h = h.clone()
        labels = labels.to(h.device)
        basin_field.to(h.device)

        label_to_char = {0: "E", 1: "C", 2: "N"}
        label_strength = {
            0: self.strength_entail,
            1: self.strength_contra,
            2: self.strength_neutral,
        }

        anchors = []

        # Route each sample to its label-specific basin (and possibly spawn)
        for i in range(h.size(0)):
            y_char = label_to_char.get(int(labels[i].item()))
            anchor, align_val, div_val, tens_val = route_to_basin(
                basin_field, h[i], y_char, step=global_step
            )
            anchors.append(anchor)
            if spawn_new:
                maybe_spawn_basin(
                    basin_field,
                    h[i],
                    y_char,
                    tens_val,
                    align_val,
                    step=global_step,
                    tension_threshold=self.basin_tension_threshold,
                    align_threshold=self.basin_align_threshold,
                )

        anchor_dirs = torch.stack([a.center for a in anchors]).to(h.device)
        strengths = torch.tensor([label_strength[int(l.item())] for l in labels], device=h.device)

        trace = {
            "alignment_local": [],
            "divergence_local": [],
            "tension_local": [],
        }

        for step in range(self.num_layers):
            h_n = self._metric_normalize(h)
            align = (h_n * anchor_dirs).sum(dim=-1)
            div = divergence_from_alignment(align)
            tens = tension(div)

            trace["alignment_local"].append(align.detach())
            trace["divergence_local"].append(div.detach())
            trace["tension_local"].append(tens.detach())

            delta = self.update(h)
            anchor_vec = F.normalize(h - anchor_dirs, dim=-1)
            # Neutral boundary boost using the per-sample anchor alignments
            # Requires routing all E and C anchors for boundary_proximity — approximate
            # with the static anchors for efficiency in the dynamic path.
            e_dir_d = F.normalize(self.anchor_entail, dim=0)
            c_dir_d = F.normalize(self.anchor_contra, dim=0)
            n_dir_d = F.normalize(self.anchor_neutral, dim=0)
            h_n_d = F.normalize(h, dim=-1)
            a_e_d = (h_n_d * e_dir_d).sum(dim=-1)
            a_c_d = (h_n_d * c_dir_d).sum(dim=-1)
            ec_boundary_d = boundary_proximity(a_e_d, a_c_d)
            n_vec_d = F.normalize(h - n_dir_d.unsqueeze(0), dim=-1)
            h = (
                h
                + delta
                - strengths.unsqueeze(-1) * div.unsqueeze(-1) * anchor_vec
                - self.strength_neutral_boost * ec_boundary_d.unsqueeze(-1) * n_vec_d
            )

            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)

        # Update anchors post-collapse
        if update_anchors:
            for i, anchor in enumerate(anchors):
                update_basin_center(anchor, h[i], lr=self.basin_anchor_lr)
                anchor.last_used_step = global_step

        # Seed neutral basins at E-C boundary positions discovered during this batch.
        # Any sample whose final state sits near the E-C boundary is geometrically in
        # neutral territory — plant a neutral anchor there to strengthen that region.
        if spawn_new:
            e_dir_post = F.normalize(self.anchor_entail, dim=0)
            c_dir_post = F.normalize(self.anchor_contra, dim=0)
            h_n_post = F.normalize(h, dim=-1)
            a_e_post = (h_n_post * e_dir_post).sum(dim=-1)
            a_c_post = (h_n_post * c_dir_post).sum(dim=-1)
            ec_boundary_post = boundary_proximity(a_e_post, a_c_post)
            for i in range(h.size(0)):
                spawn_neutral_at_boundary(
                    basin_field,
                    h[i],
                    ec_boundary_value=ec_boundary_post[i].item(),
                    step=global_step,
                    boundary_threshold=0.7,
                )

        if prune_every and global_step > 0 and global_step % prune_every == 0:
            prune_and_merge(
                basin_field,
                min_count=self.basin_prune_min_count,
                merge_cos_threshold=self.basin_prune_merge_cos,
            )

        if squeeze:
            h = h.squeeze(0)
        return h, trace

    def _collapse_static(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Collapse initial state h0 through L steps.
        
        Args:
            h0: Initial state vector (dim,) or batch of vectors [B, dim]
            
        Returns:
            Tuple of (h_final, trace)
            trace: Dict with per-anchor align/div/tension lists
        """
        squeeze = False
        h = h0
        if h.dim() == 1:
            h = h.unsqueeze(0)
            squeeze = True
        h = h.clone()
        trace = {
            "alignment_entail": [],
            "alignment_contra": [],
            "alignment_neutral": [],
            "divergence_entail": [],
            "divergence_contra": [],
            "divergence_neutral": [],
            "tension_entail": [],
            "tension_contra": [],
            "tension_neutral": [],
        }
        
        # Normalize anchor directions
        e_dir = F.normalize(self.anchor_entail, dim=0)
        c_dir = F.normalize(self.anchor_contra, dim=0)
        n_dir = F.normalize(self.anchor_neutral, dim=0)
        null_dir = F.normalize(self.anchor_null, dim=0) if self.strength_null > 0.0 else None
        
        for step in range(self.num_layers):
            # Normalize current state along feature dim
            h_n = self._metric_normalize(h)
            
            # Compute physics to each anchor
            a_e = (h_n * e_dir).sum(dim=-1)
            a_c = (h_n * c_dir).sum(dim=-1)
            a_n = (h_n * n_dir).sum(dim=-1)
            d_e = divergence_from_alignment(a_e)
            d_c = divergence_from_alignment(a_c)
            d_n = divergence_from_alignment(a_n)
            t_e = tension(d_e)
            t_c = tension(d_c)
            t_n = tension(d_n)
            
            # Log trace
            trace["alignment_entail"].append(a_e.detach())
            trace["alignment_contra"].append(a_c.detach())
            trace["alignment_neutral"].append(a_n.detach())
            trace["divergence_entail"].append(d_e.detach())
            trace["divergence_contra"].append(d_c.detach())
            trace["divergence_neutral"].append(d_n.detach())
            trace["tension_entail"].append(t_e.detach())
            trace["tension_contra"].append(t_c.detach())
            trace["tension_neutral"].append(t_n.detach())
            
            # State update
            delta = self.update(h)
            # Anchor forces: move toward/away each anchor along their difference vector
            e_vec = F.normalize(h - e_dir.unsqueeze(0), dim=-1)
            c_vec = F.normalize(h - c_dir.unsqueeze(0), dim=-1)
            n_vec = F.normalize(h - n_dir.unsqueeze(0), dim=-1)
            # Neutral boundary boost
            ec_boundary = boundary_proximity(a_e, a_c)

            # Locking zones (v7): once alignment crosses lock_threshold,
            # the basin's attractive force is amplified by (1 + lock_gain * gate).
            # sigmoid gate is ~0 below threshold, ~1 above it.
            # lock_threshold=0.0 → always mildly active; set e.g. 0.5 to engage
            # only when the vector is already clearly heading toward a basin.
            if self.lock_threshold > 0.0:
                gate_e = torch.sigmoid((a_e - self.lock_threshold) / self.lock_temp)
                gate_c = torch.sigmoid((a_c - self.lock_threshold) / self.lock_temp)
                gate_n = torch.sigmoid((a_n - self.lock_threshold) / self.lock_temp)
                # (B,) tensors — need unsqueeze(-1) below
                s_e = self.strength_entail  * (1.0 + self.lock_gain * gate_e)
                s_c = self.strength_contra  * (1.0 + self.lock_gain * gate_c)
                s_n = self.strength_neutral * (1.0 + self.lock_gain * gate_n)
                h = (
                    h
                    + delta
                    - s_e.unsqueeze(-1) * d_e.unsqueeze(-1) * e_vec
                    - s_c.unsqueeze(-1) * d_c.unsqueeze(-1) * c_vec
                    - s_n.unsqueeze(-1) * d_n.unsqueeze(-1) * n_vec
                    - self.strength_neutral_boost * ec_boundary.unsqueeze(-1) * n_vec
                )
            else:
                h = (
                    h
                    + delta
                    - self.strength_entail * d_e.unsqueeze(-1) * e_vec
                    - self.strength_contra  * d_c.unsqueeze(-1) * c_vec
                    - self.strength_neutral * d_n.unsqueeze(-1) * n_vec
                    - self.strength_neutral_boost * ec_boundary.unsqueeze(-1) * n_vec
                )
            
            # Virtual null endpoint: weak pull toward the empty region
            # Ambiguous vectors that escape E/C/N basins drift here instead
            # of being forced into neutral. No label — pure geometry.
            if null_dir is not None:
                a_null = (h_n * null_dir).sum(dim=-1)
                d_null = divergence_from_alignment(a_null)
                null_vec = F.normalize(h - null_dir.unsqueeze(0), dim=-1)
                h = h - self.strength_null * d_null.unsqueeze(-1) * null_vec

            # Rotational dynamics: spiral trajectories instead of straight-line collapse
            h = self._apply_rotation(h)

            # Soft norm control (conservation-ish)
            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)

        if squeeze:
            h = h.squeeze(0)
        return h, trace

    def collapse_inference(
        self,
        h0: torch.Tensor,
        basin_field: BasinField,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Label-free inference collapse using dynamic basins.

        Bidirectional routing: routes each sample using BOTH h0 and -h0.
        h0 = v_h - v_p captures one ordering; -h0 = v_p - v_h captures the
        reverse. Order matters in both directions — a contradiction pair looks
        like entailment if you flip the sentence order, so considering both
        directions gives a stronger routing signal.

        For each sample:
          - forward basin:  anchor most aligned with  h0  → attraction force
          - reverse basin:  anchor most aligned with -h0  → repulsion force
                           (h should be AWAY from what the reverse looks like)

        Args:
            h0: [B, dim] or [dim]
            basin_field: trained BasinField (read-only during inference)

        Returns:
            (h_final, trace)
        """
        squeeze = False
        h = h0
        if h.dim() == 1:
            h = h.unsqueeze(0)
            squeeze = True
        h = h.clone()
        basin_field.to(h.device)

        label_to_strength = {
            "E": self.strength_entail,
            "C": self.strength_contra,
            "N": self.strength_neutral,
        }

        fwd_anchors = []
        fwd_strengths_list = []
        rev_anchors = []
        rev_strengths_list = []

        for i in range(h.size(0)):
            h_n = F.normalize(h[i], dim=0)
            h_n_rev = -h_n  # reverse direction: v_p - v_h

            # Forward: find anchor most aligned with h0
            best_fwd_anchor, best_fwd_label = None, None
            best_fwd_align = None
            # Reverse: find anchor most aligned with -h0
            best_rev_anchor, best_rev_label = None, None
            best_rev_align = None

            for label_char in ["E", "C", "N"]:
                for anc in basin_field.anchors[label_char]:
                    fwd_align = torch.dot(h_n, anc.center)
                    rev_align = torch.dot(h_n_rev, anc.center)

                    if best_fwd_align is None or fwd_align > best_fwd_align:
                        best_fwd_align = fwd_align
                        best_fwd_anchor = anc
                        best_fwd_label = label_char

                    if best_rev_align is None or rev_align > best_rev_align:
                        best_rev_align = rev_align
                        best_rev_anchor = anc
                        best_rev_label = label_char

            # Fallback to static anchors if no dynamic basins exist
            if best_fwd_anchor is None:
                return self._collapse_static(h0)

            fwd_anchors.append(best_fwd_anchor)
            fwd_strengths_list.append(label_to_strength[best_fwd_label])
            rev_anchors.append(best_rev_anchor)
            rev_strengths_list.append(label_to_strength[best_rev_label])

        # Forward routing: attract h toward the best forward-direction basin
        fwd_dirs = torch.stack([a.center for a in fwd_anchors]).to(h.device)
        fwd_strengths = torch.tensor(fwd_strengths_list, device=h.device, dtype=h.dtype)

        # Reverse routing: repel h away from the best reverse-direction basin
        # (what the pair looks like when flipped is what h should NOT be)
        rev_dirs = torch.stack([a.center for a in rev_anchors]).to(h.device)
        rev_strengths = torch.tensor(rev_strengths_list, device=h.device, dtype=h.dtype)

        trace = {
            "alignment_local": [],
            "divergence_local": [],
            "tension_local": [],
        }

        e_dir = F.normalize(self.anchor_entail, dim=0)
        c_dir = F.normalize(self.anchor_contra, dim=0)
        n_dir = F.normalize(self.anchor_neutral, dim=0)

        for step in range(self.num_layers):
            h_n = self._metric_normalize(h)

            # Forward alignment/divergence/tension (attraction)
            fwd_align = (h_n * fwd_dirs).sum(dim=-1)
            fwd_div = divergence_from_alignment(fwd_align)
            fwd_tens = tension(fwd_div)

            # Reverse alignment (repulsion — how aligned is h with the flipped basin)
            rev_align = (h_n * rev_dirs).sum(dim=-1)
            rev_div = divergence_from_alignment(rev_align)

            trace["alignment_local"].append(fwd_align.detach())
            trace["divergence_local"].append(fwd_div.detach())
            trace["tension_local"].append(fwd_tens.detach())

            delta = self.update(h)

            # Attraction: pull h toward forward basin
            fwd_vec = F.normalize(h - fwd_dirs, dim=-1)
            # Repulsion: push h away from reverse basin
            # (positive force along the direction away from rev_dirs)
            rev_vec = F.normalize(h - rev_dirs, dim=-1)

            # Neutral boundary boost (same as training)
            a_e = (h_n * e_dir).sum(dim=-1)
            a_c = (h_n * c_dir).sum(dim=-1)
            ec_boundary = boundary_proximity(a_e, a_c)
            n_vec = F.normalize(h - n_dir.unsqueeze(0), dim=-1)

            # Locking zones: amplify forward attraction once alignment is high
            if self.lock_threshold > 0.0:
                lock_gate = torch.sigmoid((fwd_align - self.lock_threshold) / self.lock_temp)
                eff_fwd = fwd_strengths * (1.0 + self.lock_gain * lock_gate)
            else:
                eff_fwd = fwd_strengths

            h = (
                h
                + delta
                - eff_fwd.unsqueeze(-1) * fwd_div.unsqueeze(-1) * fwd_vec
                + rev_strengths.unsqueeze(-1) * rev_div.unsqueeze(-1) * rev_vec
                - self.strength_neutral_boost * ec_boundary.unsqueeze(-1) * n_vec
            )

            # Virtual null endpoint
            if self.strength_null > 0.0:
                null_d = F.normalize(self.anchor_null, dim=0)
                a_null = (h_n * null_d).sum(dim=-1)
                d_null = divergence_from_alignment(a_null)
                null_vec = F.normalize(h - null_d.unsqueeze(0), dim=-1)
                h = h - self.strength_null * d_null.unsqueeze(-1) * null_vec

            # Rotational dynamics
            h = self._apply_rotation(h)

            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)

        if squeeze:
            h = h.squeeze(0)
        return h, trace

    def forward(self, h0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Forward pass (alias for collapse).

        Args:
            h0: Initial state vector

        Returns:
            Tuple of (h_final, trace)
        """
        return self._collapse_static(h0)
