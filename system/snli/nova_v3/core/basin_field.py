"""
Dynamic Basin Field

Maintains per-label micro-basins that can be routed to, updated, spawned,
and pruned during training. Provides a minimal state_dict interface for
checkpointing.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


LABELS = ("E", "N", "C")


class BasinAnchor:
    """
    Single micro-basin anchor.
    """

    def __init__(self, center: torch.Tensor, label: str, step: int = 0):
        if label not in LABELS:
            raise ValueError(f"Invalid label {label}; expected one of {LABELS}")
        # Store as detached, unit vector on creation
        self.center = F.normalize(center.detach(), dim=0)
        self.label = label
        self.count: int = 0
        self.last_used_step: int = step


class BasinField:
    """
    Collection of anchors per label with simple device/state helpers.
    """

    def __init__(self, max_basins_per_label: int = 64, device: torch.device = None):
        self.anchors: Dict[str, List[BasinAnchor]] = {l: [] for l in LABELS}
        self.max_basins_per_label = max_basins_per_label
        self.device = device

    def to(self, device: torch.device):
        """
        Move all anchor centers to the target device.
        """
        self.device = device
        for anchors in self.anchors.values():
            for a in anchors:
                a.center = a.center.to(device)
        return self

    def state_dict(self) -> Dict:
        """
        Lightweight serialization for checkpoints.
        """
        return {
            "max_basins_per_label": self.max_basins_per_label,
            "anchors": {
                label: [
                    {
                        "center": anchor.center,
                        "label": anchor.label,
                        "count": anchor.count,
                        "last_used_step": anchor.last_used_step,
                    }
                    for anchor in anchors
                ]
                for label, anchors in self.anchors.items()
            },
        }

    def load_state_dict(self, state: Dict):
        """
        Restore anchors from serialized form.
        """
        self.max_basins_per_label = state.get("max_basins_per_label", self.max_basins_per_label)
        anchors_state = state.get("anchors", {})
        self.anchors = {l: [] for l in LABELS}
        for label, anchors in anchors_state.items():
            if label not in LABELS:
                continue
            for anchor_state in anchors:
                center = anchor_state["center"]
                if self.device is not None:
                    center = center.to(self.device)
                anchor = BasinAnchor(center=center, label=anchor_state["label"])
                anchor.count = anchor_state.get("count", 0)
                anchor.last_used_step = anchor_state.get("last_used_step", 0)
                self.anchors[label].append(anchor)


def route_to_basin(
    field: BasinField, h: torch.Tensor, y: str, step: int
) -> Tuple[BasinAnchor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find (or seed) the best basin for label y given state h.

    Returns anchor, best_align, divergence, tension.
    """
    if y not in LABELS:
        raise ValueError(f"Invalid label {y}; expected one of {LABELS}")
    anchors = field.anchors[y]
    h_n = F.normalize(h, dim=0)

    if not anchors:
        anchor = BasinAnchor(h_n, y, step=step)
        field.anchors[y].append(anchor)
        best_align = torch.tensor(1.0, device=h.device, dtype=h.dtype)
        divergence = torch.tensor(0.0, device=h.device, dtype=h.dtype)
        tens = torch.tensor(0.0, device=h.device, dtype=h.dtype)
        return anchor, best_align, divergence, tens

    best_anchor = None
    best_align = None
    for a in anchors:
        align = torch.dot(h_n, a.center)
        if best_align is None or align > best_align:
            best_align = align
            best_anchor = a

    from .physics_laws import BARRIER
    divergence = BARRIER - best_align
    tens = divergence.abs()
    best_anchor.count += 1
    best_anchor.last_used_step = step
    return best_anchor, best_align, divergence, tens


def update_basin_center(anchor: BasinAnchor, h: torch.Tensor, lr: float = 0.05):
    """
    Exponential moving average update on the unit sphere.
    """
    h_n = F.normalize(h, dim=0)
    new_center = (1 - lr) * anchor.center + lr * h_n.detach()
    anchor.center = F.normalize(new_center, dim=0)


def maybe_spawn_basin(
    field: BasinField,
    h: torch.Tensor,
    y: str,
    tension_value: torch.Tensor,
    align_value: torch.Tensor,
    step: int,
    tension_threshold: float = 0.15,
    align_threshold: float = 0.6,
):
    """
    Spawn a new basin when tension stays high and alignment is low.
    """
    anchors = field.anchors[y]
    if len(anchors) >= field.max_basins_per_label:
        return

    if tension_value.item() > tension_threshold and align_value.item() < align_threshold:
        h_n = F.normalize(h, dim=0)
        new_anchor = BasinAnchor(h_n, y, step=step)
        field.anchors[y].append(new_anchor)


def spawn_neutral_at_boundary(
    field: BasinField,
    h: torch.Tensor,
    ec_boundary_value: float,
    step: int,
    boundary_threshold: float = 0.7,
):
    """
    Spawn a neutral basin whenever a state lands near the E-C boundary.

    This is called during training regardless of the sample's true label.
    The idea: any state that is equidistant between the E and C anchors is
    geometrically in the neutral zone. Seeding neutral basins there gives the
    neutral attractor real density in that region of space, rather than leaving
    it as an empty residual between two strong attractors.

    Args:
        field: The shared BasinField instance
        h: Current state vector [dim]
        ec_boundary_value: boundary_proximity(align_E, align_C) for this sample
        step: Current training step
        boundary_threshold: Only spawn when ec_boundary_value exceeds this
    """
    if ec_boundary_value < boundary_threshold:
        return
    if len(field.anchors["N"]) >= field.max_basins_per_label:
        return
    h_n = F.normalize(h.detach(), dim=0)
    new_anchor = BasinAnchor(h_n, "N", step=step)
    field.anchors["N"].append(new_anchor)


def prune_and_merge(
    field: BasinField,
    min_count: int = 10,
    merge_cos_threshold: float = 0.97,
):
    """
    Prune weak anchors and merge very similar ones.
    """
    for y, anchors in field.anchors.items():
        # Prune
        anchors = [a for a in anchors if a.count >= min_count]

        # Merge close anchors
        kept: List[BasinAnchor] = []
        while anchors:
            a = anchors.pop()
            merged = False
            for b in kept:
                if torch.dot(a.center, b.center) > merge_cos_threshold:
                    total = a.count + b.count
                    merged_center = (a.center * a.count + b.center * b.count) / max(total, 1)
                    b.center = F.normalize(merged_center, dim=0)
                    b.count = total
                    merged = True
                    break
            if not merged:
                kept.append(a)
        field.anchors[y] = kept
