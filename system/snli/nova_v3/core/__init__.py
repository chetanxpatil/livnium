"""
Livnium Core v1.0 - Vector-Based Physics Engine

Layer 0: Pure physics. No tokens, no labels, no tasks.
"""

from .vector_state import VectorState
from .physics_laws import alignment, divergence_from_alignment, tension
from .vector_collapse_engine import VectorCollapseEngine
from .basin_field import (
    BasinField,
    BasinAnchor,
    route_to_basin,
    update_basin_center,
    maybe_spawn_basin,
    prune_and_merge,
)

__all__ = [
    'VectorState',
    'alignment',
    'divergence_from_alignment',
    'tension',
    'VectorCollapseEngine',
    'BasinField',
    'BasinAnchor',
    'route_to_basin',
    'update_basin_center',
    'maybe_spawn_basin',
    'prune_and_merge',
]
