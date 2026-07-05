"""Configurable vector collapse engine for Livnium."""

from .basin_field import (
    BasinField,
    maybe_spawn_vectorized,
    prune_and_merge_vectorized,
    route_to_basin_vectorized,
)
from .config import BasinConfig, CollapseConfig
from .engine import VectorCollapseEngine, divergence_from_alignment, tension

__all__ = [
    "BasinConfig",
    "BasinField",
    "CollapseConfig",
    "VectorCollapseEngine",
    "divergence_from_alignment",
    "maybe_spawn_vectorized",
    "prune_and_merge_vectorized",
    "route_to_basin_vectorized",
    "tension",
]
