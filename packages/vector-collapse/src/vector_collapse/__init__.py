"""Configurable vector collapse engine for Livnium."""

from .basin_field import (
    BasinField,
    maybe_spawn_vectorized,
    prune_and_merge_vectorized,
    route_to_basin_vectorized,
)
from .config import BasinConfig, CollapseConfig
from .engine import VectorCollapseEngine, divergence_from_alignment, tension
from .ledger import BasinEvent, DynamicsLedger, StepRecord

__all__ = [
    "BasinConfig",
    "BasinEvent",
    "BasinField",
    "CollapseConfig",
    "DynamicsLedger",
    "StepRecord",
    "VectorCollapseEngine",
    "divergence_from_alignment",
    "maybe_spawn_vectorized",
    "prune_and_merge_vectorized",
    "route_to_basin_vectorized",
    "tension",
]
