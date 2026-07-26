"""A two-level Livnium data pile owned by the neural model.

There are 27 macro cells and 27 micro cells inside each macro cell, giving 729
addressable leaves. Payload values are exact unique int64 tokens. Training and
evaluation piles use disjoint value ranges, so held-out values are genuinely
unseen during training.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
from torch import nn


class HierarchicalLivniumPile(nn.Module):
    """Persistent 27x27 payload memory saved inside ``state_dict``."""

    cells_per_level = 27
    levels = 2
    leaf_count = cells_per_level**levels

    def __init__(self, payload_values: torch.Tensor | Sequence[int] | None = None):
        super().__init__()
        if payload_values is None:
            payload_values = torch.arange(self.leaf_count, dtype=torch.long)
        payload_tensor = torch.as_tensor(payload_values, dtype=torch.long).clone()
        self._validate_payloads(payload_tensor)
        ranks, sorted_values = self._rank_payloads(payload_tensor)
        self.register_buffer("payload_values", payload_tensor)
        self.register_buffer("payload_ranks", ranks)
        self.register_buffer("rank_to_value", sorted_values)

    @classmethod
    def random(cls, seed: int) -> "HierarchicalLivniumPile":
        generator = torch.Generator().manual_seed(seed)
        offset = int(seed) * 1_000_000
        values = offset + torch.randperm(cls.leaf_count, generator=generator)
        return cls(values)

    @classmethod
    def _validate_payloads(cls, payload_values: torch.Tensor) -> None:
        if payload_values.shape != (cls.leaf_count,):
            raise ValueError(f"payload_values must have shape ({cls.leaf_count},)")
        if payload_values.unique().numel() != cls.leaf_count:
            raise ValueError("payload_values must be unique")

    @staticmethod
    def _rank_payloads(payload_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sorted_values, order = payload_values.sort()
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(order.numel(), device=order.device)
        return ranks, sorted_values

    def load_payloads(self, payload_values: torch.Tensor | Sequence[int]) -> None:
        """Replace the internal pile without changing the trained controller."""
        payload_tensor = torch.as_tensor(
            payload_values,
            dtype=torch.long,
            device=self.payload_values.device,
        )
        self._validate_payloads(payload_tensor)
        ranks, sorted_values = self._rank_payloads(payload_tensor)
        self.payload_values.copy_(payload_tensor)
        self.payload_ranks.copy_(ranks)
        self.rank_to_value.copy_(sorted_values)

    def leaf_index(self, macro_index: torch.Tensor, micro_index: torch.Tensor) -> torch.Tensor:
        return macro_index * self.cells_per_level + micro_index

    def payload_at(self, macro_index: torch.Tensor, micro_index: torch.Tensor) -> torch.Tensor:
        return self.payload_values[self.leaf_index(macro_index, micro_index)]

    def payload_rank_at(
        self,
        macro_index: torch.Tensor,
        micro_index: torch.Tensor,
    ) -> torch.Tensor:
        return self.payload_ranks[self.leaf_index(macro_index, micro_index)]

    def values_for_ranks(self, ranks: torch.Tensor) -> torch.Tensor:
        return self.rank_to_value[ranks]

    def read(
        self,
        macro_distribution: torch.Tensor,
        micro_distribution: torch.Tensor,
    ) -> torch.Tensor:
        """Read payload probabilities from soft or hard observer distributions."""
        if macro_distribution.shape != micro_distribution.shape:
            raise ValueError("macro and micro distributions must have the same shape")
        if macro_distribution.ndim != 2 or macro_distribution.shape[1] != self.cells_per_level:
            raise ValueError("observer distributions must have shape [batch, 27]")

        address_distribution = (
            macro_distribution.unsqueeze(2) * micro_distribution.unsqueeze(1)
        ).reshape(macro_distribution.shape[0], self.leaf_count)

        payload_distribution = torch.zeros_like(address_distribution)
        payload_indices = self.payload_ranks.unsqueeze(0).expand_as(address_distribution)
        return payload_distribution.scatter_add(1, payload_indices, address_distribution)

    def inventory_hash(self) -> str:
        raw = self.payload_values.detach().cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()
