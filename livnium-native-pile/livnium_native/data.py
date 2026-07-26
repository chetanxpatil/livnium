"""Synthetic tasks that force the model to navigate its internal pile."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import ACTION_NAMES, action_index_maps
from .model import PAD_TOKEN
from .pile import HierarchicalLivniumPile

# Token IDs are deliberately not action IDs.  The model is never supervised on
# this mapping; it must infer the meaning of the tokens through answer loss.
TOKEN_TO_ACTION = torch.tensor((4, 1, 7, 2, 0, 6, 3, 5), dtype=torch.long)
ACTION_TO_TOKEN = torch.empty_like(TOKEN_TO_ACTION)
ACTION_TO_TOKEN[TOKEN_TO_ACTION] = torch.arange(len(ACTION_NAMES))


@dataclass
class NavigationBatch:
    macro_start: torch.Tensor
    macro_tokens: torch.Tensor
    micro_start: torch.Tensor
    micro_tokens: torch.Tensor
    target_macro: torch.Tensor
    target_micro: torch.Tensor
    target_payloads: torch.Tensor
    target_payload_ranks: torch.Tensor

    def to(self, device: torch.device | str) -> "NavigationBatch":
        return NavigationBatch(
            **{
                name: value.to(device)
                for name, value in self.__dict__.items()
            }
        )


def _sample_level(
    batch_size: int,
    min_steps: int,
    max_steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, 27, (batch_size,), generator=generator)
    lengths = torch.randint(
        min_steps,
        max_steps + 1,
        (batch_size,),
        generator=generator,
    )
    actions = torch.randint(
        0,
        len(ACTION_NAMES),
        (batch_size, max_steps),
        generator=generator,
    )
    tokens = ACTION_TO_TOKEN[actions]
    active = torch.arange(max_steps).unsqueeze(0) < lengths.unsqueeze(1)
    tokens = torch.where(active, tokens, torch.full_like(tokens, PAD_TOKEN))

    maps = action_index_maps()
    targets = starts.clone()
    for step in range(max_steps):
        step_actions = actions[:, step]
        proposed = maps[step_actions, targets]
        targets = torch.where(active[:, step], proposed, targets)
    return starts, tokens, targets


def sample_navigation_batch(
    pile: HierarchicalLivniumPile,
    *,
    batch_size: int,
    min_steps: int,
    max_steps: int,
    generator: torch.Generator,
) -> NavigationBatch:
    macro_start, macro_tokens, target_macro = _sample_level(
        batch_size,
        min_steps,
        max_steps,
        generator,
    )
    micro_start, micro_tokens, target_micro = _sample_level(
        batch_size,
        min_steps,
        max_steps,
        generator,
    )
    target_payloads = pile.payload_at(target_macro, target_micro).cpu()
    target_payload_ranks = pile.payload_rank_at(target_macro, target_micro).cpu()
    return NavigationBatch(
        macro_start=macro_start,
        macro_tokens=macro_tokens,
        micro_start=micro_start,
        micro_tokens=micro_tokens,
        target_macro=target_macro,
        target_micro=target_micro,
        target_payloads=target_payloads,
        target_payload_ranks=target_payload_ranks,
    )
