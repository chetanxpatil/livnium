"""Trainable model whose persistent internal memory is a Livnium pile."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import ACTION_NAMES, action_permutations
from .pile import HierarchicalLivniumPile

PAD_TOKEN = len(ACTION_NAMES)


@dataclass
class LivniumOutput:
    payload_probabilities: torch.Tensor
    macro_distribution: torch.Tensor
    micro_distribution: torch.Tensor
    macro_action_probabilities: torch.Tensor
    micro_action_probabilities: torch.Tensor


class LivniumActionHead(nn.Module):
    """Learns what arbitrary instruction tokens mean in Livnium's action space."""

    def __init__(self, embedding_dim: int = 24, hidden_dim: int = 48):
        super().__init__()
        self.embedding = nn.Embedding(
            len(ACTION_NAMES) + 1,
            embedding_dim,
            padding_idx=PAD_TOKEN,
        )
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(ACTION_NAMES)),
        )

    def forward(self, instruction_tokens: torch.Tensor) -> torch.Tensor:
        return self.network(self.embedding(instruction_tokens))


class LivniumNativeModel(nn.Module):
    """A neural action head connected directly to a persistent Livnium pile."""

    def __init__(
        self,
        pile: HierarchicalLivniumPile | None = None,
        *,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.pile = pile if pile is not None else HierarchicalLivniumPile()
        self.action_head = LivniumActionHead()
        self.temperature = float(temperature)
        self.register_buffer("observer_permutations", action_permutations())

    @property
    def device(self) -> torch.device:
        return self.observer_permutations.device

    def _navigate(
        self,
        start_indices: torch.Tensor,
        instruction_tokens: torch.Tensor,
        *,
        hard: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start_indices.ndim != 1:
            raise ValueError("start_indices must have shape [batch]")
        if instruction_tokens.ndim != 2:
            raise ValueError("instruction_tokens must have shape [batch, steps]")
        if instruction_tokens.shape[0] != start_indices.shape[0]:
            raise ValueError("start_indices and instruction_tokens batch sizes differ")

        position = F.one_hot(start_indices, num_classes=27).to(
            dtype=self.observer_permutations.dtype
        )
        all_action_probabilities: list[torch.Tensor] = []

        for step in range(instruction_tokens.shape[1]):
            tokens = instruction_tokens[:, step]
            logits = self.action_head(tokens)
            soft_actions = torch.softmax(logits / self.temperature, dim=-1)
            if hard:
                chosen = soft_actions.argmax(dim=-1)
                actions = F.one_hot(chosen, num_classes=len(ACTION_NAMES)).to(
                    dtype=soft_actions.dtype
                )
            else:
                actions = soft_actions

            candidates = torch.einsum(
                "bi,aij->baj",
                position,
                self.observer_permutations,
            )
            proposed = torch.einsum("ba,baj->bj", actions, candidates)
            active = (tokens != PAD_TOKEN).unsqueeze(1)
            position = torch.where(active, proposed, position)
            all_action_probabilities.append(soft_actions)

        if all_action_probabilities:
            action_probabilities = torch.stack(all_action_probabilities, dim=1)
        else:
            action_probabilities = torch.empty(
                (start_indices.shape[0], 0, len(ACTION_NAMES)),
                device=start_indices.device,
            )
        return position, action_probabilities

    def forward(
        self,
        macro_start: torch.Tensor,
        macro_instructions: torch.Tensor,
        micro_start: torch.Tensor,
        micro_instructions: torch.Tensor,
        *,
        hard: bool = False,
    ) -> LivniumOutput:
        macro_distribution, macro_actions = self._navigate(
            macro_start,
            macro_instructions,
            hard=hard,
        )
        micro_distribution, micro_actions = self._navigate(
            micro_start,
            micro_instructions,
            hard=hard,
        )
        payload_probabilities = self.pile.read(
            macro_distribution,
            micro_distribution,
        )
        return LivniumOutput(
            payload_probabilities=payload_probabilities,
            macro_distribution=macro_distribution,
            micro_distribution=micro_distribution,
            macro_action_probabilities=macro_actions,
            micro_action_probabilities=micro_actions,
        )

    @staticmethod
    def answer_loss(output: LivniumOutput, target_payloads: torch.Tensor) -> torch.Tensor:
        probabilities = output.payload_probabilities.clamp_min(1e-12)
        return F.nll_loss(probabilities.log(), target_payloads)

    def learned_action_map(self) -> torch.Tensor:
        """Return the hard action currently assigned to each non-padding token."""
        tokens = torch.arange(len(ACTION_NAMES), device=self.device)
        return self.action_head(tokens).argmax(dim=-1)
