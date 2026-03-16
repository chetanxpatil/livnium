"""
SNLI Encoding: Premise/Hypothesis → Initial State

Builds initial state h0 = v_h − v_p from premise and hypothesis.
"""

import torch
import torch.nn as nn
from typing import List, Sequence, Tuple, Optional

# Use absolute import so module works when run as a script entry point
from text.encoder import TextEncoder
from embed.text_encoder import PretrainedTextEncoder


class SNLIEncoder(nn.Module):
    """
    SNLI-specific encoder (vocab-based).

    Takes premise and hypothesis token IDs, builds initial state h0.
    Returns (h0, v_p, v_h) where h0 = v_h − v_p.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.text_encoder = TextEncoder(vocab_size, dim, pad_idx)

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.encode_sentence(token_ids)

    def build_initial_state(
        self,
        prem_ids: torch.Tensor,
        hyp_ids: torch.Tensor,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_p = self.encode_sentence(prem_ids)
        v_h = self.encode_sentence(hyp_ids)
        h0 = v_h - v_p
        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h


class PretrainedSNLIEncoder(nn.Module):
    """
    SNLI encoder backed by the pretrained Livnium embedding table.

    alpha_first_token: symmetry-breaking injection.
        Bag-of-words mean collapses "A bites B" and "B bites A" to the
        same coordinate. Injecting a scaled first-token difference breaks
        that symmetry at zero parameter cost:

            h0 = (v_h − v_p) + alpha * (embed(hyp[0]) − embed(prem[0]))

        The first token is almost always the subject in English.
        Even a small alpha gives the collapse engine a directional crack.

        alpha=0.0  →  pure bag-of-words (symmetry intact)
        alpha=0.1  →  weak signal, minimal interference
        alpha=0.25 →  stronger subject signal
    """

    def __init__(self, ckpt_path: str, alpha_first_token: float = 0.0):
        super().__init__()
        self.text_encoder = PretrainedTextEncoder(ckpt_path)
        self.dim = self.text_encoder.dim
        self.pad_idx = self.text_encoder.pad_idx
        self.alpha_first_token = alpha_first_token

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder.encode_sentence(token_ids)

    def build_initial_state(
        self,
        prem_ids: torch.Tensor,
        hyp_ids: torch.Tensor,
        add_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_p = self.encode_sentence(prem_ids)
        v_h = self.encode_sentence(hyp_ids)
        h0 = v_h - v_p

        if self.alpha_first_token != 0.0:
            te = self.text_encoder
            device = te.embed.weight.device
            p0 = prem_ids[:, 0].to(device)
            h0_tok = hyp_ids[:, 0].to(device)
            v_p_first = te.embed(p0)
            v_h_first = te.embed(h0_tok)
            h0 = h0 + self.alpha_first_token * (v_h_first - v_p_first)

        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h


# Backward-compat alias — saved checkpoints reference 'QuantumSNLIEncoder'
QuantumSNLIEncoder = PretrainedSNLIEncoder
