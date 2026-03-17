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


class BERTSNLIEncoder(nn.Module):
    """
    Frozen BERT encoder for A/B comparison against bag-of-words.

    Uses bert-base-uncased CLS token as sentence representation.
    Weights are frozen by default — only the attractor head (or linear
    baseline) trains. This isolates the question: does the attractor head
    outperform a linear head given identical, high-quality features?

    dim = 768 (BERT hidden size, fixed)

    Usage in train.py: pass raw text strings to build_initial_state
    instead of token ID tensors — BERT tokenizes internally.
    Detect via: getattr(encoder, 'is_bert', False)
    """

    is_bert: bool = True

    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = True):
        super().__init__()
        from transformers import BertModel, BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dim = self.bert.config.hidden_size  # 768
        self.pad_idx = self.tokenizer.pad_token_id
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad_(False)

    def _encode(self, texts: list, max_len: int, device) -> torch.Tensor:
        """Tokenize and encode a list of strings → CLS token [B, 768]."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.bert(**enc)
        return out.last_hidden_state[:, 0, :]  # CLS token

    def build_initial_state(
        self,
        premises: list,
        hypotheses: list,
        add_noise: bool = True,
        max_len: int = 128,
        device=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            premises:   list of raw premise strings
            hypotheses: list of raw hypothesis strings
        Returns:
            (h0, v_p, v_h) where h0 = v_h - v_p, all [B, 768]
        """
        if device is None:
            device = next(self.bert.parameters()).device
        v_p = self._encode(premises, max_len, device)
        v_h = self._encode(hypotheses, max_len, device)
        h0 = v_h - v_p
        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h
