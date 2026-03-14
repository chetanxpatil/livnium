"""
BERTSNLIEncoder
===============
Drop-in replacement for QuantumSNLIEncoder.
Uses bert-base-uncased to produce v_p, v_h, h0.

Interface matches QuantumSNLIEncoder.build_initial_state():
    h0, v_p, v_h = encoder.build_initial_state(prem_ids, hyp_ids)

But we also support raw text input via build_initial_state_text().
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BERTSNLIEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        model_name: str = "bert-base-uncased",
        freeze_bert: bool = False,
        pooling: str = "mean",          # "mean" or "cls"
    ):
        super().__init__()
        self.out_dim = out_dim
        self.pooling = pooling

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        bert_dim = self.bert.config.hidden_size   # 768 for bert-base
        # Project 768 → out_dim
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def _encode_texts(self, texts: list, device) -> torch.Tensor:
        """Tokenize + BERT encode a list of strings → (B, out_dim)."""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = self.bert(**enc)

        if self.pooling == "cls":
            hidden = out.last_hidden_state[:, 0, :]        # [CLS] token
        else:
            # mean pool over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            hidden = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-8)

        return self.proj(hidden)                            # (B, out_dim)

    def build_initial_state(self, premises: list, hypotheses: list, device=None):
        """
        Args:
            premises:   list of str
            hypotheses: list of str
            device:     torch.device (inferred if None)
        Returns:
            h0   (B, out_dim)  — v_p - v_h  (difference vector)
            v_p  (B, out_dim)  — premise embedding
            v_h  (B, out_dim)  — hypothesis embedding
        """
        if device is None:
            device = next(self.parameters()).device

        v_p = self._encode_texts(premises, device)
        v_h = self._encode_texts(hypotheses, device)
        h0  = v_p - v_h
        return h0, v_p, v_h
