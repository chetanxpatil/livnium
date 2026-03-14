"""
QuantumEmbeddingTextEncoder [M5 Optimized]

Uses a pre-trained Livnium quantum embedding table produced by
train_quantum_embeddings.py. Drop-in replacement for the legacy
TextEncoder in nova_v3.

Optimized for M-series: ensures custom BasinField moves to the same device as
the module (so MPS works correctly).
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import torch
import torch.nn as nn

from vector_collapse import VectorCollapseEngine
from basin_field import BasinField


class QuantumTextEncoder(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()

        data = torch.load(ckpt_path, map_location="cpu")
        emb = data["embeddings"]          # [vocab_size, dim]
        vocab = data["vocab"]
        self.idx2word = vocab["idx2word"]
        self.word2idx: Dict[str, int] = {w: i for i, w in enumerate(self.idx2word)}
        self.pad_idx = vocab["pad_idx"]
        self.unk_idx = vocab["unk_idx"]
        self.dim = emb.size(1)

        self.embed = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=self.pad_idx)

        # Optional collapse state (dynamic basins)
        self.use_dynamic_basins: bool = bool(data.get("use_dynamic_basins", False))
        self.collapse_engine: Optional[VectorCollapseEngine] = None
        self.basin_field: Optional[BasinField] = None
        if self.use_dynamic_basins and "collapse_engine" in data and "basin_field" in data:
            cfg = data.get("collapse_config", {})
            self.collapse_engine = VectorCollapseEngine(
                dim=self.dim,
                num_layers=cfg.get("num_layers", 4),
                strength_entail=cfg.get("strength_entail", 0.1),
                strength_contra=cfg.get("strength_contra", 0.1),
                strength_neutral=cfg.get("strength_neutral", 0.05),
                basin_tension_threshold=cfg.get("basin_tension_threshold", 0.15),
                basin_align_threshold=cfg.get("basin_align_threshold", 0.6),
                basin_anchor_lr=cfg.get("basin_anchor_lr", 0.05),
                basin_prune_min_count=cfg.get("basin_prune_min_count", 10),
                basin_prune_merge_cos=cfg.get("basin_prune_merge_cos", 0.97),
            )
            self.collapse_engine.load_state_dict(data["collapse_engine"])
            self.basin_field = BasinField(max_basins_per_label=data["basin_field"].get("max_basins_per_label", 64))
            self.basin_field.load_state_dict(data["basin_field"])

    def to(self, device: torch.device):
        """
        Override to move non-Module BasinField alongside the encoder.
        """
        super().to(device)
        if self.basin_field is not None:
            self.basin_field.to(device)
        return self

    def tokenize(self, text: str) -> List[str]:
        pattern = r"(\w+|\s+|[^\w\s])"
        return [t for t in re.split(pattern, text) if t.strip()]

    def encode_tokens(self, tokens: List[str]) -> torch.Tensor:
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [seq_len] or [batch, seq_len]
        Returns: [dim] or [batch, dim]
        """
        if token_ids.device != self.embed.weight.device:
            token_ids = token_ids.to(self.embed.weight.device)
        emb = self.embed(token_ids)
        mask = (token_ids != self.pad_idx).float().unsqueeze(-1)

        if token_ids.dim() == 1:
            masked = emb * mask
            denom = mask.sum(dim=0).clamp(min=1.0)
            return (masked.sum(dim=0) / denom)
        else:
            masked = emb * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (masked.sum(dim=1) / denom)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_sentence(token_ids)

    def collapse_sentence(
        self,
        token_ids: torch.Tensor,
        label: int = 2,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        """
        Optional helper: compute collapsed vector given a label (0=E,1=C,2=N).
        Does not spawn/prune anchors at inference time.
        """
        if not (self.use_dynamic_basins and self.collapse_engine and self.basin_field):
            return self.forward(token_ids), {}
        if device is None:
            device = self.embed.weight.device
        if self.collapse_engine.anchor_entail.device != device:
            self.collapse_engine.to(device)
        self.basin_field.to(device)
        token_ids = token_ids.to(device)
        h0 = self.encode_sentence(token_ids)
        labels = (
            torch.tensor([label], device=device, dtype=torch.long)
            if h0.dim() == 1
            else torch.full((h0.size(0),), label, device=device, dtype=torch.long)
        )
        h_final, trace = self.collapse_engine.collapse_dynamic(
            h0,
            labels,
            self.basin_field,
            global_step=0,
            spawn_new=False,
            prune_every=0,
            update_anchors=False,
        )
        return h_final, trace
