"""
CollapseEmbeddingTextEncoder [M5 Optimized]

Uses a pre-trained Livnium collapse embedding table produced by
train_collapse_embeddings.py. Drop-in replacement for the legacy
TextEncoder in nova_v3.

Optimized for M-series: ensures custom BasinField moves to the same device as
the module (so MPS works correctly).
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from basin_field import BasinField
from vector_collapse import VectorCollapseEngine


class CollapseTextEncoder(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()

        data = torch.load(ckpt_path, map_location="cpu")
        emb = data["embeddings"]  # [vocab_size, dim]
        vocab = data["vocab"]
        self.idx2word = vocab["idx2word"]
        self.word2idx: Dict[str, int] = {w: i for i, w in enumerate(self.idx2word)}
        self.pad_idx = vocab["pad_idx"]
        self.unk_idx = vocab["unk_idx"]
        self.dim = emb.size(1)

        self.embed = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=self.pad_idx)

        # Optional collapse state.
        self.use_dynamic_basins: bool = bool(data.get("use_dynamic_basins", False))
        self.collapse_engine: Optional[VectorCollapseEngine] = None
        self.basin_field: Optional[BasinField] = None

        # Load the collapse engine whenever the checkpoint carries one, REGARDLESS
        # of dynamic-basin mode. Previously this was gated on use_dynamic_basins,
        # so static-collapse checkpoints (use_dynamic_basins=False) silently
        # discarded the trained engine and inference fell back to plain
        # mean-pooled embeddings. See COLLAPSE_ENGINE_VERDICT.md.
        if data.get("collapse_engine") is not None:
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

        # The basin field only matters for the dynamic path.
        if self.use_dynamic_basins and data.get("basin_field") is not None:
            bf_sd = data["basin_field"]
            # Infer capacity from the stored buffer rather than a missing key.
            max_b = bf_sd["centers"].shape[1] if "centers" in bf_sd else 64
            self.basin_field = BasinField(dim=self.dim, max_basins_per_label=max_b)
            self.basin_field.load_state_dict(bf_sd)

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
            return masked.sum(dim=0) / denom
        else:
            masked = emb * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            return masked.sum(dim=1) / denom

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
        if self.collapse_engine is None:
            return self.forward(token_ids), {}
        if device is None:
            device = self.embed.weight.device
        if self.collapse_engine.anchor_entail.device != device:
            self.collapse_engine.to(device)
        token_ids = token_ids.to(device)
        h0 = self.encode_sentence(token_ids)

        # Static collapse path: engine loaded but dynamic basins are off. Run the
        # learned three-anchor collapse directly (previously this case returned the
        # raw embedding and the trained engine was never applied at inference).
        if not (self.use_dynamic_basins and self.basin_field is not None):
            batched = h0 if h0.dim() > 1 else h0.unsqueeze(0)
            h_final, trace = self.collapse_engine(batched)
            return (h_final if h0.dim() > 1 else h_final.squeeze(0)), trace

        self.basin_field.to(device)
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
