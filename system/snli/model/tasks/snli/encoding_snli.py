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


class LlamaCppBERTSNLIEncoder(nn.Module):
    """
    Frozen BERT encoder via llama.cpp (GGUF) — fast CPU/Metal inference.

    Drop-in replacement for BERTSNLIEncoder that uses llama-cpp-python
    instead of HuggingFace transformers. Typically 3-5x faster on CPU
    with Q8_0 quantization and negligible quality loss.

    Requires:
        pip install llama-cpp-python
        A BERT GGUF model, e.g.:
        https://huggingface.co/ggml-org/bert-base-uncased-Q8_0-GGUF

    Output dim is detected automatically from the model (768 for bert-base).
    Weights are fully frozen — only the collapse engine and head train.

    Usage in train.py:
        python train.py --encoder-type llamacpp \\
                        --llamacpp-model /path/to/bert-base-uncased.Q8_0.gguf \\
                        ...
    """

    is_bert: bool = True  # tells train_epoch to pass raw text strings, not token IDs

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        n_batch: int = 512,
        cache_path: Optional[str] = None,
    ):
        super().__init__()
        try:
            from llama_cpp import Llama as _Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed.\n"
                "Install it with:  pip install llama-cpp-python\n\n"
                "Then download a BERT GGUF model:\n"
                "  https://huggingface.co/ggml-org/bert-base-uncased\n"
                "  (pick bert-base-uncased-Q8_0.gguf)"
            )

        self._llm = _Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=False,
        )

        # Probe dim correctly: pass a bare string (not a list).
        # embed(str)  → List[List[float]]  shape (n_tokens, dim)
        # embed([str]) → List[List[List[float]]]  shape (n_inputs, n_tokens, dim)
        # _probe[0] is the CLS token embedding → length = dim (e.g. 768)
        _probe = self._llm.embed("probe")   # (n_tokens, dim)
        self.dim = len(_probe[0])            # dim of one token embedding
        self.pad_idx = 0  # kept for interface compatibility only

        # Optional pre-computed embedding cache (dict: sentence → fp16 tensor).
        # When set, build_initial_state does a pure dict lookup instead of
        # running BERT — makes training 20-50x faster when BERT is frozen.
        self._cache: Optional[dict] = None
        if cache_path is not None:
            from pathlib import Path as _Path
            if _Path(cache_path).exists():
                print(f"  Loading embedding cache from {cache_path} ...")
                self._cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                print(f"  Cache loaded: {len(self._cache):,} entries")
            else:
                print(f"  WARNING: --embed-cache path not found ({cache_path}), falling back to live inference")

        # Dummy buffer so .to(device) on the nn.Module shell works correctly.
        # llama.cpp runs outside PyTorch (on CPU / Metal backend), so we only
        # need this to know where to place the output tensors.
        self.register_buffer("_device_probe", torch.zeros(1))

    def _embed_texts(self, texts: list) -> torch.Tensor:
        """
        Embed a list of strings, using the pre-computed cache when available.

        Cache path (fast): pure dict lookup + fp16→fp32 cast. ~1000x faster
        than running BERT live. Use precompute_embeddings.py to build the cache.

        Live path (slow): one llama.cpp call per sentence (safe, avoids n_batch
        overflow from batching all sentences together).

        Pooling: CLS token (index 0), consistent with BERTSNLIEncoder.

        Returns (B, dim) float32 tensor on self._device_probe.device.
        """
        target = self._device_probe.device

        if self._cache is not None:
            # Fast path: cache lookup
            vecs = []
            for text in texts:
                if text in self._cache:
                    vecs.append(self._cache[text].float())  # fp16 → fp32
                else:
                    # Cache miss (should be rare) — fall back to live inference
                    token_embs = self._llm.embed(text)
                    vecs.append(torch.tensor(token_embs[0], dtype=torch.float32))
            return torch.stack(vecs).to(target)

        # Slow path: live llama.cpp inference, one sentence at a time
        all_vecs = []
        for text in texts:
            token_embs = self._llm.embed(text)          # (n_tokens, dim) as nested lists
            cls_vec = torch.tensor(token_embs[0], dtype=torch.float32)  # (dim,)
            all_vecs.append(cls_vec)
        return torch.stack(all_vecs).to(target)

    def build_initial_state(
        self,
        premises: list,
        hypotheses: list,
        add_noise: bool = True,
        max_len: int = 128,   # forwarded by train_epoch; llama.cpp handles truncation
        device=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed premises and hypotheses via llama.cpp, compute h0 = v_h - v_p.

        Args:
            premises:   list of B raw premise strings
            hypotheses: list of B raw hypothesis strings

        Returns:
            (h0, v_p, v_h)  all shape [B, dim]
        """
        target_device = device if device is not None else self._device_probe.device

        v_p = self._embed_texts(premises).to(target_device)
        v_h = self._embed_texts(hypotheses).to(target_device)
        h0 = v_h - v_p

        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)

        return h0, v_p, v_h

    def encode_sentence(self, token_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "LlamaCppBERTSNLIEncoder works on raw text strings. "
            "Call build_initial_state(premises, hypotheses) instead."
        )


class CrossEncoderBERTSNLIEncoder(nn.Module):
    """
    Cross-encoder BERT for NLI — the correct architecture for relational reasoning.

    Feeds BOTH sentences into BERT simultaneously:
        [CLS] premise [SEP] hypothesis [SEP]

    BERT's attention heads can now attend across sentences — "man" in the premise
    can directly attend to "dog" in the hypothesis. Role structure, negation,
    quantifier mismatches — all visible inside the transformer, not lost before
    the collapse engine sees anything.

    Returns:
        h0  = joint CLS token  — has seen both sentences via cross-attention
        v_p = mean pool of premise tokens  — for SNLIHead geometry features
        v_h = mean pool of hypothesis tokens — for SNLIHead geometry features

    One BERT forward pass. No separate encodings. No information loss.

    Compare to BERTSNLIEncoder (bi-encoder):
        BERTSNLIEncoder:       BERT(prem) → v_p
                               BERT(hyp)  → v_h
                               h0 = v_h - v_p   ← relation recovered post-hoc
        CrossEncoderBERTSNLIEncoder:
                               BERT([prem SEP hyp]) → h0, v_p, v_h  ← relation built-in

    This is how SOTA NLI models (>90%) work.
    """

    is_bert: bool = True

    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = False):
        super().__init__()
        from transformers import BertModel, BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.dim = self.bert.config.hidden_size  # 768
        self.pad_idx = self.tokenizer.pad_token_id
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad_(False)

    def build_initial_state(
        self,
        premises: list,
        hypotheses: list,
        add_noise: bool = True,
        max_len: int = 256,   # longer — holds premise + SEP + hypothesis
        device=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Joint encoding: [CLS] premise [SEP] hypothesis [SEP]

        Returns:
            h0  : (B, dim) — joint CLS, cross-attended
            v_p : (B, dim) — mean pool of premise tokens (token_type_id=0, excl CLS/SEP)
            v_h : (B, dim) — mean pool of hypothesis tokens (token_type_id=1, excl SEP)
        """
        if device is None:
            device = next(self.bert.parameters()).device

        # Tokenize jointly — HuggingFace builds:
        #   input_ids:       [CLS] p1..pN [SEP] h1..hM [SEP]
        #   token_type_ids:   0    0..0    0     1..1    1
        enc = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = self.bert(**enc)
        hidden = outputs.last_hidden_state   # (B, seq_len, dim)

        # h0 = joint CLS — entire cross-attended representation
        h0 = hidden[:, 0, :]                 # (B, dim)

        # Segment-aware mean pooling for SNLIHead geometry features.
        # token_type_ids=0 → premise side (positions 1..N, skip CLS at 0)
        # token_type_ids=1 → hypothesis side (positions N+2..M+2, skip final SEP)
        token_type_ids = enc["token_type_ids"]   # (B, seq_len)
        attention_mask = enc["attention_mask"]   # (B, seq_len)

        # v_p: premise tokens (type=0), excluding position 0 (CLS)
        prem_mask = (token_type_ids == 0) & (attention_mask == 1)
        prem_mask[:, 0] = False              # drop CLS from premise mean
        prem_vecs = hidden * prem_mask.unsqueeze(-1).float()
        v_p = prem_vecs.sum(dim=1) / prem_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

        # v_h: hypothesis tokens (type=1)
        hyp_mask = (token_type_ids == 1) & (attention_mask == 1)
        hyp_vecs = hidden * hyp_mask.unsqueeze(-1).float()
        v_h = hyp_vecs.sum(dim=1) / hyp_mask.float().sum(dim=1, keepdim=True).clamp(min=1)

        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)

        return h0, v_p, v_h


class LivniumNativeEncoder(nn.Module):
    """
    Livnium-native sentence encoder — no BERT, no pretrained weights.

    Trained end-to-end against the Livnium attractor objective.
    Maps raw text directly into the Livnium coordinate space using a small
    transformer (default: ~3M params vs BERT's 110M).

    This is Stage 4 of the Livnium pipeline: once the BERT-based model has
    discovered the attractor geometry, this encoder is trained to reproduce
    that geometry from scratch — without any external language model.

    Architecture:
        - Token embedding table (vocab_size × dim)
        - Sinusoidal positional encoding (not learned)
        - Segment embeddings (0 = premise, 1 = hypothesis, for cross-encoding)
        - N-layer transformer encoder (pre-norm, batch_first)
        - CLS token output as h0 (cross-encoder) or v_h - v_p (bi-encoder)

    Cross-encoder mode (default, use_cross_encoder=True):
        [CLS] premise [SEP] hypothesis [SEP] → joint encoding
        - Attention spans both sentences, like CrossEncoderBERTSNLIEncoder
        - Fixes role-reversal failures ("man bites dog" ≠ "dog bites man")
        - h0 = joint CLS token

    Bi-encoder mode (use_cross_encoder=False):
        Encode premise and hypothesis independently, h0 = v_h - v_p.
        Faster but loses cross-sentence signal.

    Parameter count at dim=64, 2 layers:
        Transformer (all layers)  ~100K
        Embeddings (50K × 64)     ~3.2M
        Total                     ~3.3M   (33x smaller than BERT)

    Usage in train.py:
        python train.py --encoder-type livnium \\
                        --livnium-dim 64 \\
                        --livnium-layers 2 \\
                        --livnium-nhead 4 \\
                        --livnium-cross-encoder \\
                        ...

    Anchor initialisation (recommended):
        Use extract_livnium_basis.py to project trained BERT+Livnium anchors
        into livnium_dim space, then pass --livnium-basis to train.py.
        This seeds the collapse engine with meaningful anchor positions
        instead of random initialisations.
    """

    is_bert: bool = False  # uses token IDs — train_epoch passes prem_ids/hyp_ids

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        pad_idx: int = 0,
        num_layers: int = 2,
        nhead: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        use_cross_encoder: bool = True,
    ):
        import math
        super().__init__()
        self.dim = dim
        self.pad_idx = pad_idx
        self.use_cross_encoder = use_cross_encoder
        self._math = math

        # Token embeddings
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)

        # Segment embeddings — 0=premise, 1=hypothesis (cross-encoder only)
        if use_cross_encoder:
            self.segment_embed = nn.Embedding(2, dim)

        # Sinusoidal positional encoding (fixed, not learned — saves params)
        self.register_buffer('pos_enc', self._make_sinusoidal(max_seq_len, dim))

        # Learnable special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        if use_cross_encoder:
            self.sep_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Transformer encoder — pre-norm (more stable training from scratch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

        # Weight initialisation — small uniform for embeddings, standard for transformer
        nn.init.uniform_(self.embed.weight, -0.05, 0.05)
        if use_cross_encoder:
            nn.init.uniform_(self.segment_embed.weight, -0.01, 0.01)

    @staticmethod
    def _make_sinusoidal(max_len: int, dim: int) -> torch.Tensor:
        """Standard sinusoidal positional encoding. Shape: (1, max_len, dim)."""
        import math
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        if dim % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        return pe.unsqueeze(0)  # (1, max_len, dim)

    def _encode_single(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode a single sentence batch → CLS representation.

        Args:
            token_ids: (B, L) long tensor

        Returns:
            (B, dim) — CLS token output after transformer
        """
        B, L = token_ids.shape
        x = self.embed(token_ids)                            # (B, L, dim)
        x = x + self.pos_enc[:, :L, :]                      # add positions

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)                      # (B, L+1, dim)

        # Padding mask: True = ignore that position
        # CLS is always attended; pad tokens are masked out
        cls_valid = torch.zeros(B, 1, dtype=torch.bool, device=token_ids.device)
        pad_mask = torch.cat([cls_valid, token_ids == self.pad_idx], dim=1)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        return x[:, 0, :]                                    # CLS output (B, dim)

    def _encode_joint(
        self,
        prem_ids: torch.Tensor,
        hyp_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cross-encode: [CLS] premise [SEP] hypothesis [SEP]

        BERT's cross-attention trick at 33x fewer parameters.
        Attention can span both sentences — role structure, negation,
        quantifier mismatches are all visible inside the transformer.

        Returns:
            h0  : (B, dim) — joint CLS, cross-attended
            v_p : (B, dim) — mean pool of premise tokens (excl padding)
            v_h : (B, dim) — mean pool of hypothesis tokens (excl padding)
        """
        B = prem_ids.shape[0]
        L_p = prem_ids.shape[1]
        L_h = hyp_ids.shape[1]
        device = prem_ids.device

        # Embed + positional encoding
        e_p = self.embed(prem_ids) + self.pos_enc[:, :L_p, :]     # (B, L_p, dim)
        e_h = self.embed(hyp_ids) + self.pos_enc[:, :L_h, :]      # (B, L_h, dim)

        # Segment embeddings mark which sentence each token belongs to
        zeros = torch.zeros(B, L_p, dtype=torch.long, device=device)
        ones  = torch.ones(B, L_h, dtype=torch.long, device=device)
        e_p = e_p + self.segment_embed(zeros)
        e_h = e_h + self.segment_embed(ones)

        # Special tokens (expanded to batch)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, dim)
        sep = self.sep_token.expand(B, -1, -1)   # (B, 1, dim)

        # Full sequence: [CLS] premise [SEP] hypothesis [SEP]
        # positions:      0     1..L_p  L_p+1 L_p+2..L_p+1+L_h  L_p+2+L_h
        x = torch.cat([cls, e_p, sep, e_h, sep], dim=1)   # (B, 1+L_p+1+L_h+1, dim)

        # Padding mask — False = attend, True = ignore
        false1 = torch.zeros(B, 1, dtype=torch.bool, device=device)
        prem_pad = (prem_ids == self.pad_idx)               # (B, L_p)
        hyp_pad  = (hyp_ids  == self.pad_idx)               # (B, L_h)
        pad_mask = torch.cat([false1, prem_pad, false1, hyp_pad, false1], dim=1)

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        # Extract outputs
        h0 = x[:, 0, :]                                    # joint CLS  (B, dim)

        # v_p — mean of premise region, excluding padding
        prem_region = x[:, 1:1+L_p, :]                    # (B, L_p, dim)
        prem_valid  = (~prem_pad).float().unsqueeze(-1)    # (B, L_p, 1)
        v_p = (prem_region * prem_valid).sum(1) / prem_valid.sum(1).clamp(min=1)

        # v_h — mean of hypothesis region, excluding padding
        hyp_start  = 1 + L_p + 1
        hyp_region = x[:, hyp_start:hyp_start+L_h, :]     # (B, L_h, dim)
        hyp_valid  = (~hyp_pad).float().unsqueeze(-1)
        v_h = (hyp_region * hyp_valid).sum(1) / hyp_valid.sum(1).clamp(min=1)

        return h0, v_p, v_h

    def build_initial_state(
        self,
        prem_ids: torch.Tensor,
        hyp_ids: torch.Tensor,
        add_noise: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build h0, v_p, v_h from token ID tensors.

        Cross-encoder (default): h0 is the joint CLS token — attends across both.
        Bi-encoder: h0 = v_h - v_p, sentences encoded independently.
        """
        if self.use_cross_encoder:
            h0, v_p, v_h = self._encode_joint(prem_ids, hyp_ids)
        else:
            v_p = self._encode_single(prem_ids)
            v_h = self._encode_single(hyp_ids)
            h0 = v_h - v_p

        if add_noise:
            h0 = h0 + 0.01 * torch.randn_like(h0)
        return h0, v_p, v_h

    def param_count(self) -> dict:
        """Breakdown of parameter counts by component."""
        embed_params = sum(p.numel() for p in self.embed.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        other = sum(p.numel() for n, p in self.named_parameters()
                    if 'embed' not in n and 'transformer' not in n)
        total = sum(p.numel() for p in self.parameters())
        return {
            'embeddings': embed_params,
            'transformer': transformer_params,
            'other': other,
            'total': total,
        }


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
