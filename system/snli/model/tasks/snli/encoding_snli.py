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
