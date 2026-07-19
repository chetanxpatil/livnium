"""
experiment/pure_reply.py — 100% Clean-Room Pure Geometric Chat Collapse Engine

This is a parameter-free (MLP-free, linear-free) chat reply generator.
The state trajectory and decode step dynamics are driven entirely by closed-form
energy gradients of V(h) = -cos(h, T). All neural projection layers are removed.
"""

import sys
import os
import argparse
import time
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the active chat experiment for its data/model helpers.
HERE = os.path.dirname(os.path.abspath(__file__))
CHAT_DIR = os.path.join(HERE, "..", "chat-brain")
sys.path.append(CHAT_DIR)

from chat_typer import MAXLEN, PAD, SEED
from char_fingerprint import (
    LETTERS, MAX_WORD as CHAR_MAX_WORD,
    letter_anchors, char_fingerprint
)
from chat_reply import (
    SPECIALS, CTX_WORDS,
    read_pairs, semantic_init, meaning_weights,
    CharMinter, WordMinter, decode
)

# Default paths pointing to chat/ assets
DEFAULT_DATA = os.path.join(CHAT_DIR, "data", "chat_context.tsv")
DEFAULT_TYPER_CKPT = os.path.join(CHAT_DIR, "model", "chat_typer.pt")
DEFAULT_FAST_CKPT = os.path.join(CHAT_DIR, "model", "fast_reader.pt")
DEFAULT_SEMANTIC_INIT = os.path.join(HERE, "..", "..", "models", "noun-collapse", "model", "noun_collapse_pure.pt")
CKPT_OUT_DIR = os.path.join(HERE, "model")
CKPT_OUT = os.path.join(CKPT_OUT_DIR, "chat_reply_pure.pt")


def encode_context(sents, stoi, unk, eos, ctx_words=CTX_WORDS, minter=None):
    """Context encoder. ctx_words <= 0 keeps every token in each prepared context."""
    token_rows = []
    width = 0
    for s in sents:
        toks = s.split()
        if ctx_words > 0:
            toks = toks[-ctx_words:]
        token_rows.append(toks)
        width = max(width, len(toks))
    width = max(width + 2, 1)
    out = []
    for toks in token_rows:
        ids = [stoi[t] if t in stoi else
               (minter.id_for(t) if minter is not None else unk) for t in toks]
        ids += [eos] + [PAD] * (width - len(ids) - 1)
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)


def encode_reply(sents, stoi, unk, eos, reply_words=MAXLEN - 2):
    """Reply encoder. reply_words <= 0 keeps every token in each prepared reply."""
    token_rows = []
    width = 0
    for s in sents:
        toks = s.split()
        if reply_words > 0:
            toks = toks[:reply_words]
        token_rows.append(toks)
        width = max(width, len(toks))
    width = max(width + 2, 1)
    out = []
    for toks in token_rows:
        ids = [stoi.get(t, unk) for t in toks]
        ids += [eos] + [PAD] * (width - len(ids) - 1)
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)


def char_fingerprint_many(words, dim, out_device, chunk_size=4096):
    """Vectorized char fingerprints for large uncapped vocab builds."""
    if not words:
        return torch.empty(0, dim, device=out_device)
    anchors = letter_anchors(dim, device="cpu")
    stride = max(1, dim // CHAR_MAX_WORD)
    table = torch.zeros(CHAR_MAX_WORD, len(LETTERS) + 1, dim)
    for pos in range(CHAR_MAX_WORD):
        shift = pos * stride
        for li in range(len(LETTERS)):
            table[pos, li + 1] = torch.roll(anchors[li], shifts=shift, dims=-1)
    letter_to_id = {c: i + 1 for i, c in enumerate(LETTERS)}
    pos_idx_base = torch.arange(CHAR_MAX_WORD).view(1, -1)
    chunks = []
    for start in range(0, len(words), chunk_size):
        batch = words[start:start + chunk_size]
        ids = torch.zeros(len(batch), CHAR_MAX_WORD, dtype=torch.long)
        fallback_rows = []
        for row, word in enumerate(batch):
            chars = [letter_to_id[c] for c in word.lower()
                     if c in letter_to_id][:CHAR_MAX_WORD]
            if chars:
                ids[row, :len(chars)] = torch.tensor(chars, dtype=torch.long)
            else:
                fallback_rows.append(row)
        pos_idx = pos_idx_base.expand(len(batch), -1)
        vec = table[pos_idx, ids].sum(1)
        if fallback_rows:
            for row in fallback_rows:
                vec[row] = char_fingerprint(batch[row], anchors)
        chunks.append(F.normalize(vec, dim=-1))
    return torch.cat(chunks, dim=0).to(out_device)


def shrink_vocab_fast(train_pairs, warm, stoi, itos, unk, eos, min_freq):
    """Large-corpus version of chat_reply.shrink_vocab with batched minting."""
    from collections import Counter

    cnt = Counter()
    for m, r in train_pairs:
        cnt.update(m.split())
        cnt.update(r.split())
    keep = sorted((w for w, c in cnt.items()
                   if c >= min_freq and w not in SPECIALS),
                  key=lambda w: (-cnt[w], w))
    dim = warm.size(1)
    rows = [warm[PAD]]
    unknown_words = [w for w in keep if w not in stoi]
    if unknown_words:
        print(f"minting {len(unknown_words):,} fresh letter wells in batches ...",
              flush=True)
    minted_rows = char_fingerprint_many(unknown_words, dim, warm.device)
    minted_i = 0
    for w in keep:
        if w in stoi:
            rows.append(warm[stoi[w]])
        else:
            rows.append(minted_rows[minted_i])
            minted_i += 1
    rows += [warm[unk], warm[eos]]
    new_stoi = {w: i + 1 for i, w in enumerate(keep)}
    new_unk, new_eos = len(keep) + 1, len(keep) + 2
    new_itos = {i + 1: w for i, w in enumerate(keep)}
    new_itos[new_unk] = "<unk>"
    new_itos[new_eos] = "<eos>"
    n = new_eos + 1
    for tok in SPECIALS:
        new_stoi[tok] = n
        new_itos[n] = tok
        rows.append(warm[stoi[tok]])
        n += 1
    return torch.stack(rows), new_stoi, new_itos, new_unk, new_eos, n, len(unknown_words)


def assert_not_none(name, value):
    assert value is not None, f"{name} cannot be None"
    return value


def assert_no_none_tree(name, value):
    """Checkpoint/config guard: required saved payloads must not contain None."""
    assert value is not None, f"{name} cannot be None"
    if isinstance(value, dict):
        for k, v in value.items():
            assert k is not None, f"{name} contains a None key"
            assert_no_none_tree(f"{name}.{k}", v)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            assert_no_none_tree(f"{name}[{i}]", v)


def assert_tensor(name, value, *, ndim=None, shape=None, dtype=None,
                  device=None, finite=False, nonempty=True):
    assert_not_none(name, value)
    assert torch.is_tensor(value), f"{name} must be a torch.Tensor, got {type(value).__name__}"
    if ndim is not None:
        assert value.ndim == ndim, f"{name} must be {ndim}D, got shape {tuple(value.shape)}"
    if shape is not None:
        assert tuple(value.shape) == tuple(shape), (
            f"{name} shape {tuple(value.shape)} does not match expected {tuple(shape)}"
        )
    if dtype is not None:
        assert value.dtype == dtype, f"{name} dtype {value.dtype} does not match {dtype}"
    if device is not None:
        expected = torch.device(device)
        if expected.index is None:
            assert value.device.type == expected.type, (
                f"{name} device {value.device} does not match {expected}"
            )
        else:
            assert value.device == expected, f"{name} device {value.device} does not match {expected}"
    if nonempty:
        assert value.numel() > 0, f"{name} cannot be empty"
    if finite and value.numel() > 0:
        assert bool(torch.isfinite(value).all().item()), f"{name} contains NaN or Inf"
    return value


def assert_id_tensor(name, value, *, upper, allow_empty=False):
    assert_tensor(name, value, ndim=2, dtype=torch.long, nonempty=not allow_empty)
    assert upper > 0, f"{name} upper bound must be positive, got {upper}"
    if value.numel() == 0:
        return value
    lo = int(value.min().item())
    hi = int(value.max().item())
    assert lo >= 0, f"{name} contains negative token id {lo}"
    assert hi < upper, f"{name} token id {hi} is outside table size {upper}"
    return value


def assert_token_maps(stoi, itos, *, n_words, unk, eos):
    assert isinstance(stoi, dict), f"stoi must be a dict, got {type(stoi).__name__}"
    assert isinstance(itos, dict), f"itos must be a dict, got {type(itos).__name__}"
    assert n_words > 0, f"n_words must be positive, got {n_words}"
    assert 0 <= unk < n_words, f"unk id {unk} outside vocab size {n_words}"
    assert 0 <= eos < n_words, f"eos id {eos} outside vocab size {n_words}"
    assert len(stoi) > 0, "stoi cannot be empty"
    assert len(itos) > 0, "itos cannot be empty"
    ids = list(stoi.values())
    assert all(isinstance(k, str) and k for k in stoi), "stoi keys must be non-empty strings"
    assert all(isinstance(v, int) for v in ids), "stoi values must be ints"
    assert all(0 <= v < n_words for v in ids), "stoi contains an id outside vocab bounds"
    assert len(set(ids)) == len(ids), "stoi contains duplicate token ids"
    assert itos.get(unk) == "<unk>", "itos[unk] must be '<unk>'"
    assert itos.get(eos) == "<eos>", "itos[eos] must be '<eos>'"
    for tok in SPECIALS:
        assert tok in stoi, f"missing special token {tok!r} in stoi"
        assert itos.get(stoi[tok]) == tok, f"itos mismatch for special token {tok!r}"


def coerce_token_maps(stoi, itos):
    assert_not_none("stoi", stoi)
    assert_not_none("itos", itos)
    if not isinstance(stoi, dict):
        stoi = dict(stoi)
    if not isinstance(itos, dict):
        itos = dict(itos)
    stoi = {str(k): int(v) for k, v in stoi.items()}
    itos = {int(k): str(v) for k, v in itos.items()}
    return stoi, itos


def assert_checkpoint_payload(state):
    required = ("epoch", "state_dict", "config", "stoi", "itos", "unk", "eos", "best")
    for key in required:
        assert key in state, f"checkpoint missing required key {key!r}"
    assert_no_none_tree("checkpoint", state)
    assert isinstance(state["state_dict"], dict) and state["state_dict"], (
        "checkpoint state_dict cannot be empty"
    )
    for name, value in state["state_dict"].items():
        assert_tensor(f"state_dict.{name}", value, finite=True)


def assert_model_finite(model):
    for name, param in model.named_parameters():
        assert_tensor(f"model parameter {name}", param.detach(), finite=True)


# ---------------------------------------------------------------- the model

class PureReplyBrain(nn.Module):
    """100% Pure Geometric Chat Generator. No MLPs, no attention keys/queries."""

    def __init__(self, n_words, dim, eos, warm=None, pos_well=True, align=True,
                 max_reply_len=MAXLEN):
        super().__init__()
        assert isinstance(n_words, int) and n_words > 0, "n_words must be a positive int"
        assert isinstance(dim, int) and dim > 0, "dim must be a positive int"
        assert isinstance(eos, int) and 0 <= eos < n_words, "eos must be inside the vocab"
        assert isinstance(max_reply_len, int) and max_reply_len > 0, (
            "max_reply_len must be a positive int"
        )
        self.eos, self.dim, self.n_words = eos, dim, n_words
        self.max_reply_len = max_reply_len
        self.pos_well = bool(pos_well)
        self.align = bool(align)
        
        # training levers, set from outside:
        self.sample_p = 0.0       # scheduled sampling probability
        self.neg_samples = 0      # sampled-softmax negatives (0 = full CE)
        self.word_w = None        # per-word loss weights (meaning weighting)
        self.fast_alpha = None    # distilled reader taps
        self.oov_wells = None     # frozen char wells for OOV words (READ only)

        # Word Anchors: warm-started from the Wikipedia pre-trained collapse wells
        self.word_anchors = nn.Parameter(torch.randn(n_words, dim) / dim ** 0.5)
        if warm is not None:
            assert_tensor("warm", warm, ndim=2, shape=(n_words, dim), finite=True)
            with torch.no_grad():
                self.word_anchors.copy_(warm)
                self.word_anchors[PAD].zero_()

        # Reader: learned start vector + collapse strength
        self.start = nn.Parameter(torch.randn(dim) * 0.05)
        self.log_strength_read = nn.Parameter(torch.tensor(2.2))

        # Writer: position wells + alignment temp + strength + temp
        if self.pos_well:
            self.pos_anchor = nn.Parameter(torch.randn(max_reply_len + 2, dim) * 0.05)
        if self.align:
            self.log_align_temp = nn.Parameter(torch.tensor(0.0))
            
        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def strength(self):
        return torch.sigmoid(self.log_strength)

    @property
    def temp(self):
        return F.softplus(self.log_temp) + 1e-3

    @property
    def strength_read(self):
        return torch.sigmoid(self.log_strength_read)

    @property
    def align_temp(self):
        return F.softplus(self.log_align_temp) + 1e-3

    # -- reading ------------------------------------------------------------

    def read(self, ids, A):
        """Sequential context reader using pure analytical energy gradient."""
        assert_tensor("read.ids", ids, ndim=2, dtype=torch.long)
        assert_tensor("read.A", A, ndim=2, device=ids.device, nonempty=True)
        assert A.size(1) == self.dim, f"read.A dim {A.size(1)} does not match model dim {self.dim}"
        B, L = ids.shape
        assert B > 0 and L > 0, f"read.ids must have non-empty batch and sequence, got {ids.shape}"
        mask = (ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        s = self.strength_read
        states = []
        for i in range(L):
            target = self.lookup_wells(ids[:, i], A)
            m = mask[:, i].float().unsqueeze(-1)
            
            h_norm = h.norm(dim=-1, keepdim=True)
            h_n = h / (h_norm + 1e-8)
            align = (h_n * target).sum(-1, keepdim=True)
            
            # Analytical energy gradient of V(h) = -cos(h, target)
            grad = -(target - h_n * align) / (h_norm + 1e-8)
            h = h + m * (-s * grad)
            
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
            
        S = torch.stack(states, dim=1)
        return F.normalize(S, dim=-1), mask, h

    def read_fast(self, ids, A):
        """Distilled convolution reader."""
        assert_tensor("read_fast.ids", ids, ndim=2, dtype=torch.long)
        assert_tensor("read_fast.A", A, ndim=2, device=ids.device, nonempty=True)
        assert self.fast_alpha is not None, "fast_alpha cannot be None in read_fast"
        assert_tensor("fast_alpha", self.fast_alpha, ndim=1, device=ids.device, finite=True)
        assert A.size(1) == self.dim, (
            f"read_fast.A dim {A.size(1)} does not match model dim {self.dim}"
        )
        mask = (ids != PAD)
        wells = self.lookup_wells(ids, A) * mask.unsqueeze(-1)
        B, L, D = wells.shape
        assert B > 0 and L > 0 and D == self.dim, f"bad fast-reader wells shape {wells.shape}"
        K = self.fast_alpha.numel()
        assert K > 0, "fast_alpha cannot be empty"
        x = F.pad(wells.transpose(1, 2), (K - 1, 0))
        w = self.fast_alpha.flip(0).view(1, 1, K).expand(D, 1, K)
        S = F.conv1d(x, w, groups=D).transpose(1, 2)
        lens = mask.sum(1).clamp(min=1) - 1
        hT = S[torch.arange(B, device=ids.device), lens]
        return F.normalize(S + 1e-8, dim=-1), mask, hT

    def read_context(self, ids, A):
        if self.fast_alpha is not None:
            return self.read_fast(ids, A)
        return self.read(ids, A)

    def lookup_wells(self, ids, A):
        """Gather trained and OOV read wells without materializing [A | OOV]."""
        assert_tensor("lookup.ids", ids, dtype=torch.long, device=A.device)
        assert_tensor("lookup.A", A, ndim=2, device=ids.device)
        assert A.size(0) == self.n_words, (
            f"trained well rows {A.size(0)} must equal n_words {self.n_words}"
        )
        assert A.size(1) == self.dim, f"trained well dim {A.size(1)} != model dim {self.dim}"
        if self.oov_wells is None or self.oov_wells.numel() == 0:
            return A[ids]
        assert_tensor("lookup.oov_wells", self.oov_wells, ndim=2, device=A.device)
        assert self.oov_wells.size(1) == self.dim, (
            f"OOV well dim {self.oov_wells.size(1)} != model dim {self.dim}"
        )
        trained_mask = ids < self.n_words
        trained_ids = ids.clamp(max=self.n_words - 1)
        wells = A[trained_ids]
        oov_ids = (ids - self.n_words).clamp(min=0)
        oov = self.oov_wells[oov_ids]
        return torch.where(trained_mask.unsqueeze(-1), wells, oov)

    def read_table(self, A):
        assert_tensor("read_table.A", A, ndim=2, finite=False)
        if self.oov_wells is None or self.oov_wells.numel() == 0:
            return A
        assert_tensor("oov_wells", self.oov_wells, ndim=2, device=A.device, finite=True)
        assert self.oov_wells.size(1) == A.size(1), (
            f"oov_wells dim {self.oov_wells.size(1)} does not match A dim {A.size(1)}"
        )
        return torch.cat([A, self.oov_wells], dim=0)

    # -- writing ------------------------------------------------------------

    def collapse_step(self, h, target):
        """Sequential writing step using analytical energy gradient."""
        assert_tensor("collapse.h", h, ndim=2)
        assert_tensor("collapse.target", target, ndim=2, device=h.device)
        assert h.shape == target.shape, (
            f"collapse target shape {tuple(target.shape)} must match h shape {tuple(h.shape)}"
        )
        h_norm = h.norm(dim=-1, keepdim=True)
        h_n = h / (h_norm + 1e-8)
        align = (h_n * target).sum(-1, keepdim=True)
        
        # Analytical energy gradient of V(h) = -cos(h, target)
        grad = -(target - h_n * align) / (h_norm + 1e-8)
        h = h - self.strength * grad
        
        n = h.norm(dim=-1, keepdim=True)
        return torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)

    def reply_nll(self, msg_ids, rep_ids, reduce_mean=True):
        """Teacher-forced NLL of the reply under pure geometric equations."""
        assert_tensor("reply_nll.msg_ids", msg_ids, ndim=2, dtype=torch.long)
        assert_tensor("reply_nll.rep_ids", rep_ids, ndim=2, dtype=torch.long, device=msg_ids.device)
        A = F.normalize(self.word_anchors, dim=-1)
        
        # READ: Context -> trajectory of states & final state
        _, _, hread = self.read_context(msg_ids, A)
        if self.align:
            Cwells = self.lookup_wells(msg_ids, A)
            cmask = (msg_ids != PAD)
            
        # THINK: Zero-parameter semantic bridge
        z = hread
        h = z
        
        B, L = rep_ids.shape
        assert B == msg_ids.size(0), "message and reply batch sizes must match"
        assert L > 0, "reply sequence length cannot be zero"
        tok_nll = torch.zeros(B, device=rep_ids.device)
        tok_cnt = torch.zeros(B, device=rep_ids.device)
        sampled = self.training and self.neg_samples > 0
        
        for t in range(L):
            hn = F.normalize(h, dim=-1)
            q = hn + z
            
            if self.pos_well:
                q = q + self.pos_anchor[min(t, self.pos_anchor.size(0) - 1)]
                
            if self.align:
                probe = F.normalize(hn + z, dim=-1).unsqueeze(1)
                sc = torch.bmm(probe, Cwells.transpose(1, 2)).squeeze(1) / self.align_temp
                a = torch.softmax(sc.masked_fill(~cmask, float("-inf")), dim=-1)
                q = q + torch.bmm(a.unsqueeze(1), Cwells).squeeze(1)
                
            query = F.normalize(q, dim=-1)
            tgt = rep_ids[:, t]
            
            if sampled:
                pos = (query * A[tgt]).sum(-1, keepdim=True) / self.temp
                neg_ids = torch.randint(1, self.n_words, (self.neg_samples,), device=rep_ids.device)
                neg = (query @ A[neg_ids].t()) / self.temp
                cand = torch.cat([pos, neg], dim=1)
                ll = F.cross_entropy(cand, torch.zeros(cand.size(0), dtype=torch.long, device=rep_ids.device), reduction="none")
            else:
                ll = F.cross_entropy((query @ A.t()) / self.temp, tgt, reduction="none")
                
            mask = (tgt != PAD).float()
            if self.training and self.word_w is not None:
                mask = mask * self.word_w[tgt]
                
            tok_nll += ll * mask
            tok_cnt += mask
            
            walk = tgt
            if self.training and self.sample_p > 0:
                coin = (torch.rand(B, device=rep_ids.device) < self.sample_p) & (tgt != PAD)
                if bool(coin.any()):
                    with torch.no_grad():
                        lg = query[coin] @ A.t()
                        lg[:, PAD] = float("-inf")
                        walk = tgt.clone()
                        walk[coin] = lg.argmax(-1)
                        
            h = self.collapse_step(h, A[walk])
            
        if reduce_mean:
            loss = (tok_nll / tok_cnt.clamp(min=1)).mean()
            assert_tensor("reply_nll.loss", loss)
            return loss
        return tok_nll, tok_cnt

    @torch.no_grad()
    def generate(self, msg_ids, max_len, unk=None, ban=(),
                 rep_penalty=1.0, no_repeat_bigram=False, temperature=0.0):
        """Free-running decode using pure geometric collapse."""
        assert_tensor("generate.msg_ids", msg_ids, ndim=2, dtype=torch.long)
        assert isinstance(max_len, int) and max_len > 0, "max_len must be a positive int"
        assert rep_penalty > 0, "rep_penalty must be positive"
        assert temperature >= 0, "temperature cannot be negative"
        A = F.normalize(self.word_anchors, dim=-1)
        
        traj, tmask, hread = self.read_context(msg_ids, A)
        z = hread
        h = z
        
        if self.align:
            Cwells = self.lookup_wells(msg_ids, A)
            cmask = tmask
            
        B = msg_ids.size(0)
        toks = []
        done = torch.zeros(B, dtype=torch.bool, device=msg_ids.device)
        prev = None
        seen_bigrams = [set() for _ in range(B)]
        ar = torch.arange(B, device=msg_ids.device)
        
        for t in range(max_len):
            hn = F.normalize(h, dim=-1)
            q = hn + z
            
            if self.pos_well:
                q = q + self.pos_anchor[min(t, self.pos_anchor.size(0) - 1)]
                
            if self.align:
                probe = F.normalize(hn + z, dim=-1).unsqueeze(1)
                sc = torch.bmm(probe, Cwells.transpose(1, 2)).squeeze(1) / self.align_temp
                a = torch.softmax(sc.masked_fill(~cmask, float("-inf")), dim=-1)
                q = q + torch.bmm(a.unsqueeze(1), Cwells).squeeze(1)
                
            query = F.normalize(q, dim=-1)
            logits = (query @ A.t()) / self.temp
            
            if rep_penalty != 1.0 and toks:
                hist = torch.stack(toks, dim=1)
                g = torch.gather(logits, 1, hist)
                g = torch.where(g > 0, g / rep_penalty, g * rep_penalty)
                logits.scatter_(1, hist, g)
                
            logits[:, PAD] = float("-inf")
            if unk is not None:
                logits[:, unk] = float("-inf")
            for b in ban:
                logits[:, b] = float("-inf")
            if prev is not None:
                logits[ar, prev] = float("-inf")
            if no_repeat_bigram and prev is not None:
                for b in range(B):
                    for (p, y) in seen_bigrams[b]:
                        if p == int(prev[b]):
                            logits[b, y] = float("-inf")
                            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
                
            if prev is not None:
                for b in range(B):
                    seen_bigrams[b].add((int(prev[b]), int(nxt[b])))
                    
            prev = nxt
            toks.append(nxt)
            h = self.collapse_step(h, A[nxt])
            done = done | (nxt == self.eos)
            if bool(done.all()):
                break
                
        T = torch.stack(toks, dim=1)
        return T, None


# ---------------------------------------------------------------- data loader

def load_wells_from_typer(device, path=DEFAULT_TYPER_CKPT):
    assert os.path.exists(path), f"Word wells checkpoint path not found: {path}"
    ck = torch.load(path, map_location="cpu")
    for key in ("word_anchors", "stoi", "itos", "unk", "eos", "config"):
        assert key in ck, f"typer checkpoint missing key {key!r}"
    warm = assert_tensor("typer.word_anchors", ck["word_anchors"], ndim=2, finite=True)
    stoi, itos = coerce_token_maps(ck["stoi"], ck["itos"])
    n_words, dim = int(ck["config"]["n_words"]), int(ck["config"]["dim"])
    assert warm.shape == (n_words, dim), (
        f"typer word_anchors shape {tuple(warm.shape)} does not match config {(n_words, dim)}"
    )
    unk, eos = int(ck["unk"]), int(ck["eos"])
    for tok in SPECIALS:
        assert tok not in stoi, f"typer checkpoint already contains duplicate special token {tok!r}"
        stoi[tok] = n_words
        itos[n_words] = tok
        warm = torch.cat([warm, torch.randn(1, dim) / dim ** 0.5], dim=0)
        n_words += 1
    extras = {"start": ck.get("start"), "strength": ck.get("strength")}
    if extras["start"] is not None:
        assert_tensor("typer.start", extras["start"], ndim=1, shape=(dim,), finite=True)
    assert_token_maps(stoi, itos, n_words=n_words, unk=unk, eos=eos)
    return warm, stoi, itos, unk, eos, n_words, dim, extras


# ---------------------------------------------------------------- chat loop

def chat_loop(args, device):
    ck = torch.load(args.ckpt, map_location=device)
    for key in ("config", "state_dict", "stoi", "itos", "unk", "eos"):
        assert key in ck, f"chat checkpoint missing key {key!r}"
    cfg = ck["config"]
    stoi, itos = coerce_token_maps(ck["stoi"], ck["itos"])
    unk, eos = int(ck["unk"]), int(ck["eos"])
    ctx_words = int(cfg.get("ctx_words", CTX_WORDS))
    reply_words = int(cfg.get("reply_words", MAXLEN - 2))
    max_reply_len = int(cfg.get("max_reply_len", reply_words + 2))
    assert_token_maps(stoi, itos, n_words=int(cfg["n_words"]), unk=unk, eos=eos)
    
    model = PureReplyBrain(cfg["n_words"], cfg["dim"], eos,
                           pos_well=cfg.get("pos_well", True),
                           align=cfg.get("align", True),
                           max_reply_len=max_reply_len).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)
    
    if ck.get("fast_alpha") is not None:
        model.fast_alpha = ck["fast_alpha"].to(device)
    model.eval()
    
    ban = [stoi[t] for t in SPECIALS if t in stoi]
    oov_src = args.oov_words or DEFAULT_SEMANTIC_INIT
    if os.path.exists(oov_src):
        minter = WordMinter(cfg["dim"], cfg["n_words"], device, oov_src)
    else:
        minter = CharMinter(cfg["dim"], cfg["n_words"], device)
        
    print(f"loaded {args.ckpt}   ctx {ctx_words} words   device {device}")
    print("Pure Collapse Chat: :reset to wipe, :q to quit\n")
    
    from prep_chat_context import clean as clean_text
    history = []
    
    while True:
        try:
            line = input("you   > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line == ":q":
            break
        if line == ":reset":
            history = []; print("[history wiped]"); continue
            
        clean_line = clean_text(line)
        history.append(f"<you> {clean_line}")
        ctx_str = " ".join(history)
        
        ids = encode_context([ctx_str], stoi, unk, eos, ctx_words, minter=minter).to(device)
        model.oov_wells = minter.table()
        assert_id_tensor("chat.ids", ids, upper=model.n_words + model.oov_wells.size(0))
        
        reply_ids, _ = model.generate(ids, max_len=reply_words, unk=unk, ban=ban,
                                      rep_penalty=args.rep_penalty,
                                      no_repeat_bigram=args.no_repeat_bigram,
                                      temperature=args.temperature)
        
        reply = decode(reply_ids[0], itos, eos)
        print(f"ai    > {reply}")
        history.append(f"<me> {reply}")
        
        # limit history size
        if len(history) > 20:
            history = history[-20:]


# ---------------------------------------------------------------- training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--typer-ckpt", default=DEFAULT_TYPER_CKPT)
    ap.add_argument("--fast-ckpt", default=DEFAULT_FAST_CKPT)
    ap.add_argument("--ckpt", default=CKPT_OUT)
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--ctx-words", type=int, default=CTX_WORDS,
                    help="context token budget; 0 = keep every token in prepared context")
    ap.add_argument("--reply-words", type=int, default=MAXLEN - 2,
                    help="reply token budget; 0 = keep every token in prepared reply")
    
    # model options
    ap.add_argument("--no-pos-well", action="store_true")
    ap.add_argument("--no-align", action="store_true")
    
    # training params
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--no-bucket", action="store_true",
                    help="disable length bucketing; useful only for fixed-length/capped data")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--semantic-init", default=DEFAULT_SEMANTIC_INIT)
    ap.add_argument("--oov-words", default="")
    ap.add_argument("--neg-samples", type=int, default=512)
    ap.add_argument("--meaning-w", type=float, default=1.0)
    ap.add_argument("--fast-reader", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    
    # decode options
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rep-penalty", type=float, default=1.2)
    ap.add_argument("--no-repeat-bigram", action="store_true")
    
    args = ap.parse_args()
    device = torch.device(args.device)
    assert args.epochs > 0, "--epochs must be positive"
    assert args.lr > 0, "--lr must be positive"
    assert args.batch_size > 0, "--batch-size must be positive"
    assert args.min_freq > 0, "--min-freq must be positive"
    assert args.neg_samples >= 0, "--neg-samples cannot be negative"
    assert args.meaning_w >= 0, "--meaning-w cannot be negative"
    assert args.patience > 0, "--patience must be positive"
    assert args.ctx_words >= 0, "--ctx-words cannot be negative"
    assert args.reply_words >= 0, "--reply-words cannot be negative"
    if device.type == "mps":
        assert torch.backends.mps.is_available(), "MPS requested but torch MPS is not available"
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but torch CUDA is not available"
    
    if args.chat:
        assert os.path.exists(args.ckpt), f"Chat checkpoint does not exist: {args.ckpt}"
        chat_loop(args, device)
        return
        
    print(f"starting pure collapse chat reply training pipeline on {device}...")
    
    assert os.path.exists(args.data), f"Training data path does not exist: {args.data}"
    # 1. Load context pairs
    pairs = read_pairs(args.data, args.max_lines)
    assert len(pairs) > 0, "Loaded dataset is empty. Cannot train on zero data."
    print(f"loaded {len(pairs):,} pairs")
    
    random.seed(SEED); random.shuffle(pairs)
    n_dev = max(10, int(len(pairs) * 0.05))
    assert len(pairs) > n_dev, f"Dataset size {len(pairs)} is too small for validation split size {n_dev}."
    train_pairs, dev_pairs = pairs[:-n_dev], pairs[-n_dev:]
    assert len(train_pairs) > 0 and len(dev_pairs) > 0, "train/dev split produced an empty side"
    ctx_budget = args.ctx_words if args.ctx_words > 0 else max(len(m.split()) for m, _ in pairs)
    reply_budget = args.reply_words if args.reply_words > 0 else max(len(r.split()) for _, r in pairs)
    max_reply_len = reply_budget + 2
    assert ctx_budget > 0, "computed context budget must be positive"
    assert reply_budget > 0, "computed reply budget must be positive"
    print(f"sequence budgets: ctx {ctx_budget} words | reply {reply_budget} words")
    
    # 2. Setup Vocabulary & Wells from Typer
    assert os.path.exists(args.typer_ckpt), f"Sentence typer checkpoint does not exist: {args.typer_ckpt}"
    warm, stoi, itos, unk, eos, n_words, dim, extras = load_wells_from_typer(device, args.typer_ckpt)
    assert_tensor("initial warm wells", warm, ndim=2, shape=(n_words, dim), finite=True)
    assert_token_maps(stoi, itos, n_words=n_words, unk=unk, eos=eos)
    
    # Shrink vocabulary to keep negative sampling fast and dense
    warm, stoi, itos, unk, eos, n_words, minted = shrink_vocab_fast(
        train_pairs, warm, stoi, itos, unk, eos, args.min_freq
    )
    assert_tensor("shrunk warm wells", warm, ndim=2, shape=(n_words, dim), finite=True)
    assert_token_maps(stoi, itos, n_words=n_words, unk=unk, eos=eos)
    print(f"vocab cut: {len(warm):,} wells (min-freq {args.min_freq}; {minted} fresh letter-minted wells)")
    
    # Initialize shared model architecture
    model = PureReplyBrain(n_words, dim, eos, warm=warm,
                           pos_well=not args.no_pos_well,
                           align=not args.no_align,
                           max_reply_len=max_reply_len).to(device)
                           
    # Load start states and strength priors
    if extras.get("start") is not None:
        assert_tensor("extras.start", extras["start"], ndim=1, shape=(dim,), finite=True)
        with torch.no_grad():
            model.start.copy_(extras["start"].to(model.start.device))
            
    # Load checkpoint if resuming
    if args.resume:
        assert os.path.exists(args.resume), f"Resume checkpoint does not exist: {args.resume}"
        ck = torch.load(args.resume, map_location=device)
        for key in ("state_dict", "config"):
            assert key in ck, f"resume checkpoint missing key {key!r}"
        rcfg = ck["config"]
        assert int(rcfg["n_words"]) == n_words, (
            f"resume vocab size {rcfg['n_words']} does not match current vocab size {n_words}"
        )
        assert int(rcfg["dim"]) == dim, (
            f"resume dim {rcfg['dim']} does not match current dim {dim}"
        )
        resume_reply_len = int(rcfg.get("max_reply_len",
                                       int(rcfg.get("reply_words", MAXLEN - 2)) + 2))
        assert resume_reply_len == max_reply_len, (
            f"resume max_reply_len {resume_reply_len} does not match current {max_reply_len}"
        )
        model.load_state_dict(ck["state_dict"], strict=False)
        print(f"resumed checkpoint: {args.resume}")
    elif os.path.exists(args.semantic_init):
        semantic_init(model, stoi, args.semantic_init, device)
    assert_model_finite(model)
        
    # Read distilled causal reader taps
    if args.fast_reader and os.path.exists(args.fast_ckpt):
        fck = torch.load(args.fast_ckpt, map_location=device)
        assert "alpha" in fck, f"fast-reader checkpoint missing key 'alpha': {args.fast_ckpt}"
        model.fast_alpha = fck["alpha"].to(device)
        assert_tensor("fast_reader.alpha", model.fast_alpha, ndim=1, device=device, finite=True)
        print(f"loaded causal fast-reader: {args.fast_ckpt}")
        
    # Setup Minter for OOV context words
    oov_src = args.oov_words or args.semantic_init
    if oov_src and os.path.exists(oov_src):
        minter = WordMinter(dim, n_words, device, oov_src)
    else:
        minter = CharMinter(dim, n_words, device)

    # Prepare meaning weights
    train_rep = encode_reply([r for _, r in train_pairs], stoi, unk, eos, reply_budget)
    assert_id_tensor("train_rep", train_rep, upper=n_words)
    if args.meaning_w > 0:
        w, most_common = meaning_weights(train_rep, n_words, args.meaning_w)
        assert_tensor("meaning weights", w, ndim=1, shape=(n_words,), finite=True)
        assert 0 <= most_common < n_words, "most_common token id is outside vocab"
        model.word_w = w.to(device)
        print(f"meaning weights enabled: common word weight {w[most_common]:.2f}")
        
    # Set model config
    model.neg_samples = args.neg_samples
    
    # Prepare batch indexes
    train_msg = encode_context([m for m, _ in train_pairs], stoi, unk, eos, ctx_budget,
                               minter=minter)
    train_rep = train_rep.contiguous()
    
    dev_msg = encode_context([m for m, _ in dev_pairs], stoi, unk, eos, ctx_budget,
                             minter=minter)
    dev_rep = encode_reply([r for _, r in dev_pairs], stoi, unk, eos, reply_budget)
    assert_id_tensor("dev_rep", dev_rep, upper=n_words)
    
    if minter is not None:
        model.oov_wells = minter.table()
        assert_tensor("model.oov_wells", model.oov_wells, ndim=2, device=device, finite=True,
                      nonempty=False)
        assert model.oov_wells.size(1) == dim, "OOV well dim must match model dim"
        if isinstance(minter, WordMinter):
            print(f"oov wells: {minter.hits:,} real wells <- {oov_src} | {minter.miss:,} fallback")
        else:
            print(f"char wells: minted {len(minter.rows):,} read-wells for OOV context words")
    read_upper = n_words + (model.oov_wells.size(0) if model.oov_wells is not None else 0)
    assert_id_tensor("train_msg", train_msg, upper=read_upper)
    assert_id_tensor("dev_msg", dev_msg, upper=read_upper)
    assert train_msg.size(0) == train_rep.size(0) == len(train_pairs), "train tensor row count mismatch"
    assert dev_msg.size(0) == dev_rep.size(0) == len(dev_pairs), "dev tensor row count mismatch"
            
    print(f"train size: {train_msg.size(0)} | dev size: {dev_msg.size(0)}")
    train_lengths = (train_msg != PAD).sum(1) + (train_rep != PAD).sum(1)
    length_order = torch.argsort(train_lengths)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_nll = float("inf")
    patience_cnt = 0
    
    # Ensure checkpoint output folder exists
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    def trim_batch(msg, rep=None):
        lm = int((msg != PAD).sum(1).max().item()) if msg.numel() else 1
        msg = msg[:, :max(lm, 1)].to(device)
        if rep is None:
            return msg
        lr = int((rep != PAD).sum(1).max().item()) if rep.numel() else 1
        rep = rep[:, :max(lr, 1)].to(device)
        return msg, rep

    def evaluate_dev():
        model.eval()
        total_nll = 0.0
        total_count = 0.0
        with torch.no_grad():
            for j in range(0, dev_msg.size(0), args.batch_size):
                msg, rep = trim_batch(dev_msg[j:j + args.batch_size],
                                      dev_rep[j:j + args.batch_size])
                tok_nll, tok_cnt = model.reply_nll(msg, rep, reduce_mean=False)
                total_nll += tok_nll.sum().item()
                total_count += tok_cnt.sum().item()
        assert total_count > 0, "dev token count is zero"
        dev_value = total_nll / total_count
        assert math.isfinite(dev_value), "dev NLL is not finite"
        return dev_value

    def epoch_order():
        if args.no_bucket:
            return torch.randperm(train_msg.size(0))
        chunks = [length_order[i:i + args.batch_size]
                  for i in range(0, length_order.numel(), args.batch_size)]
        random.shuffle(chunks)
        return torch.cat(chunks)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Annealing schedules
        model.sample_p = min(0.25, 0.25 * (epoch / 10.0))
        
        perm = epoch_order()
        epoch_loss = 0.0
        n_batches = 0
        total_batches = (train_msg.size(0) + args.batch_size - 1) // args.batch_size
        t0 = time.time()
        hb_t = t0
        hb_batches = 0
        
        for i in range(0, train_msg.size(0), args.batch_size):
            idx = perm[i:i + args.batch_size]
            b_msg, b_rep = trim_batch(train_msg[idx], train_rep[idx])
            
            optimizer.zero_grad()
            loss = model.reply_nll(b_msg, b_rep)
            loss.backward()
            
            # Clip gradients to keep system stable
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            
            loss_value = loss.item()
            assert math.isfinite(loss_value), "training loss is not finite"
            epoch_loss += loss_value
            n_batches += 1
            if n_batches == 1 or n_batches % 25 == 0:
                now = time.time()
                steps_per_sec = (n_batches - hb_batches) / max(now - hb_t, 1e-9)
                print(f"  ep{epoch:02d} {n_batches}/{total_batches} | "
                      f"loss {epoch_loss/n_batches:.4f} | {steps_per_sec:.2f} batch/s",
                      flush=True)
                hb_t = now
                hb_batches = n_batches
            
        assert n_batches > 0, "epoch completed without any training batches"
        # Evaluation pass
        dev_nll = evaluate_dev()
        assert math.isfinite(epoch_loss), "epoch loss accumulator is not finite"
        assert_model_finite(model)
            
        dt = time.time() - t0
        print(f"ep {epoch:02d} | loss {epoch_loss/n_batches:.4f} | dev nll {dev_nll:.4f} | sample_p {model.sample_p:.2f} | {dt:.1f}s")
        
        # Print check examples
        if epoch % 5 == 0 or epoch == 1:
            print("\n  --- Dev Check Example ---")
            with torch.no_grad():
                demo_ids = trim_batch(dev_msg[:2])
                rep_ids, _ = model.generate(demo_ids, max_len=min(reply_budget, 128), unk=unk)
                for b in range(min(2, demo_ids.size(0))):
                    prompt = decode(demo_ids[b], itos, eos)
                    real = decode(dev_rep[b], itos, eos)
                    gen = decode(rep_ids[b], itos, eos)
                    print(f"   you   : {' '.join(prompt.split()[-15:])}")
                    print(f"   real  : {real}")
                    print(f"   ai    : {gen}")
            print()
            
        # Early stopping and checkpointing
        if dev_nll < best_nll:
            best_nll = dev_nll
            patience_cnt = 0
            
            # Save state checkpoint
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "config": {
                    "n_words": n_words,
                    "dim": dim,
                    "pos_well": model.pos_well,
                    "align": model.align,
                    "ctx_words": ctx_budget,
                    "reply_words": reply_budget,
                    "max_reply_len": max_reply_len
                },
                "stoi": dict(stoi),
                "itos": dict(itos),
                "unk": unk,
                "eos": eos,
                "best": {"epoch": epoch, "nll": dev_nll}
            }
            if model.fast_alpha is not None:
                state["fast_alpha"] = model.fast_alpha
            assert_checkpoint_payload(state)
            torch.save(state, args.ckpt)
            assert os.path.exists(args.ckpt), f"checkpoint was not created: {args.ckpt}"
            assert os.path.getsize(args.ckpt) > 0, f"checkpoint is empty: {args.ckpt}"
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"early stopping triggered (best dev nll {best_nll:.4f})")
                break
                
    print(f"best model = model saved to {args.ckpt}")


if __name__ == "__main__":
    main()
