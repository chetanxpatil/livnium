"""
char_collapse.py — Position-aware character-level collapse.

A word is represented two ways AT ONCE:

  * SPLIT view  — the ordered character process  c, h, e, t, a, n  run through a
                  position-aware collapse trajectory. The running state is pulled
                  toward each letter's anchor in turn; the endpoint is the word
                  vector. This is the *route* that builds the word.

  * FULL  view  — the whole word seen at once as an order-free pooled vector
                  (mean of the letter anchors). This is the *destination as a
                  bag* — it ignores order on purpose, so it is the natural foil
                  that proves the SPLIT view is doing the ordering work.

Both share the SAME attractor mechanics as VectorCollapseEngine
(see vector_collapse.py). The per-character update is:

    h <- h + delta(h + pos_i) - strength * (1 - cos(h, A_c)) * normalize(h - A_c)

where
    A_c    = unit-normalized anchor of character c   (the gravity well),
    pos_i  = a learned positional signal injected into the residual MLP so the
             dynamics are ORDER-SENSITIVE: 'chetan' and 'cntheta' trace
             different paths and land in different places.

Identical conventions to VectorCollapseEngine:
    - divergence law  div = 1 - cos        (always attractive, zero on the anchor)
    - residual update is zero-initialised   (starts as the identity, delta = 0)
    - state norm clamped to <= 10.0

A tiny position-conditioned readout decodes the SPLIT vector back into its
characters. If a single word vector can be unspelled at every position, it has
provably retained the *ordered* spelling — that is the position-awareness test.

The fused (SPLIT + FULL) word vector is a drop-in replacement for the frozen
nn.Embedding lookup in CollapseTextEncoder: any word — seen, unseen, or typo'd —
becomes a path through known letter anchors, so there is no out-of-vocabulary
hole.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- shared force law (identical to vector_collapse.py) -----------------------

def divergence_from_alignment(align_value: torch.Tensor) -> torch.Tensor:
    """div = 1 - cos: always attractive, vanishing exactly on the anchor."""
    return 1.0 - align_value


# --- character alphabet -------------------------------------------------------

class CharVocab:
    """Maps characters <-> ids. Index 0 is reserved for PAD."""

    PAD = "\x00"

    def __init__(self, chars: str = "abcdefghijklmnopqrstuvwxyz"):
        # PAD first so pad_idx == 0, mirroring the word-level encoder.
        self.itos: List[str] = [self.PAD] + list(chars)
        self.stoi: Dict[str, int] = {c: i for i, c in enumerate(self.itos)}
        self.pad_idx = 0

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, word: str, max_len: int) -> List[int]:
        ids = [self.stoi[c] for c in word.lower() if c in self.stoi][:max_len]
        ids += [self.pad_idx] * (max_len - len(ids))
        return ids

    def encode_batch(self, words: List[str], max_len: int) -> torch.Tensor:
        return torch.tensor([self.encode(w, max_len) for w in words], dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if i != self.pad_idx)


# --- the module ---------------------------------------------------------------

class CharCollapse(nn.Module):
    """Position-aware character-level collapse encoder + reconstructive decoder.

    encode(char_ids) -> (z_seq, z_bag, fused, path)
        z_seq  : (B, dim)  SPLIT view  — ordered, position-aware endpoint
        z_bag  : (B, dim)  FULL view   — order-free mean of letter anchors
        fused  : (B, dim)  the word vector to hand downstream (uses both views)
        path   : list of (B, dim) states, one per character step (for probing)

    decode(z_seq, length) -> (B, length, n_chars) character logits.
    """

    def __init__(
        self,
        dim: int = 256,
        max_len: int = 32,
        chars: str = "abcdefghijklmnopqrstuvwxyz",
        strength: float = 0.1,
    ):
        super().__init__()
        self.vocab = CharVocab(chars)
        self.dim = dim
        self.max_len = max_len
        self.pad_idx = self.vocab.pad_idx
        self.strength = strength

        n_chars = len(self.vocab)

        # One gravity well per character (PAD anchor exists but is never targeted).
        self.letter_anchors = nn.Parameter(torch.randn(n_chars, dim))

        # Learned position signal, injected into the residual update so the same
        # letter at different positions evolves the state differently.
        self.pos_embed = nn.Parameter(torch.randn(max_len, dim) * 0.02)

        # Learned start state (the empty-word origin of every trajectory).
        self.start = nn.Parameter(torch.randn(dim) * 0.02)

        # Residual update MLP — identical shape to VectorCollapseEngine.update,
        # zero-initialised on the last layer so the block starts as the identity
        # (delta = 0) and the thermodynamic force shapes the early dynamics.
        self.update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.update[2].weight)
        nn.init.zeros_(self.update[2].bias)

        # Fuse SPLIT + FULL into a single word vector for downstream use.
        self.fuse = nn.Linear(2 * dim, dim)

        # Position-conditioned readout: decode a character at each position from
        # the SPLIT vector. If this works, z_seq retained the *ordered* spelling.
        self.readout = nn.Linear(dim, n_chars)

    # -- encode ----------------------------------------------------------------

    def encode(
        self, char_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if char_ids.dim() == 1:
            char_ids = char_ids.unsqueeze(0)
        B, L = char_ids.shape
        device = char_ids.device

        anchors = F.normalize(self.letter_anchors, dim=-1)  # (n_chars, dim)
        mask = (char_ids != self.pad_idx)                   # (B, L) real chars

        # SPLIT view: sequential, position-aware collapse trajectory.
        h = self.start.to(device).expand(B, -1).contiguous()
        path: List[torch.Tensor] = []
        for i in range(L):
            c = char_ids[:, i]                # (B,)
            target = anchors[c]               # (B, dim) this letter's well
            pos = self.pos_embed[i]           # (dim,) position signal

            delta = self.update(h + pos)      # position-aware nonlinear update

            h_n = F.normalize(h, dim=-1)
            align = (h_n * target).sum(dim=-1)            # (B,)
            div = divergence_from_alignment(align)        # (B,)
            away = F.normalize(h - target, dim=-1)        # direction anchor -> h

            step = delta - self.strength * div.unsqueeze(-1) * away

            # Apply the step only at real (non-pad) positions, so padding is inert.
            m = mask[:, i].float().unsqueeze(-1)          # (B, 1)
            h = h + m * step

            # Clamp norm, exactly like VectorCollapseEngine.
            h_norm = h.norm(p=2, dim=-1, keepdim=True)
            h = torch.where(h_norm > 10.0, h * (10.0 / (h_norm + 1e-8)), h)

            path.append(h)

        z_seq = h  # endpoint of the route

        # FULL view: order-free mean of the letter anchors (bag of characters).
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        z_bag = (anchors[char_ids] * mask.unsqueeze(-1).float()).sum(dim=1) / denom

        fused = self.fuse(torch.cat([z_seq, z_bag], dim=-1))
        return z_seq, z_bag, fused, path

    # -- decode ----------------------------------------------------------------

    def decode(self, z_seq: torch.Tensor, length: int) -> torch.Tensor:
        """Reconstruct character logits at each position from the SPLIT vector."""
        if z_seq.dim() == 1:
            z_seq = z_seq.unsqueeze(0)
        logits = []
        for i in range(length):
            logits.append(self.readout(z_seq + self.pos_embed[i]))  # (B, n_chars)
        return torch.stack(logits, dim=1)  # (B, length, n_chars)

    # -- training loss ---------------------------------------------------------

    def reconstruction_loss(self, char_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoencoder objective: encode word -> z_seq -> decode characters.

        Returns (loss, per-token correctness mask over real chars).
        """
        if char_ids.dim() == 1:
            char_ids = char_ids.unsqueeze(0)
        L = char_ids.size(1)
        z_seq, _, _, _ = self.encode(char_ids)
        logits = self.decode(z_seq, L)  # (B, L, n_chars)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            char_ids.reshape(-1),
            ignore_index=self.pad_idx,
        )
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            real = char_ids != self.pad_idx
            correct = (pred == char_ids) & real
        return loss, correct.float()

    # -- convenience -----------------------------------------------------------

    def word_vectors(self, words: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode a list of words straight to fused word vectors (zero-OOV)."""
        ids = self.vocab.encode_batch(words, self.max_len)
        if device is not None:
            ids = ids.to(device)
        _, _, fused, _ = self.encode(ids)
        return fused

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        _, _, fused, _ = self.encode(char_ids)
        return fused
