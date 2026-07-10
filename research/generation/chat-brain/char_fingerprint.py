"""
char_fingerprint.py — give ANY word a well from its SPELLING, standalone.

This is the char->word bridge, rebuilt so it needs NOTHING trained (the old
model/char_typer_pure.pt is gone and we don't want that dependency). It exists
for one job: when the word typer meets a word it has never seen, mint a stable,
distinct well for it FROM ITS LETTERS, on the fly — no retraining.

Design (kept deliberately tiny):
  * 26 letter anchors, drawn ONCE from a fixed seed -> deterministic. The same
    letter always has the same anchor, so the same word always gets the same
    fingerprint (across runs, across sessions).
  * Order-aware "roll binding" (VSA-style): letter i is cyclically rotated by
    i*stride dims before summing, so POSITION is a near-orthogonal role and
    anagrams ('listen' vs 'silent') get DIFFERENT vectors instead of colliding.
  * Result is L2-normalized -> lives on the same unit sphere the trained wells
    are compared on (decode is cosine), so a minted well drops straight in.

Why this is safe next to the trained wells: in 256-d, an independent
fingerprint direction is near-orthogonal to the ~20k trained wells, so a new
word gets its own corner and can be typed back without stepping on known words.
What it does NOT do: place the new word near its MEANING-neighbours — spelling
can't know usage. That needs training. This only guarantees "typeable + stable".
"""

import torch
import torch.nn.functional as F

LETTERS = "abcdefghijklmnopqrstuvwxyz"
MAX_WORD = 18                     # letters used per word (rest ignored)
ANCHOR_SEED = 0                   # fixed -> deterministic letter geometry


def letter_anchors(dim, seed=ANCHOR_SEED, device="cpu"):
    """26 unit letter anchors, fixed by seed. row i == LETTERS[i]."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(len(LETTERS), dim, generator=g)
    return F.normalize(A, dim=-1).to(device)


def char_fingerprint(word, A, dim=None):
    """Order-aware spelling vector for one word (unit length).

    A = letter_anchors(dim). Non-letters are dropped; empty/garbage words fall
    back to a fixed deterministic vector so they still get a stable well.
    """
    dim = dim or A.size(1)
    stride = max(1, dim // MAX_WORD)
    ids = [LETTERS.index(c) for c in word.lower() if c in LETTERS][:MAX_WORD]
    if not ids:
        # deterministic hash: python's hash() is salted per process, which
        # would give "75" a different well every session. crc32 never moves.
        import zlib
        g = torch.Generator().manual_seed(zlib.crc32(word.encode("utf-8")))
        return F.normalize(torch.randn(dim, generator=g), dim=-1).to(A.device)
    v = torch.zeros(dim, device=A.device)
    for i, ci in enumerate(ids):
        v = v + torch.roll(A[ci], shifts=i * stride, dims=-1)
    return F.normalize(v, dim=-1)


def fingerprint_batch(words, A, dim=None):
    return torch.stack([char_fingerprint(w, A, dim) for w in words], dim=0)


if __name__ == "__main__":
    # sanity: determinism + anagram separation + orthogonality to random wells
    dim = 256
    A = letter_anchors(dim)
    a1 = char_fingerprint("collapse", A)
    a2 = char_fingerprint("collapse", A)
    print(f"determinism  cos(collapse, collapse again) = {F.cosine_similarity(a1, a2, dim=0):.3f}  (want 1.000)")
    for w1, w2 in [("listen", "silent"), ("stop", "spot"), ("premise", "promise")]:
        c = F.cosine_similarity(char_fingerprint(w1, A), char_fingerprint(w2, A), dim=0)
        print(f"anagram/near cos({w1}, {w2}) = {c:.3f}  (want < 1, they separate)")
    W = F.normalize(torch.randn(20000, dim), dim=-1)
    fp = char_fingerprint("supercalifragilistic", A)
    print(f"max cos(new word, 20k random wells) = {(W @ fp).max():.3f}  (want low -> own corner)")
