"""
word_to_char.py — a small model whose only job is to learn, for each word,
WHICH LETTERS IT CONTAINS.

Setup (Livnium-native):
  * every word is a single learned point      word_vec  in R^dim
  * every letter a..z is a learned anchor      A_c       in R^dim
  * "word contains letter c"  <=>  word_vec aligns with anchor A_c

So the model is just:   score(word, c) = cos(word_vec, A_c) * temp
trained with binary cross-entropy against the true letter-presence of the word.
After training, the 26 cosines of a word's point tell you its letter set.

This is the WORD -> CHAR direction: input is the whole word (as one point),
output is its characters. It learns the letter *content* of the words in its
vocabulary (it memorises content into the point — that is the task; it is not
meant to generalise to unseen words, which is the char-encoder's job).

Runs on CPU in a few seconds. Needs torch.
"""

import random
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
DIM = 32          # small model
N_WORDS = 800
STEPS = 600
BATCH = 128
LR = 5e-3
TEMP = 10.0       # sharpens the cosine into a decisive yes/no

LETTERS = string.ascii_lowercase          # a..z
L2I = {c: i for i, c in enumerate(LETTERS)}


def make_word(rng):
    v, c = "aeiou", "bcdfghjklmnpqrstvwxyz"
    n = rng.randint(3, 9)
    return "".join(rng.choice(v if i % 2 else c) for i in range(n))


def letter_multihot(word):
    y = torch.zeros(26)
    for ch in word:
        y[L2I[ch]] = 1.0
    return y


class WordToChar(nn.Module):
    """Word point + letter anchors. Presence = cosine alignment."""

    def __init__(self, vocab_size, dim=DIM):
        super().__init__()
        self.word_vec = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.word_vec.weight, std=0.1)
        self.letter_anchors = nn.Parameter(torch.randn(26, dim))
        self.temp = TEMP

    def forward(self, word_ids):
        w = F.normalize(self.word_vec(word_ids), dim=-1)      # (B, dim)
        a = F.normalize(self.letter_anchors, dim=-1)          # (26, dim)
        return (w @ a.t()) * self.temp                        # (B, 26) logits


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    # build a vocabulary of unique words + their letter-presence targets
    words, seen = [], set()
    while len(words) < N_WORDS:
        w = make_word(rng)
        if w not in seen:
            seen.add(w)
            words.append(w)
    targets = torch.stack([letter_multihot(w) for w in words])  # (N, 26)

    model = WordToChar(len(words))
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"vocab words : {len(words)}    dim : {DIM}    letters : 26\n")

    model.train()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(words), (BATCH,))
        logits = model(idx)
        loss = F.binary_cross_entropy_with_logits(logits, targets[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:4d}   loss {loss.item():.4f}")

    # ---- evaluation: did each word learn its own letter set? ----
    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(len(words))
        pred = (torch.sigmoid(model(all_ids)) > 0.5).float()   # (N, 26)
        tgt = targets

        per_letter_acc = (pred == tgt).float().mean().item()
        exact_set = (pred == tgt).all(dim=1).float().mean().item()
        tp = (pred * tgt).sum().item()
        precision = tp / max(1.0, pred.sum().item())
        recall = tp / max(1.0, tgt.sum().item())
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

    print("\n--- Does the model know which letters each word contains? ---")
    print(f"  per-letter accuracy : {per_letter_acc*100:5.1f}%")
    print(f"  exact letter-set    : {exact_set*100:5.1f}%   (whole set correct)")
    print(f"  precision / recall  : {precision*100:5.1f}% / {recall*100:5.1f}%   F1 {f1*100:5.1f}%")

    print("\n--- examples (word -> letters the model says it contains) ---")
    with torch.no_grad():
        for w in words[:8]:
            wid = torch.tensor([words.index(w)])
            p = (torch.sigmoid(model(wid))[0] > 0.5)
            got = "".join(LETTERS[i] for i in range(26) if p[i])
            true = "".join(sorted(set(w)))
            mark = "OK " if got == true else "XX "
            print(f"  {mark}{w:>10s}  true:{true:<12s} pred:{got}")


if __name__ == "__main__":
    main()
