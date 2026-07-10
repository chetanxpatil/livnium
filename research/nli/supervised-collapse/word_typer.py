"""
word_typer.py — a small model that, when it SEES a word, TYPES it letter by letter.

  word  ->  [one learned point]  ->  decoder unrolls it  ->  c, h, e, t, a, n, <end>

How it works:
  * every word is a single learned point      word_point  in R^dim
  * a tiny GRU decoder is seeded with that point and emits one character at a
    time, feeding its own previous letter back in, until it types <EOS>.

That is literal "typing": the output is produced sequentially, left to right,
and the model decides on its own when the word is finished.

Trained by teacher forcing (show it the word, ask it to type the next letter).
At test time it types unaided from just the word's point.

Small + fast: dim 64, ~800 words, runs on CPU in a few seconds. Needs torch.
"""

import random
import string

import torch
import torch.nn as nn

SEED = 0
DIM = 64
HID = 64
N_WORDS = 800
STEPS = 800
BATCH = 128
LR = 4e-3

# token table: 0=PAD, 1=BOS (start typing), 2=EOS (done), 3..28 = a..z
PAD, BOS, EOS = 0, 1, 2
LETTERS = string.ascii_lowercase
C2T = {c: i + 3 for i, c in enumerate(LETTERS)}
T2C = {i + 3: c for i, c in enumerate(LETTERS)}
N_TOK = len(LETTERS) + 3
MAXLEN = 10  # longest word + room for EOS


def make_word(rng):
    v, c = "aeiou", "bcdfghjklmnpqrstvwxyz"
    n = rng.randint(3, 9)
    return "".join(rng.choice(v if i % 2 else c) for i in range(n))


def encode_target(word):
    """decoder input = BOS + letters ; target = letters + EOS (padded)."""
    body = [C2T[ch] for ch in word]
    dec_in = [BOS] + body
    tgt = body + [EOS]
    dec_in += [PAD] * (MAXLEN - len(dec_in))
    tgt += [PAD] * (MAXLEN - len(tgt))
    return dec_in[:MAXLEN], tgt[:MAXLEN]


class WordTyper(nn.Module):
    def __init__(self, n_words, dim=DIM, hid=HID):
        super().__init__()
        self.word_point = nn.Embedding(n_words, dim)   # what the model "sees"
        nn.init.normal_(self.word_point.weight, std=0.1)
        self.char_embed = nn.Embedding(N_TOK, dim, padding_idx=PAD)
        self.seed = nn.Linear(dim, hid)                # word point -> GRU hidden
        self.gru = nn.GRU(dim, hid, batch_first=True)
        self.out = nn.Linear(hid, N_TOK)

    def forward(self, word_ids, dec_in):
        h0 = torch.tanh(self.seed(self.word_point(word_ids))).unsqueeze(0)  # (1,B,hid)
        x = self.char_embed(dec_in)                                         # (B,T,dim)
        y, _ = self.gru(x, h0)
        return self.out(y)                                                  # (B,T,N_TOK)

    @torch.no_grad()
    def type_word(self, word_id):
        """Greedy typing from just the word's point. Returns the typed string."""
        h = torch.tanh(self.seed(self.word_point(torch.tensor([word_id])))).unsqueeze(0)
        tok = torch.tensor([[BOS]])
        typed = []
        for _ in range(MAXLEN):
            y, h = self.gru(self.char_embed(tok), h)
            nxt = int(self.out(y[:, -1]).argmax(-1))
            if nxt == EOS:
                break
            typed.append(T2C.get(nxt, "?"))
            tok = torch.tensor([[nxt]])
        return "".join(typed)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    words, seen = [], set()
    while len(words) < N_WORDS:
        w = make_word(rng)
        if w not in seen:
            seen.add(w)
            words.append(w)

    dec_in = torch.tensor([encode_target(w)[0] for w in words])
    tgt = torch.tensor([encode_target(w)[1] for w in words])

    model = WordTyper(len(words))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)

    print(f"vocab words : {len(words)}   dim : {DIM}   tokens : {N_TOK} (a-z + PAD/BOS/EOS)\n")

    model.train()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(words), (BATCH,))
        logits = model(idx, dec_in[idx])
        loss = lossf(logits.reshape(-1, N_TOK), tgt[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step:4d}   loss {loss.item():.4f}")

    # ---- can it type each word correctly, unaided? ----
    model.eval()
    correct = sum(model.type_word(i) == words[i] for i in range(len(words)))
    print(f"\nexact-word typing accuracy : {correct}/{len(words)} = {correct/len(words)*100:.1f}%")

    print("\n--- watch it type (greedy, letter by letter) ---")
    for i in range(8):
        w = words[i]
        # show the keystroke sequence
        h = torch.tanh(model.seed(model.word_point(torch.tensor([i])))).unsqueeze(0)
        tok = torch.tensor([[BOS]])
        keys = []
        with torch.no_grad():
            for _ in range(MAXLEN):
                y, h = model.gru(model.char_embed(tok), h)
                nxt = int(model.out(y[:, -1]).argmax(-1))
                if nxt == EOS:
                    keys.append("<end>")
                    break
                keys.append(T2C.get(nxt, "?"))
                tok = torch.tensor([[nxt]])
        typed = "".join(k for k in keys if k != "<end>")
        mark = "OK " if typed == w else "XX "
        print(f"  {mark}target {w:>10s}  ->  {' '.join(keys)}")


if __name__ == "__main__":
    main()
