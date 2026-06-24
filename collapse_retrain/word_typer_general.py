"""
word_typer_general.py — type words it has NEVER SEEN.

Same typer as word_typer.py, but the seed is no longer a free per-word point.
Instead the word's seed is BUILT from its characters by the position-aware
CharCollapse encoder:

    word's letters --(CharCollapse)--> word vector --(GRU)--> type c,h,e,t,a,n,<end>

Because the seed is composed from shared letter anchors, a word the model never
trained on still produces a valid seed, so it can be typed. We train on one set
of words and measure typing accuracy on a DISJOINT held-out set.

This is the read-then-write ("copy") task done geometry-native on the encode
side. Needs torch. Runs on CPU in ~1 minute.
"""

import random
import string

import torch
import torch.nn as nn

from char_collapse import CharCollapse

SEED = 0
DIM = 64
HID = 128
ENC_MAXLEN = 12
N_TRAIN = 1600
N_TEST = 400
STEPS = 1500
BATCH = 128
LR = 3e-3

# decoder token table: 0=PAD, 1=BOS, 2=EOS, 3..28 = a..z
PAD, BOS, EOS = 0, 1, 2
LETTERS = string.ascii_lowercase
C2T = {c: i + 3 for i, c in enumerate(LETTERS)}
T2C = {i + 3: c for i, c in enumerate(LETTERS)}
N_TOK = len(LETTERS) + 3
DEC_MAXLEN = 12


def make_word(rng):
    v, c = "aeiou", "bcdfghjklmnpqrstvwxyz"
    n = rng.randint(3, 9)
    return "".join(rng.choice(v if i % 2 else c) for i in range(n))


def dec_target(word):
    body = [C2T[ch] for ch in word]
    dec_in = ([BOS] + body)[:DEC_MAXLEN]
    tgt = (body + [EOS])[:DEC_MAXLEN]
    dec_in += [PAD] * (DEC_MAXLEN - len(dec_in))
    tgt += [PAD] * (DEC_MAXLEN - len(tgt))
    return dec_in, tgt


class TyperGen(nn.Module):
    def __init__(self, dim=DIM, hid=HID):
        super().__init__()
        self.enc = CharCollapse(dim=dim, max_len=ENC_MAXLEN)   # letters -> word vector
        self.char_embed = nn.Embedding(N_TOK, dim, padding_idx=PAD)
        self.seed = nn.Linear(dim, hid)
        self.gru = nn.GRU(dim, hid, batch_first=True)
        self.out = nn.Linear(hid, N_TOK)

    def word_vec(self, enc_ids):
        _, _, fused, _ = self.enc.encode(enc_ids)
        return fused

    def forward(self, enc_ids, dec_in):
        h0 = torch.tanh(self.seed(self.word_vec(enc_ids))).unsqueeze(0)
        y, _ = self.gru(self.char_embed(dec_in), h0)
        return self.out(y)

    @torch.no_grad()
    def type_word(self, enc_ids_single):
        h = torch.tanh(self.seed(self.word_vec(enc_ids_single))).unsqueeze(0)
        tok = torch.tensor([[BOS]])
        typed = []
        for _ in range(DEC_MAXLEN):
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
    while len(words) < N_TRAIN + N_TEST:
        w = make_word(rng)
        if w not in seen:
            seen.add(w)
            words.append(w)
    train_words, test_words = words[:N_TRAIN], words[N_TRAIN:]

    model = TyperGen()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)

    enc_ids_tr = model.enc.vocab.encode_batch(train_words, ENC_MAXLEN)
    dec_in_tr = torch.tensor([dec_target(w)[0] for w in train_words])
    tgt_tr = torch.tensor([dec_target(w)[1] for w in train_words])

    print(f"train / test words : {N_TRAIN} / {N_TEST}  (DISJOINT)")
    print(f"dim {DIM}  hid {HID}\n")

    model.train()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, N_TRAIN, (BATCH,))
        logits = model(enc_ids_tr[idx], dec_in_tr[idx])
        loss = lossf(logits.reshape(-1, N_TOK), tgt_tr[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 150 == 0 or step == 1:
            print(f"step {step:4d}   loss {loss.item():.4f}")

    model.eval()

    def typing_acc(word_list):
        ids = model.enc.vocab.encode_batch(word_list, ENC_MAXLEN)
        correct = 0
        for k, w in enumerate(word_list):
            if model.type_word(ids[k:k + 1]) == w:
                correct += 1
        return correct / len(word_list)

    tr = typing_acc(train_words[:N_TEST])
    te = typing_acc(test_words)
    print(f"\ntrained-word typing accuracy : {tr*100:5.1f}%")
    print(f"UNSEEN-word typing accuracy  : {te*100:5.1f}%   <-- generalization")

    print("\n--- watch it type words it NEVER trained on ---")
    ids = model.enc.vocab.encode_batch(test_words[:10], ENC_MAXLEN)
    for k, w in enumerate(test_words[:10]):
        typed = model.type_word(ids[k:k + 1])
        mark = "OK " if typed == w else "XX "
        print(f"  {mark}saw {w:>10s}  ->  {' '.join(typed)}")


if __name__ == "__main__":
    main()
