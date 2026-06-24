"""
train_snli_typer.py — train the letter-by-letter typer on ALL the words in SNLI.

Pipeline (generalizing typer + the attention fix):

    word's letters --(CharCollapse, position-aware)--> per-letter states
            |                                                  |
            +------------------ seed ----------------> GRU decoder with ATTENTION
                                                               |
                                                       types c,h,e,t,a,n,<end>

The decoder attends over the encoder's per-letter collapse states while typing,
so it can look back at the source instead of squeezing the whole spelling
through one fixed vector — this is the cure for the order-scramble failures.

The word list is the real SNLI vocabulary, read from the trained checkpoint
model_nli_v1/nli_epoch20.pt (data["vocab"]["idx2word"]). We keep purely
alphabetic words, split train/held-out, and measure typing accuracy on words
the typer never trained on.

Needs torch. CPU is fine; a few minutes.
"""

import random
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from char_collapse import CharCollapse

CKPT = "model_nli_v1/nli_epoch20.pt"
SEED = 0
DIM = 64
HID = 128
ATT = 64
MAXLEN = 18          # encoder & decoder cap; longer words are dropped (rare)
HELDOUT_FRAC = 0.1
STEPS = 4000
BATCH = 256
LR = 2e-3

PAD, BOS, EOS = 0, 1, 2
LETTERS = string.ascii_lowercase
C2T = {c: i + 3 for i, c in enumerate(LETTERS)}
T2C = {i + 3: c for i, c in enumerate(LETTERS)}
N_TOK = len(LETTERS) + 3
ALPHA = set(LETTERS)


def load_snli_words():
    try:
        data = torch.load(CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(CKPT, map_location="cpu")
    idx2word = data["vocab"]["idx2word"]
    words, seen = [], set()
    dropped_long = 0
    for w in idx2word:
        lw = w.lower()
        if not lw or any(c not in ALPHA for c in lw):
            continue  # skip <pad>, <unk>, punctuation, numbers, mixed tokens
        if len(lw) > MAXLEN:
            dropped_long += 1
            continue
        if lw not in seen:
            seen.add(lw)
            words.append(lw)
    return words, len(idx2word), dropped_long


def dec_target(word):
    body = [C2T[c] for c in word]
    dec_in = ([BOS] + body)[:MAXLEN]
    tgt = (body + [EOS])[:MAXLEN]
    dec_in += [PAD] * (MAXLEN - len(dec_in))
    tgt += [PAD] * (MAXLEN - len(tgt))
    return dec_in, tgt


class AttnTyper(nn.Module):
    def __init__(self, dim=DIM, hid=HID, att=ATT):
        super().__init__()
        self.enc = CharCollapse(dim=dim, max_len=MAXLEN)
        self.char_embed = nn.Embedding(N_TOK, dim, padding_idx=PAD)
        self.seed = nn.Linear(dim, hid)
        self.cell = nn.GRUCell(dim + dim, hid)        # input = [prev letter ; context]
        # additive attention
        self.W_h = nn.Linear(hid, att)
        self.W_e = nn.Linear(dim, att)
        self.v = nn.Linear(att, 1)
        self.out = nn.Linear(hid + dim, N_TOK)

    def encode_mem(self, enc_ids):
        _, _, fused, path = self.enc.encode(enc_ids)
        states = torch.stack(path, dim=1)             # (B, L, dim)
        mask = (enc_ids != self.enc.pad_idx)          # (B, L)
        return states, fused, mask

    def attend(self, h, states, mask):
        scores = self.v(torch.tanh(self.W_h(h).unsqueeze(1) + self.W_e(states))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(scores, dim=-1)             # (B, L)
        ctx = (alpha.unsqueeze(-1) * states).sum(1)   # (B, dim)
        return ctx

    def forward(self, enc_ids, dec_in):
        states, fused, mask = self.encode_mem(enc_ids)
        h = torch.tanh(self.seed(fused))
        logits = []
        for t in range(dec_in.size(1)):
            ctx = self.attend(h, states, mask)
            x = torch.cat([self.char_embed(dec_in[:, t]), ctx], dim=-1)
            h = self.cell(x, h)
            logits.append(self.out(torch.cat([h, ctx], dim=-1)))
        return torch.stack(logits, dim=1)             # (B, T, N_TOK)

    @torch.no_grad()
    def type_word(self, enc_ids_single):
        states, fused, mask = self.encode_mem(enc_ids_single)
        h = torch.tanh(self.seed(fused))
        tok = torch.tensor([BOS])
        typed = []
        for _ in range(MAXLEN):
            ctx = self.attend(h, states, mask)
            x = torch.cat([self.char_embed(tok), ctx], dim=-1)
            h = self.cell(x, h)
            nxt = int(self.out(torch.cat([h, ctx], dim=-1)).argmax(-1))
            if nxt == EOS:
                break
            typed.append(T2C.get(nxt, "?"))
            tok = torch.tensor([nxt])
        return "".join(typed)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    words, total_vocab, dropped = load_snli_words()
    random.shuffle(words)
    n_test = max(1, int(len(words) * HELDOUT_FRAC))
    test_words = words[:n_test]
    train_words = words[n_test:]

    print(f"SNLI vocab tokens         : {total_vocab}")
    print(f"kept alphabetic words     : {len(words)}  (dropped {dropped} over {MAXLEN} chars)")
    print(f"train / held-out          : {len(train_words)} / {len(test_words)}\n")

    model = AttnTyper()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)

    enc_tr = model.enc.vocab.encode_batch(train_words, MAXLEN)
    din_tr = torch.tensor([dec_target(w)[0] for w in train_words])
    tgt_tr = torch.tensor([dec_target(w)[1] for w in train_words])

    model.train()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(train_words), (BATCH,))
        logits = model(enc_tr[idx], din_tr[idx])
        loss = lossf(logits.reshape(-1, N_TOK), tgt_tr[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == 1:
            print(f"step {step:4d}   loss {loss.item():.4f}")

    model.eval()

    def typing_acc(word_list, cap=2000):
        sample = word_list[:cap]
        ids = model.enc.vocab.encode_batch(sample, MAXLEN)
        ok = sum(model.type_word(ids[k:k + 1]) == sample[k] for k in range(len(sample)))
        return ok / len(sample), len(sample)

    tr_acc, n_tr = typing_acc(train_words)
    te_acc, n_te = typing_acc(test_words)
    print(f"\ntrained-word typing accuracy : {tr_acc*100:5.1f}%  (n={n_tr})")
    print(f"HELD-OUT typing accuracy     : {te_acc*100:5.1f}%  (n={n_te})  <-- generalization")

    print("\n--- typing SNLI words it NEVER trained on ---")
    ids = model.enc.vocab.encode_batch(test_words[:15], MAXLEN)
    for k, w in enumerate(test_words[:15]):
        typed = model.type_word(ids[k:k + 1])
        mark = "OK " if typed == w else "XX "
        print(f"  {mark}saw {w:>15s}  ->  {' '.join(typed)}")

    # Save the trained CharCollapse encoder so stage 2 (the NLI meaning head)
    # can load it FROZEN and use it to build word vectors. Spelling structure
    # learned here becomes the substrate that meaning gets attached to later.
    torch.save(
        {
            "char_collapse": model.enc.state_dict(),
            "config": {"dim": DIM, "max_len": MAXLEN},
        },
        "char_typer.pt",
    )
    print("\nsaved trained typer -> char_typer.pt  (use it in stage 2)")


if __name__ == "__main__":
    main()
