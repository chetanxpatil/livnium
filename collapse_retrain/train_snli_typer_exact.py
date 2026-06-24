"""
train_snli_typer_exact.py — push CharCollapse typing toward 100% correct.

Typing a word you can see is a COPY task. The earlier version scrambled order on
long words because the decoder only had CharCollapse's *cumulative* per-letter
states to attend to, which blur together. Three targeted fixes:

  1. CLEAN per-letter memory: each source slot is exposed as
        [ collapse-state_i ; letter-anchor_i ; position_i ]
     so every input letter is uniquely identifiable (not just the running h).
  2. MONOTONIC attention: a learnable bias  -softplus(alpha)*(i - t)^2  pins the
     attention of output step t onto source letter i≈t. Copy becomes near-exact.
  3. More capacity (dim 128 / hid 256) + longer training.

CharCollapse still does the encoding — the geometry produces the per-letter
memory the decoder copies from. Trained on ALL alphabetic SNLI words (the goal is
"type every SNLI word correctly"), with a small held-out slice reported too.

Needs torch. Use MPS/GPU if available; ~10-20 min on Apple Silicon.
"""

import random
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from char_collapse import CharCollapse

NLI_CKPT = "model_nli_v1/nli_epoch20.pt"
SEED = 0
DIM = 128
HID = 256
ATT = 128
MAXLEN = 18
STEPS = 8000
BATCH = 256
LR = 2e-3
HELDOUT = 1500          # reported for generalization; rest is trained

PAD, BOS, EOS = 0, 1, 2
LETTERS = string.ascii_lowercase
C2T = {c: i + 3 for i, c in enumerate(LETTERS)}
T2C = {i + 3: c for i, c in enumerate(LETTERS)}
N_TOK = len(LETTERS) + 3
ALPHA = set(LETTERS)


def load_snli_words():
    try:
        data = torch.load(NLI_CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(NLI_CKPT, map_location="cpu")
    words, seen = [], set()
    for w in data["vocab"]["idx2word"]:
        lw = w.lower()
        if lw and all(c in ALPHA for c in lw) and len(lw) <= MAXLEN and lw not in seen:
            seen.add(lw)
            words.append(lw)
    return words


def dec_target(word):
    body = [C2T[c] for c in word]
    dec_in = ([BOS] + body)[:MAXLEN]
    tgt = (body + [EOS])[:MAXLEN]
    dec_in += [PAD] * (MAXLEN - len(dec_in))
    tgt += [PAD] * (MAXLEN - len(tgt))
    return dec_in, tgt


class ExactTyper(nn.Module):
    def __init__(self, dim=DIM, hid=HID, att=ATT):
        super().__init__()
        self.enc = CharCollapse(dim=dim, max_len=MAXLEN)
        self.key = nn.Linear(3 * dim, att)
        self.val = nn.Linear(3 * dim, dim)
        self.char_embed = nn.Embedding(N_TOK, dim, padding_idx=PAD)
        self.seed = nn.Linear(dim, hid)
        self.cell = nn.GRUCell(dim + dim, hid)
        self.Wq = nn.Linear(hid, att)
        self.v = nn.Linear(att, 1)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))   # monotonic sharpness
        self.out = nn.Linear(hid + dim, N_TOK)

    def memory(self, enc_ids):
        _, _, fused, path = self.enc.encode(enc_ids)
        states = torch.stack(path, dim=1)                       # (B, L, dim)
        B, L, _ = states.shape
        anc = F.normalize(self.enc.letter_anchors, dim=-1)[enc_ids]   # (B, L, dim)
        pos = self.enc.pos_embed[:L].unsqueeze(0).expand(B, L, -1)    # (B, L, dim)
        mem = torch.cat([states, anc, pos], dim=-1)             # (B, L, 3*dim)
        return fused, self.key(mem), self.val(mem), (enc_ids != self.enc.pad_idx)

    def attend(self, h, key, val, mask, t):
        L = key.size(1)
        src = torch.arange(L, device=key.device).float()
        bias = -F.softplus(self.log_alpha) * (src - float(t)) ** 2   # (L,) monotonic
        score = self.v(torch.tanh(self.Wq(h).unsqueeze(1) + key)).squeeze(-1)  # (B, L)
        score = score + bias.unsqueeze(0)
        score = score.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(score, dim=-1)
        return (alpha.unsqueeze(-1) * val).sum(1)

    def forward(self, enc_ids, dec_in):
        fused, key, val, mask = self.memory(enc_ids)
        h = torch.tanh(self.seed(fused))
        logits = []
        for t in range(dec_in.size(1)):
            ctx = self.attend(h, key, val, mask, t)
            h = self.cell(torch.cat([self.char_embed(dec_in[:, t]), ctx], -1), h)
            logits.append(self.out(torch.cat([h, ctx], -1)))
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def type_word(self, enc_ids_single):
        fused, key, val, mask = self.memory(enc_ids_single)
        h = torch.tanh(self.seed(fused))
        tok = torch.tensor([BOS], device=fused.device)
        typed = []
        for t in range(MAXLEN):
            ctx = self.attend(h, key, val, mask, t)
            h = self.cell(torch.cat([self.char_embed(tok), ctx], -1), h)
            nxt = int(self.out(torch.cat([h, ctx], -1)).argmax(-1))
            if nxt == EOS:
                break
            typed.append(T2C.get(nxt, "?"))
            tok = torch.tensor([nxt], device=fused.device)
        return "".join(typed)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    words = load_snli_words()
    random.shuffle(words)
    test_words = words[:HELDOUT]
    train_words = words[HELDOUT:]
    print(f"alphabetic SNLI words : {len(words)}   device: {device}")
    print(f"train / held-out      : {len(train_words)} / {len(test_words)}\n")

    model = ExactTyper().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)

    enc_tr = model.enc.vocab.encode_batch(train_words, MAXLEN)
    din_tr = torch.tensor([dec_target(w)[0] for w in train_words])
    tgt_tr = torch.tensor([dec_target(w)[1] for w in train_words])

    model.train()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(train_words), (BATCH,))
        logits = model(enc_tr[idx].to(device), din_tr[idx].to(device))
        loss = lossf(logits.reshape(-1, N_TOK), tgt_tr[idx].reshape(-1).to(device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == 1:
            print(f"step {step:5d}   loss {loss.item():.4f}")

    model.eval()

    def exact(word_list, cap=3000):
        s = word_list[:cap]
        ids = model.enc.vocab.encode_batch(s, MAXLEN).to(device)
        ok = sum(model.type_word(ids[k:k + 1]) == s[k] for k in range(len(s)))
        return ok, len(s)

    ok_tr, n_tr = exact(train_words)
    ok_te, n_te = exact(test_words)
    print(f"\ntrained-word exact typing : {ok_tr}/{n_tr} = {ok_tr/n_tr*100:.2f}%")
    print(f"held-out exact typing     : {ok_te}/{n_te} = {ok_te/n_te*100:.2f}%")

    # show any remaining trained-word misses
    print("\n--- remaining misses on trained words (first 15) ---")
    shown = 0
    ids = model.enc.vocab.encode_batch(train_words[:3000], MAXLEN).to(device)
    for k, w in enumerate(train_words[:3000]):
        typed = model.type_word(ids[k:k + 1])
        if typed != w:
            print(f"  {w:>16s} -> {typed}")
            shown += 1
            if shown >= 15:
                break
    if shown == 0:
        print("  (none — 100% on the shown sample)")

    torch.save({"char_collapse": model.enc.state_dict(),
                "config": {"dim": DIM, "max_len": MAXLEN},
                "typer": model.state_dict()}, "char_typer_exact.pt")
    print("\nsaved -> char_typer_exact.pt")


if __name__ == "__main__":
    main()
