"""
char_typer_symbols.py — the pure char typer, now for ALL symbols + space-ending.

Same pure Livnium geometry as char_collapse_pure.py (no MLP, no GRU — only learned
anchors + the collapse force + nearest-anchor readout), generalized:

  * The alphabet is built FROM THE DATA: every character that appears in SNLI
    tokens gets its own well — letters, digits, and punctuation like , . - ' etc.
    Whatever symbol comes, it has a well and gets typed/joined.
  * The word ENDS AT SPACE: instead of a generic EOS, there is a dedicated
    SPACE well. The trajectory types characters and only stops when it lands in
    the space well. No space -> the word keeps going (to the length cap).

dim is 128 so the larger symbol set still sits near-orthogonally (you can fit up
to `dim` orthogonal directions; need >= number of symbols).

Run: python3 char_typer_symbols.py   (needs torch; reads model_nli_v1/nli_epoch20.pt)
Saves char_typer_symbols.pt + the learned symbol map.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

NLI_CKPT = "model_nli_v1/nli_epoch20.pt"
SEED = 0
DIM = 128
MAX_WORD = 18
MAXLEN = 20               # token + space, padded
STEPS_PER_CHAR = 1
STEPS = 4000
BATCH = 256
LR = 5e-3
HELDOUT = 1500
PAD = 0


def load_tokens():
    """Every SNLI token (lowercased), keeping punctuation/digits; skip specials."""
    try:
        data = torch.load(NLI_CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(NLI_CKPT, map_location="cpu")
    toks, seen = [], set()
    for w in data["vocab"]["idx2word"]:
        lw = w.lower()
        if not lw or lw.startswith("<") or " " in lw:   # skip <pad>/<unk>, no internal space
            continue
        if len(lw) > MAX_WORD or lw in seen:
            continue
        seen.add(lw)
        toks.append(lw)
    chars = sorted({c for t in toks for c in t})         # the data's alphabet
    return toks, chars


class CharVocabSym:
    def __init__(self, chars):
        self.chars = chars
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}   # 0 = PAD
        self.itos = {i + 1: c for i, c in enumerate(chars)}
        self.space = len(chars) + 1                            # the SPACE / stop well
        self.n = len(chars) + 2                                # PAD + symbols + SPACE

    def encode(self, tok):
        ids = [self.stoi[c] for c in tok if c in self.stoi][:MAX_WORD] + [self.space]
        return ids + [PAD] * (MAXLEN - len(ids))

    def encode_batch(self, toks):
        return torch.tensor([self.encode(t) for t in toks], dtype=torch.long)

    def decode(self, row):
        out = []
        for t in row:
            t = int(t)
            if t == self.space or t == PAD:    # word ends at the space well
                break
            out.append(self.itos.get(t, "?"))
        return "".join(out)


class CharCollapseSym(nn.Module):
    def __init__(self, n_symbols, dim=DIM):
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(n_symbols, dim))
        self.start = nn.Parameter(torch.randn(dim) * 0.05)
        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def strength(self):
        return torch.sigmoid(self.log_strength)

    @property
    def temp(self):
        return F.softplus(self.log_temp) + 1e-3

    def encode(self, ids):
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, L = ids.shape
        A = F.normalize(self.anchors, dim=-1)
        mask = (ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        states = []
        s = self.strength
        for i in range(L):
            target = A[ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            for _ in range(STEPS_PER_CHAR):
                hn = F.normalize(h, dim=-1)
                div = 1.0 - (hn * target).sum(-1)
                away = F.normalize(h - target, dim=-1)
                h = h + m * (-s * div.unsqueeze(-1) * away)
                n = h.norm(dim=-1, keepdim=True)
                h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
        return torch.stack(states, dim=1), mask

    def logits(self, states):
        A = F.normalize(self.anchors, dim=-1)
        return (F.normalize(states, dim=-1) @ A.t()) / self.temp

    def loss(self, ids):
        states, _ = self.encode(ids)
        return F.cross_entropy(self.logits(states).reshape(-1, self.anchors.size(0)),
                               ids.reshape(-1), ignore_index=PAD)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    toks, chars = load_tokens()
    vocab = CharVocabSym(chars)
    random.shuffle(toks)
    test, train = toks[:HELDOUT], toks[HELDOUT:]

    shown = "".join(c if c.strip() else "·" for c in chars)
    print(f"tokens (incl. punctuation) : {len(toks)}")
    print(f"learned symbols ({len(chars)}) : {shown}")
    print(f"+ PAD + SPACE-well  ->  {vocab.n} wells   dim {DIM}")
    print(f"train / held-out    : {len(train)} / {len(test)}   (MLP: NONE, GRU: NONE)\n")

    model = CharCollapseSym(vocab.n)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_ids = vocab.encode_batch(train)

    model.train()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(train), (BATCH,))
        loss = model.loss(train_ids[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  strength {model.strength.item():.3f}")

    model.eval()

    def score(lst, cap=3000):
        s = lst[:cap]
        ids = vocab.encode_batch(s)
        with torch.no_grad():
            states, mask = model.encode(ids)
            pred = model.logits(states).argmax(-1)
        char_ok = ((pred == ids) & mask).sum().item() / mask.sum().item()
        exact = sum(vocab.decode(pred[k]) == s[k] for k in range(len(s)))
        return char_ok, exact / len(s)

    trc, trw = score(train)
    tec, tew = score(test)
    print("\n--- PURE GEOMETRY, all symbols, space-ended ---")
    print(f"  trained : per-char {trc*100:5.1f}%   exact-token {trw*100:5.1f}%")
    print(f"  held-out: per-char {tec*100:5.1f}%   exact-token {tew*100:5.1f}%")
    print(f"  strength {model.strength.item():.3f}  temp {model.temp.item():.3f}")

    print("\n--- examples (held-out, punctuation included) ---")
    picks = [t for t in test if any(not c.isalpha() for c in t)][:8] + test[:4]
    ids = vocab.encode_batch(picks)
    with torch.no_grad():
        pred = model.logits(model.encode(ids)[0]).argmax(-1)
    for k, w in enumerate(picks):
        got = vocab.decode(pred[k])
        print(f"  {'OK ' if got == w else 'XX '}{w:>16s} -> {got}")

    torch.save({"anchors": model.anchors.detach(), "start": model.start.detach(),
                "strength": model.strength.item(), "temp": model.temp.item(),
                "chars": chars, "config": {"dim": DIM}}, "char_typer_symbols.pt")
    print("\nsaved -> char_typer_symbols.pt")


if __name__ == "__main__":
    main()
