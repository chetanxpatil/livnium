"""
char_typer_symbols.py — pure CharCollapse, but the alphabet is the WHOLE symbol set.

Same deal as char_collapse_pure.py: NO MLP, NO GRU, NO learned readout matrix.
The only learnable things are the symbol anchors (gravity wells), a start state,
and two scalars (strength, temp). The classifier IS the geometry.

What's different here, exactly what you asked for:
  - The alphabet is BUILT FROM THE DATA. Every character that shows up in an
    SNLI token gets its own well — letters, digits, and  ,  .  -  '  /  etc.
    "whatever symbol comes, let it join."
  - The old EOS well becomes a dedicated SPACE well. A token is encoded as
    chars... + SPACE, and decoding types symbols and STOPS only when it lands
    in the space well. "end the word at ' '; no space, no word end."

DIM is bumped to 128 so the bigger symbol set still fits near-orthogonally
(you need at least as many dimensions as symbols to keep the wells distinct).

Run:  python3 char_typer_symbols.py
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_ckpt():
    """Locate the NLI vocab checkpoint whether run from livnium/ or research/nli/supervised-collapse/."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "model_nli_v1/nli_epoch20.pt",
        "research/nli/supervised-collapse/model_nli_v1/nli_epoch20.pt",
        os.path.join(here, "../../nli/supervised-collapse/model_nli_v1/nli_epoch20.pt"),
        os.path.join(here, "model_nli_v1/nli_epoch20.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


NLI_CKPT = _find_ckpt()
SEED = 0
DIM = 128
MAX_WORD = 18             # longest token kept (chars), space appended after
MAXLEN = 20               # token + SPACE, then padded
STEPS_PER_CHAR = 1        # 1 = mirror CharCollapse exactly
STEPS = 3500
BATCH = 256
LR = 5e-3
HELDOUT = 1500

PAD = 0                   # index 0 is reserved for padding


def load_snli_tokens():
    """Pull every token from the NLI vocab, lowercased, keep any symbol it has."""
    try:
        data = torch.load(NLI_CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(NLI_CKPT, map_location="cpu")
    toks, seen = [], set()
    for w in data["vocab"]["idx2word"]:
        lw = str(w).lower().strip()
        # drop empties, special <tokens>, anything with a literal space inside,
        # and over-long tokens. EVERYTHING else (punctuation, digits) is kept.
        if not lw or " " in lw or len(lw) > MAX_WORD:
            continue
        if lw.startswith("<") and lw.endswith(">"):
            continue
        if lw in seen:
            continue
        seen.add(lw)
        toks.append(lw)
    return toks


def build_charset(tokens):
    """One well per character seen in the data, plus PAD(0) and a SPACE well."""
    chars = sorted({c for t in tokens for c in t})
    # index 0 = PAD; symbols start at 1; SPACE is the last index (the terminator)
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    itos = {i + 1: c for i, c in enumerate(chars)}
    space = len(chars) + 1            # dedicated SPACE / STOP well
    n_chars = len(chars) + 2          # PAD + symbols + SPACE
    return stoi, itos, space, n_chars, chars


def encode_batch(tokens, stoi, space):
    out = []
    for t in tokens:
        ids = [stoi[c] for c in t][:MAX_WORD] + [space]   # append the SPACE terminator
        ids += [PAD] * (MAXLEN - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)


def decode_pred(row, itos, space):
    """Type symbols, joining whatever comes; STOP only at the space well (or pad)."""
    out = []
    for t in row:
        t = int(t)
        if t == space or t == PAD:
            break
        out.append(itos.get(t, "?"))
    return "".join(out)


class CharTyperSymbols(nn.Module):
    def __init__(self, n_chars, dim=DIM):
        super().__init__()
        self.n_chars = n_chars
        self.symbol_anchors = nn.Parameter(torch.randn(n_chars, dim))
        self.start = nn.Parameter(torch.randn(dim) * 0.05)
        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def strength(self):
        return torch.sigmoid(self.log_strength)

    @property
    def temp(self):
        return F.softplus(self.log_temp) + 1e-3

    def encode(self, char_ids):
        if char_ids.dim() == 1:
            char_ids = char_ids.unsqueeze(0)
        B, L = char_ids.shape
        anchors = F.normalize(self.symbol_anchors, dim=-1)
        mask = (char_ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        states = []
        s = self.strength
        for i in range(L):
            target = anchors[char_ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            for _ in range(STEPS_PER_CHAR):
                align = (F.normalize(h, dim=-1) * target).sum(-1)
                div = 1.0 - align
                away = F.normalize(h - target, dim=-1)
                step = -s * div.unsqueeze(-1) * away
                h = h + m * step
                n = h.norm(dim=-1, keepdim=True)
                h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
        return torch.stack(states, dim=1), mask

    def logits(self, states):
        anchors = F.normalize(self.symbol_anchors, dim=-1)
        sn = F.normalize(states, dim=-1)
        return (sn @ anchors.t()) / self.temp

    def loss(self, char_ids):
        states, _ = self.encode(char_ids)
        logits = self.logits(states)
        return F.cross_entropy(
            logits.reshape(-1, self.n_chars), char_ids.reshape(-1), ignore_index=PAD
        )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokens = load_snli_tokens()
    random.shuffle(tokens)
    stoi, itos, space, n_chars, chars = build_charset(tokens)

    test_tokens = tokens[:HELDOUT]
    train_tokens = tokens[HELDOUT:]
    print(f"SNLI tokens (any symbol): {len(tokens)}")
    print(f"train / held-out        : {len(train_tokens)} / {len(test_tokens)}")
    print(f"symbol wells learned    : {len(chars)}  (+ PAD + SPACE = {n_chars} anchors)")
    print(f"dim {DIM}   steps/char {STEPS_PER_CHAR}   (MLP: NONE, GRU: NONE)\n")
    print("learned symbol set:")
    print("  " + " ".join(repr(c) for c in chars) + "\n")

    model = CharTyperSymbols(n_chars)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_ids = encode_batch(train_tokens, stoi, space)

    model.train()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(train_tokens), (BATCH,))
        loss = model.loss(train_ids[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  strength {model.strength.item():.3f}")

    model.eval()

    def scores(tok_list, cap=3000):
        s = tok_list[:cap]
        ids = encode_batch(s, stoi, space)
        with torch.no_grad():
            states, mask = model.encode(ids)
            pred = model.logits(states).argmax(-1)
        char_ok = ((pred == ids) & mask).sum().item() / mask.sum().item()
        exact = sum(decode_pred(pred[k], itos, space) == w for k, w in enumerate(s))
        return char_ok, exact / len(s)

    tr_c, tr_w = scores(train_tokens)
    te_c, te_w = scores(test_tokens)
    print("\n--- PURE GEOMETRY (no MLP, no GRU) ---")
    print(f"  trained : per-symbol {tr_c*100:5.1f}%   exact-token {tr_w*100:5.1f}%")
    print(f"  held-out: per-symbol {te_c*100:5.1f}%   exact-token {te_w*100:5.1f}%")
    print(f"  learned strength {model.strength.item():.3f}   temp {model.temp.item():.3f}")

    # show punctuation / digit tokens typing back, since that's the point here
    print("\n--- examples (held-out), punctuation & digits included ---")
    nonword = [w for w in test_tokens if any(not c.isalpha() for c in w)][:6]
    show = (nonword + test_tokens)[:12]
    ids = encode_batch(show, stoi, space)
    with torch.no_grad():
        pred = model.logits(model.encode(ids)[0]).argmax(-1)
    for k, w in enumerate(show):
        got = decode_pred(pred[k], itos, space)
        mark = "OK " if got == w else "XX "
        print(f"  {mark}{w:>16s} -> {got}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- where the knowledge lives ({n_params} numbers total) ---")
    print(f"  symbol_anchors {tuple(model.symbol_anchors.shape)} = {model.symbol_anchors.numel()}  <- ~all of it")
    print(f"  MLP params: 0    GRU params: 0    word table: none")

    torch.save({"symbol_anchors": model.symbol_anchors.detach(),
                "start": model.start.detach(),
                "strength": model.strength.item(),
                "temp": model.temp.item(),
                "stoi": stoi, "itos": itos, "space": space,
                "config": {"dim": DIM, "n_chars": n_chars}}, "char_typer_symbols.pt")
    print("\nsaved -> char_typer_symbols.pt")


if __name__ == "__main__":
    main()
