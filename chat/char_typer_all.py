"""
char_typer_all.py — the CHAR rung, trained on the RAW chats. ALL characters.

Same collapse engine as chat_typer.py, one level down:
    a WORD  is to CHARS  what a SENTENCE is to WORDS.
Chars are wells, a line of text is the trajectory, and the job is to TYPE the
line back char by char. No cleaning, no stripping: code stays code, markdown
stays markdown, and ENTER (\\n) is a first-class char with its own well — the
model learns where lines end by typing enter.

Alphabet: EVERY character in the raw export (~1,620 incl emoji, Greek, box
drawing, Devanagari). --min-count folds one-off chars into <rare> if you want,
but the default keeps all — at char level the vocabulary is tiny either way.

ENCODE (one attraction step per char, in order):
    h <- h - strength * (1 - cos(h, W_char)) * normalize(h - W_char)
DECODE (pure geometry):
    at position i: cos(state_i, EVERY char well) -> nearest char, stop at EOS.

Data: reads the RAW conversations.json — the same source, walked by the same
canonical path (prep_chat_context.canonical_turns), as every other rung. One
source of truth; the flatten is not used anywhere. Text splits into lines;
each line keeps its trailing ENTER; long lines are chunked.

Usage:
    python3 char_typer_all.py                          # full run on the raw export
    python3 char_typer_all.py --max-lines 20000 --steps 500    # smoke test
"""

import argparse
import json
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

RAW_IN = ("/Users/chetanpatil/Desktop/test/lab/infected/projects/"
          "chat_crystal/build/unit_test_assets/assets/conversations.json")

SEED = 0
DIM = 128                 # char space (smaller than word space; alphabet is tiny)
MAX_CHARS = 80            # longest line kept per sequence; longer lines are chunked
MAXLEN = 82               # chars + ENTER + EOS, then padded
STEPS = 6000
BATCH = 128
LR = 5e-3
HELDOUT = 2000

PAD = 0
ENTER = "\n"
CKPT_OUT = "model/char_typer_all.pt"


def read_lines(path, max_lines=0):
    """Every line of every turn on every conversation's CANONICAL path — the
    same walk as prep_chat_context (one source of truth, no flatten). Trailing
    ENTER kept as a char. Lines longer than MAX_CHARS are chunked. Deduped."""
    from prep_chat_context import canonical_turns
    with open(path, encoding="utf-8") as f:
        convs = json.load(f)
    seqs, seen = [], set()
    for conv in convs:
        for role, text in canonical_turns(conv):
            for line in text.split("\n"):
                for i in range(0, max(len(line), 1), MAX_CHARS):
                    chunk = line[i:i + MAX_CHARS]
                    if len(chunk) < 2:
                        continue
                    s = chunk + ENTER          # enter closes every line
                    if s in seen:
                        continue
                    seen.add(s)
                    seqs.append(s)
                    if max_lines and len(seqs) >= max_lines:
                        return seqs
    return seqs


def build_char_vocab(seqs, min_count=1):
    """One well per char. min_count>1 folds rarities into <rare>."""
    cnt = Counter(ch for s in seqs for ch in s)
    keep = [ch for ch, n in cnt.most_common() if n >= min_count]
    stoi = {ch: i + 1 for i, ch in enumerate(keep)}    # 0 = PAD
    rare = len(keep) + 1
    eos = len(keep) + 2
    itos = {i: ch for ch, i in stoi.items()}
    itos[rare] = "<rare>"; itos[eos] = "<eos>"
    return stoi, itos, rare, eos, len(keep) + 3


def encode_batch(seqs, stoi, rare, eos):
    out = []
    for s in seqs:
        ids = [stoi.get(ch, rare) for ch in s[:MAX_CHARS + 1]] + [eos]
        ids += [PAD] * (MAXLEN - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)


def decode_pred(row, itos, eos):
    out = []
    for t in row:
        t = int(t)
        if t == eos or t == PAD:
            break
        out.append(itos.get(t, "?"))
    return "".join(out)


class CharTyper(nn.Module):
    """Char wells TRAINABLE, random init. Pure wells + cosine. Same engine."""

    def __init__(self, n_chars, dim=DIM):
        super().__init__()
        self.n_chars = n_chars
        self.char_anchors = nn.Parameter(torch.randn(n_chars, dim) * (1.0 / dim ** 0.5))
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
        B, L = char_ids.shape
        anchors = F.normalize(self.char_anchors, dim=-1)
        mask = (char_ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        states = []
        s = self.strength
        for i in range(L):
            target = anchors[char_ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            align = (F.normalize(h, dim=-1) * target).sum(-1)
            away = F.normalize(h - target, dim=-1)
            h = h + m * (-s * (1.0 - align).unsqueeze(-1) * away)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
        return torch.stack(states, dim=1), mask

    def logits(self, states):
        anchors = F.normalize(self.char_anchors, dim=-1)
        return (F.normalize(states, dim=-1) @ anchors.t()) / self.temp

    def loss(self, char_ids):
        states, _ = self.encode(char_ids)
        return F.cross_entropy(self.logits(states).reshape(-1, self.n_chars),
                               char_ids.reshape(-1), ignore_index=PAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_IN)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--min-count", type=int, default=1,
                    help="1 = every char gets a well (default: ALL)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--out", default=CKPT_OUT)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    import time
    random.seed(SEED); torch.manual_seed(SEED)
    device = (torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    print("reading raw export ...", flush=True)
    seqs = read_lines(args.raw, args.max_lines)
    random.shuffle(seqs)
    heldout = min(HELDOUT, len(seqs) // 10)
    test_seqs, train_seqs = seqs[:heldout], seqs[heldout:]

    stoi, itos, rare, eos, n_chars = build_char_vocab(train_seqs, args.min_count)
    print(f"lines (unique, chunked) : {len(seqs)}")
    print(f"train / held-out        : {len(train_seqs)} / {len(test_seqs)}")
    print(f"char wells (TRAINABLE)  : {n_chars-3} chars (+PAD+RARE+EOS = {n_chars})   incl ENTER")
    print(f"dim {DIM}   lr {args.lr}   device {device}   (MLP: NONE, GRU: NONE)\n")

    model = CharTyper(n_chars).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_ids = encode_batch(train_seqs, stoi, rare, eos).to(device)

    model.train()
    t0 = time.time(); t_mark = t0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train_seqs), (args.batch,), device=device)
        loss = model.loss(train_ids[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == 1:
            now = time.time()
            sps = 250 / (now - t_mark) if step > 1 else float("nan")
            eta = (args.steps - step) / sps / 60 if sps == sps else float("nan")
            t_mark = now
            print(f"step {step:5d}  ce {loss.item():.4f}  "
                  f"strength {model.strength.item():.3f}  temp {model.temp.item():.3f}  "
                  f"| {sps:5.1f} steps/s  eta {eta:4.1f} min", flush=True)
    total = time.time() - t0
    print(f"\ntrained {args.steps} steps in {total/60:.1f} min on {device}\n", flush=True)

    model.eval()

    def scores(seq_list, cap=3000, bs=256):
        s = seq_list[:cap]
        hits = tot = exact = 0
        with torch.no_grad():
            for j in range(0, len(s), bs):
                chunk = s[j:j + bs]
                ids = encode_batch(chunk, stoi, rare, eos).to(device)
                states, mask = model.encode(ids)
                lg = model.logits(states)
                lg[..., PAD] = float("-inf")
                pred = lg.argmax(-1)
                hits += ((pred == ids) & mask).sum().item()
                tot += mask.sum().item()
                for k in range(len(chunk)):
                    if decode_pred(pred[k], itos, eos) == decode_pred(ids[k], itos, eos):
                        exact += 1
        return hits / max(1, tot), exact / max(1, len(s))

    tr_c, tr_x = scores(train_seqs)
    te_c, te_x = scores(test_seqs)
    print("--- PURE GEOMETRY — typing the raw chats back, char by char ---")
    print(f"  trained : per-char {tr_c*100:5.1f}%   exact-line {tr_x*100:5.1f}%")
    print(f"  held-out: per-char {te_c*100:5.1f}%   exact-line {te_x*100:5.1f}%")
    print(f"  learned strength {model.strength.item():.3f}   temp {model.temp.item():.3f}")

    print("\n--- example lines typed back (held-out) ---")
    show = test_seqs[:8]
    ids = encode_batch(show, stoi, rare, eos).to(device)
    with torch.no_grad():
        lg = model.logits(model.encode(ids)[0])
        lg[..., PAD] = float("-inf")
        pred = lg.argmax(-1)
    for k in range(len(show)):
        got = decode_pred(pred[k], itos, eos)
        tgt = decode_pred(ids[k], itos, eos)
        mark = "OK " if got == tgt else "XX "
        print(f"  {mark}{tgt.rstrip()}")
        if got != tgt:
            print(f"     got -> {got.rstrip()}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"char_anchors": model.char_anchors.detach().cpu(),
                "start": model.start.detach().cpu(),
                "strength": model.strength.item(), "temp": model.temp.item(),
                "stoi": stoi, "itos": itos, "rare": rare, "eos": eos,
                "config": {"dim": DIM, "n_chars": n_chars, "max_chars": MAX_CHARS}},
               args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
