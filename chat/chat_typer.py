"""
chat_typer.py — the sentence typer, SIMPLIFIED, learning to WRITE your chats.

Same idea as sentence_typer.py: WORDS are wells, a SENTENCE is the trajectory,
and the only job is to WRITE the sentence back from the word-collapse path.
Nothing but pure wells + cosine. No MLP, no GRU, no attention.

WHAT CHANGED vs sentence_typer.py:
  * DROPPED the char stage. sentence_typer seeded every word well from a frozen
    char-geometry model (char_typer_pure.pt) and kept a "char tie" penalty. That
    checkpoint isn't here and it complicated things, so we cut it entirely.
  * Word wells now start from a plain random init and are free to move under the
    ONE loss that matters: reconstruct the sentence. (No char_init, no tie term.)
  * Reads plain sentences (one per line) instead of SNLI jsonl.

Everything else — the collapse encode, the cosine decode, strength/temp/start —
is unchanged, so this is the same typer, just standing on its own.

ENCODE (one attraction step per word, in order):
    h <- h - strength * (1 - cos(h, W_word)) * normalize(h - W_word)
DECODE (pure geometry):
    at position i: cos(state_i, EVERY word well) -> pick nearest word, stop at EOS.

Usage:
    python3 chat_typer.py                                  # full run (uses defaults)
    python3 chat_typer.py --data data/chat_sentences.txt --steps 6000 --max-vocab 20000
    python3 chat_typer.py --max-lines 2000 --steps 300     # quick smoke test
"""

import argparse
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
DIM = 256                 # sentence-level space
MAX_WORDS = 32            # longest sentence kept (words), EOS appended after
MAXLEN = 34               # words + EOS, then padded
STEPS_PER_WORD = 1
STEPS = 6000
BATCH = 128
LR = 5e-3
HELDOUT = 2000
MAX_VOCAB = 20000

PAD = 0
CKPT_OUT = "model/chat_typer.pt"


def read_sentences(path, max_lines=0):
    """One lowercase sentence per line. Dedup, keep order."""
    sents, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_lines and len(sents) >= max_lines:
                break
            s = line.strip().lower()
            if s and s not in seen:
                seen.add(s)
                sents.append(s)
    return sents


def build_word_vocab(sents, max_vocab):
    """One well per word (most frequent), plus PAD(0), <unk>, EOS.
    max_vocab=0 keeps EVERY word — the full corpus gets wells."""
    cnt = Counter(t for s in sents for t in s.split())
    keep = [w for w, _ in cnt.most_common(max_vocab if max_vocab > 0 else None)]
    stoi = {w: i + 1 for i, w in enumerate(keep)}     # 0 = PAD
    unk = len(keep) + 1
    eos = len(keep) + 2
    itos = {i: w for w, i in stoi.items()}
    itos[unk] = "<unk>"; itos[eos] = "<eos>"
    n_words = len(keep) + 3
    return stoi, itos, unk, eos, n_words


def encode_batch(sents, stoi, unk, eos):
    out = []
    for s in sents:
        ids = [stoi.get(t, unk) for t in s.split()][:MAX_WORDS] + [eos]
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
    return " ".join(out)


class ChatTyper(nn.Module):
    """Word wells TRAINABLE, random init. Pure wells + cosine. No char stage."""

    def __init__(self, n_words, dim=DIM):
        super().__init__()
        self.n_words = n_words
        self.word_anchors = nn.Parameter(torch.randn(n_words, dim) * (1.0 / dim ** 0.5))
        self.start = nn.Parameter(torch.randn(dim) * 0.05)
        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def strength(self):
        return torch.sigmoid(self.log_strength)

    @property
    def temp(self):
        return F.softplus(self.log_temp) + 1e-3

    def encode(self, word_ids):
        if word_ids.dim() == 1:
            word_ids = word_ids.unsqueeze(0)
        B, L = word_ids.shape
        anchors = F.normalize(self.word_anchors, dim=-1)
        mask = (word_ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        states = []
        s = self.strength
        for i in range(L):
            target = anchors[word_ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            for _ in range(STEPS_PER_WORD):
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
        anchors = F.normalize(self.word_anchors, dim=-1)
        sn = F.normalize(states, dim=-1)
        return (sn @ anchors.t()) / self.temp

    def loss(self, word_ids):
        states, _ = self.encode(word_ids)
        logits = self.logits(states)
        ce = F.cross_entropy(
            logits.reshape(-1, self.n_words), word_ids.reshape(-1), ignore_index=PAD
        )
        return ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/chat_sentences.txt")
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--max-vocab", type=int, default=MAX_VOCAB,
                    help="0 = keep every word in the corpus (full vocabulary)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--out", default=CKPT_OUT)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    import time
    random.seed(SEED); torch.manual_seed(SEED)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    sents = read_sentences(args.data, args.max_lines)
    random.shuffle(sents)
    sents = [s for s in sents if len(s.split()) <= MAX_WORDS]

    heldout = min(HELDOUT, len(sents) // 10)
    test_sents = sents[:heldout]
    train_sents = sents[heldout:]

    stoi, itos, unk, eos, n_words = build_word_vocab(train_sents, args.max_vocab)
    print(f"sentences (unique)   : {len(sents)}")
    print(f"train / held-out      : {len(train_sents)} / {len(test_sents)}")
    print(f"word wells (TRAINABLE): {n_words-3} words (+PAD+UNK+EOS = {n_words})")
    print(f"dim {DIM}   steps/word {STEPS_PER_WORD}   lr {args.lr}   device {device}   (MLP: NONE, GRU: NONE, CHAR: NONE)\n")

    model = ChatTyper(n_words).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_ids = encode_batch(train_sents, stoi, unk, eos).to(device)

    model.train()
    t0 = time.time(); t_mark = t0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train_sents), (args.batch,), device=device)
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
    print(f"\ntrained {args.steps} steps in {total/60:.1f} min "
          f"({args.steps/total:.1f} steps/s) on {device}\n", flush=True)

    model.eval()

    def scores(sent_list, cap=3000, bs=256):
        s = sent_list[:cap]
        word_hits = word_tot = exact = 0
        exact_clean = clean_tot = 0
        with torch.no_grad():
            for j in range(0, len(s), bs):
                chunk = s[j:j + bs]
                ids = encode_batch(chunk, stoi, unk, eos).to(device)
                states, mask = model.encode(ids)
                lg = model.logits(states)
                lg[..., PAD] = float("-inf"); lg[..., unk] = float("-inf")
                pred = lg.argmax(-1)
                word_hits += ((pred == ids) & mask).sum().item()
                word_tot += mask.sum().item()
                for k in range(len(chunk)):
                    got = decode_pred(pred[k], itos, eos)
                    tgt = decode_pred(ids[k], itos, eos)
                    if got == tgt:
                        exact += 1
                    if not (ids[k] == unk).any().item():
                        clean_tot += 1
                        if got == tgt:
                            exact_clean += 1
        return (word_hits / max(1, word_tot),
                exact / max(1, len(s)),
                exact_clean / max(1, clean_tot))

    tr_w, tr_s, tr_clean = scores(train_sents)
    te_w, te_s, te_clean = scores(test_sents)

    print("\n--- PURE GEOMETRY (random-init wells) — writing your chat sentences ---")
    print(f"  trained : per-word {tr_w*100:5.1f}%   exact-sentence {tr_s*100:5.1f}%   (clean OOV-free: {tr_clean*100:5.1f}%)")
    print(f"  held-out: per-word {te_w*100:5.1f}%   exact-sentence {te_s*100:5.1f}%   (clean OOV-free: {te_clean*100:5.1f}%)")
    print(f"  learned strength {model.strength.item():.3f}   temp {model.temp.item():.3f}")

    print("\n--- example sentences typed back (held-out) ---")
    show = test_sents[:8]
    ids = encode_batch(show, stoi, unk, eos).to(device)
    with torch.no_grad():
        lg = model.logits(model.encode(ids)[0])
        lg[..., PAD] = float("-inf"); lg[..., unk] = float("-inf")
        pred = lg.argmax(-1)
    for k in range(len(show)):
        got = decode_pred(pred[k], itos, eos)
        tgt = decode_pred(ids[k], itos, eos)
        mark = "OK " if got == tgt else "XX "
        print(f"  {mark}{tgt}")
        if got != tgt:
            print(f"     got -> {got}")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"word_anchors": model.word_anchors.detach().cpu(),
                "start": model.start.detach().cpu(),
                "strength": model.strength.item(), "temp": model.temp.item(),
                "stoi": stoi, "itos": itos, "unk": unk, "eos": eos,
                "config": {"dim": DIM, "n_words": n_words}}, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
