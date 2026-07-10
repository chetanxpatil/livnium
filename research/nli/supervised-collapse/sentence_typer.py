"""
sentence_typer.py — the char typer, one level up. WORDS are wells, the SENTENCE
is the trajectory. No NLI. No labels. Nothing but learning to WRITE the sentence
back from the word-collapse path.

Exact parallel to char_typer_symbols.py:
    char typer : characters are wells -> types a WORD back     (per-letter readout)
    this       : words are wells      -> types a SENTENCE back (per-word readout)

PURE geometry, no MLP / no GRU. The only learnable things are:
    - the word anchors      (the wells — where each word sits)
    - a start state         (where every sentence's trajectory begins)
    - one scalar strength   (how hard the wells pull)
    - one scalar temp       (sharpness of the cosine readout)

ENCODE (one attraction step per word, in order):
    h <- h - strength * (1 - cos(h, W_word)) * normalize(h - W_word)
A state is kept per word position (NOT fused to one point) so the sentence can
be unrolled — same reason the char typer keeps a state per character.

DECODE (pure geometry, no network):
    at position i: cos(state_i, EVERY word anchor) -> pick nearest word.
    type words, joining them, and STOP at the EOS well.

Trained only on SNLI *sentences* (premises + hypotheses), labels ignored.

Run from collapse_retrain/:  python3 sentence_typer.py
Needs torch.
"""

import argparse
import json
import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
DIM = 256                 # words need more room than 26 letters; still << vocab
MAX_WORDS = 32            # longest sentence kept (words), EOS appended after
MAXLEN = 34               # words + EOS, then padded
STEPS_PER_WORD = 1        # 1 = mirror the char typer exactly
STEPS = 6000
BATCH = 128
LR = 5e-3
HELDOUT = 2000
MAX_VOCAB = 20000         # keep the most frequent words as wells; rest -> <unk>

PAD = 0


def read_sentences(path, max_lines=0):
    """Just the sentences. Labels are thrown away — this is unsupervised typing."""
    sents, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_lines and len(sents) >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for key in ("sentence1", "sentence2"):
                s = (d.get(key) or "").lower().strip()
                if s and s not in seen:
                    seen.add(s)
                    sents.append(s)
    return sents


def build_word_vocab(sents, max_vocab):
    """One well per word (the most frequent ones), plus PAD(0), <unk>, EOS."""
    cnt = Counter(t for s in sents for t in s.split())
    keep = [w for w, _ in cnt.most_common(max_vocab)]
    stoi = {w: i + 1 for i, w in enumerate(keep)}     # 0 = PAD
    unk = len(keep) + 1
    eos = len(keep) + 2
    itos = {i: w for w, i in stoi.items()}
    itos[unk] = "<unk>"; itos[eos] = "<eos>"
    n_words = len(keep) + 3                            # PAD + words + UNK + EOS
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


class SentenceTyper(nn.Module):
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
        return F.cross_entropy(
            logits.reshape(-1, self.n_words), word_ids.reshape(-1), ignore_index=PAD
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default="/Users/chetanpatil/Desktop/test/data-bank/snli_1.0_train.jsonl")
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--max-vocab", type=int, default=MAX_VOCAB)
    args = ap.parse_args()

    random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    sents = read_sentences(args.nli_path, args.max_lines)
    random.shuffle(sents)
    sents = [s for s in sents if len(s.split()) <= MAX_WORDS]
    stoi, itos, unk, eos, n_words = build_word_vocab(sents, args.max_vocab)

    test_sents = sents[:HELDOUT]
    train_sents = sents[HELDOUT:]
    print(f"unique SNLI sentences   : {len(sents)}")
    print(f"train / held-out        : {len(train_sents)} / {len(test_sents)}")
    print(f"word wells learned      : {n_words-3}  (+ PAD + UNK + EOS = {n_words} anchors)")
    print(f"dim {DIM}   steps/word {STEPS_PER_WORD}   device {device}   (MLP: NONE, GRU: NONE)\n")

    model = SentenceTyper(n_words).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_ids = encode_batch(train_sents, stoi, unk, eos).to(device)

    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train_sents), (args.batch,), device=device)
        loss = model.loss(train_ids[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  strength {model.strength.item():.3f}")

    model.eval()

    def scores(sent_list, cap=3000, bs=256):
        s = sent_list[:cap]
        word_hits = word_tot = exact = 0
        with torch.no_grad():
            for j in range(0, len(s), bs):                 # batch so logits stay small
                chunk = s[j:j + bs]
                ids = encode_batch(chunk, stoi, unk, eos).to(device)
                states, mask = model.encode(ids)
                pred = model.logits(states).argmax(-1)     # (bs, L, V) only for bs rows
                word_hits += ((pred == ids) & mask).sum().item()
                word_tot += mask.sum().item()
                for k in range(len(chunk)):
                    if decode_pred(pred[k], itos, eos) == decode_pred(ids[k], itos, eos):
                        exact += 1
        return word_hits / max(1, word_tot), exact / len(s)

    tr_w, tr_s = scores(train_sents)
    te_w, te_s = scores(test_sents)
    print("\n--- PURE GEOMETRY (no MLP, no GRU) — writing sentences from word wells ---")
    print(f"  trained : per-word {tr_w*100:5.1f}%   exact-sentence {tr_s*100:5.1f}%")
    print(f"  held-out: per-word {te_w*100:5.1f}%   exact-sentence {te_s*100:5.1f}%")
    print(f"  learned strength {model.strength.item():.3f}   temp {model.temp.item():.3f}")

    print("\n--- example sentences typed back (held-out) ---")
    show = test_sents[:8]
    ids = encode_batch(show, stoi, unk, eos).to(device)
    with torch.no_grad():
        pred = model.logits(model.encode(ids)[0]).argmax(-1)
    for k, w in enumerate(show):
        got = decode_pred(pred[k], itos, eos)
        tgt = decode_pred(ids[k], itos, eos)               # target after unk-mapping
        mark = "OK " if got == tgt else "XX "
        print(f"  {mark}{tgt}")
        if got != tgt:
            print(f"     got -> {got}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- where the knowledge lives ({n_params} numbers total) ---")
    print(f"  word_anchors {tuple(model.word_anchors.shape)} = {model.word_anchors.numel()}  <- ~all of it")
    print(f"  MLP params: 0    GRU params: 0")

    torch.save({"word_anchors": model.word_anchors.detach().cpu(),
                "start": model.start.detach().cpu(),
                "strength": model.strength.item(), "temp": model.temp.item(),
                "stoi": stoi, "itos": itos, "unk": unk, "eos": eos,
                "config": {"dim": DIM, "n_words": n_words}}, "sentence_typer.pt")
    print("\nsaved -> sentence_typer.pt")


if __name__ == "__main__":
    main()
