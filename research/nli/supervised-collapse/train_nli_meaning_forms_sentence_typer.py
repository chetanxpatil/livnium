"""
train_nli_meaning_forms_sentence_typer.py — meaning forms around the FROZEN
SENTENCE-WRITER. The top rung of the ladder.

Lower rungs (already done):
    chars are wells  -> writes the WORD      (char_typer_symbols.pt, 100%)
    words are wells  -> writes the SENTENCE  (sentence_typer.pt,     100%)

Now we freeze the sentence-writer's geometry (its 20k word wells + collapse
dynamics) and let MEANING grow around it under SNLI supervision, exactly like
meaning grew around the char structure before:

    word_vec = word_well(FROZEN, from sentence_typer.pt)     # the writing geometry
             + meaning_residual(TRAINABLE, init 0)           # meaning forms here

    sentence = collapse word_vecs in ORDER  (FROZEN start/strength from the typer)
    pair = engine(sent_prem - sent_hyp) -> cosine to E / N / C   (supervised)

At step 0 every word sits on its sentence-writer well (residual 0) and the model
scores ~chance. Training grows the residual so synonyms drift together and
E/N/C separates — meaning forms around the frozen sentence structure.

Frozen: the word wells + the collapse start/strength (the writer).
Trainable: meaning residual + NLI collapse engine + class anchors.

Run from collapse_retrain/:  python3 train_nli_meaning_forms_sentence_typer.py
Needs torch + sentence_typer.pt + vector_collapse.py.
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_collapse import VectorCollapseEngine

TYPER_CKPT = "sentence_typer.pt"
NLI_LABEL_TO_IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


def find_typer():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [TYPER_CKPT, os.path.join(here, TYPER_CKPT), os.path.join(here, "..", TYPER_CKPT)]:
        if os.path.exists(c):
            return c
    return TYPER_CKPT


def read_nli_jsonl(path, max_lines=0):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_lines and len(out) >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lab = d.get("gold_label", "-")
            if lab not in NLI_LABEL_TO_IDX:
                continue
            s1 = (d.get("sentence1") or "").lower()
            s2 = (d.get("sentence2") or "").lower()
            if s1 and s2:
                out.append((s1, s2, NLI_LABEL_TO_IDX[lab]))
    return out


def encode_line(line, stoi, unk):
    """Tokenize straight into the sentence-typer's WELL ids (its vocab)."""
    return [stoi.get(t, unk) for t in line.strip().split() if t] or [unk]


def collate(batch, pad=0):
    pms, hys, ys = zip(*batch)
    mp = max(len(x) for x in pms); mh = max(len(x) for x in hys)
    f = lambda seqs, m: torch.tensor([s + [pad] * (m - len(s)) for s in seqs], dtype=torch.long)
    return f(pms, mp), f(hys, mh), torch.tensor(ys, dtype=torch.long)


class MeaningOnSentenceWriter(nn.Module):
    def __init__(self, ck, dim):
        super().__init__()
        wells = F.normalize(ck["word_anchors"], dim=-1)       # the frozen writing geometry
        self.well = nn.Embedding.from_pretrained(wells, freeze=True, padding_idx=0)
        self.residual = nn.Embedding(wells.size(0), dim, padding_idx=0)
        nn.init.zeros_(self.residual.weight)                  # meaning starts at nothing
        # frozen sentence-collapse dynamics, taken straight from the trained typer
        self.register_buffer("sent_start", ck["start"].clone())
        self.sent_strength = float(ck["strength"])
        self.engine = VectorCollapseEngine(dim=dim, num_layers=4)

    def encode_sentence(self, ids):
        words = self.well(ids) + self.residual(ids)           # well + grown meaning
        mask = (ids != 0)
        B, L, _ = words.shape
        h = self.sent_start.expand(B, -1).contiguous()
        s = self.sent_strength
        states = []
        for i in range(L):
            target = words[:, i, :]
            tn = F.normalize(target, dim=-1)
            m = mask[:, i].float().unsqueeze(-1)
            align = (F.normalize(h, dim=-1) * tn).sum(-1)
            div = 1.0 - align
            away = F.normalize(h - target, dim=-1)
            h = h + m * (-s * div.unsqueeze(-1) * away)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
        # FINAL trajectory state, not the mean: padded steps leave h unchanged
        # (m=0 -> no update), so this is the state after the LAST real word -> the end
        # of the ordered walk. Fully order-dependent: "dog bites man" != "man bites dog".
        return h                                               # order-preserving sentence vec

    def anchors(self):
        return torch.stack([
            F.normalize(self.engine.anchor_entail, dim=0),
            F.normalize(self.engine.anchor_neutral, dim=0),
            F.normalize(self.engine.anchor_contra, dim=0),
        ])

    def forward(self, prem, hyp, temp=0.1):
        u = self.encode_sentence(prem)
        v = self.encode_sentence(hyp)
        pair, _ = self.engine(u - v)
        return (F.normalize(pair, dim=-1) @ self.anchors().t()) / temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default="/Users/chetanpatil/Desktop/test/data-bank/snli_1.0_train.jsonl")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.02)
    ap.add_argument("--residual-reg", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    ck = torch.load(find_typer(), map_location="cpu")
    stoi, unk = ck["stoi"], ck["unk"]
    dim = ck["config"]["dim"]
    print(f"loaded sentence_typer.pt  wells {ck['word_anchors'].shape[0]}  dim {dim}  device {device}")
    print("sentence-writer geometry FROZEN (wells + collapse); meaning residual + engine TRAINABLE\n")

    model = MeaningOnSentenceWriter(ck, dim).to(device)
    data = read_nli_jsonl(args.nli_path, args.max_lines)
    data = [(encode_line(s1, stoi, unk), encode_line(s2, stoi, unk), y) for s1, s2, y in data]
    random.seed(0); random.shuffle(data)
    n_dev = int(len(data) * args.dev_frac)
    dev, train = data[:n_dev], data[n_dev:]
    print(f"train {len(train)}   dev {len(dev)}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def acc(d):
        model.eval(); ok = tot = 0
        with torch.no_grad():
            for i in range(0, len(d), args.batch_size):
                p, h, y = collate(d[i:i + args.batch_size])
                ok += (model(p.to(device), h.to(device)).argmax(-1).cpu() == y).sum().item()
                tot += len(y)
        return ok / max(1, tot)

    best_dev = best_ep = 0
    for ep in range(1, args.epochs + 1):
        model.train(); random.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            p, h, y = collate(train[i:i + args.batch_size])
            logits = model(p.to(device), h.to(device))
            a = model.anchors()
            sep = sum(torch.relu((a[x] * a[z]).sum()) for x, z in [(0, 1), (1, 2), (0, 2)])
            reg = args.residual_reg * model.residual.weight.pow(2).sum(-1).mean()
            loss = F.cross_entropy(logits, y.to(device)) + sep + reg
            opt.zero_grad(); loss.backward(); opt.step()
            if (i // args.batch_size) % 200 == 0:
                print(f"  ep{ep} step {i//args.batch_size:4d}  ce {loss.item():.4f}")
        tr = acc(train[:20000]); dv = acc(dev)
        rnorm = model.residual.weight.norm(dim=-1).mean().item()
        flag = ""
        if dv > best_dev:
            best_dev, best_ep = dv, ep
            torch.save({"model": model.state_dict(), "dim": dim, "epoch": ep, "dev": dv},
                       "nli_meaning_forms_sentence_typer.pt")
            flag = "  <- best, saved"
        print(f"epoch {ep}: train {tr*100:.2f}%   dev {dv*100:.2f}%   resid|.| {rnorm:.3f}{flag}")

    print(f"\nbest dev {best_dev*100:.2f}% at epoch {best_ep}  -> nli_meaning_forms_sentence_typer.pt")


if __name__ == "__main__":
    main()
