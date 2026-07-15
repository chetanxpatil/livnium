"""
train_nli_meaning_forms_symbols.py — meaning FORMS AROUND the FULL-symbol char structure.

Same idea as train_nli_meaning_forms.py, but the frozen char scaffold now comes
from the symbol typer (char_typer_symbols.pt) instead of the a-z-only pure typer.

So a word's frozen position is the mean of its symbol anchors over EVERY character
it contains — letters, digits, AND punctuation. "u.s.", "don't", "well-being"
keep their dots / apostrophes / hyphens in the scaffold instead of dropping them.

    word_vec = symbol_anchor(word)      # FROZEN: mean of ALL its char wells
             + meaning_residual(word)   # TRAINABLE, init 0 -> meaning forms here

    sentence = mean-pool word_vecs
    pair = u - v  ->  VectorCollapseEngine  ->  cosine to E / N / C   (supervised)

At step 0 every word sits on its char scaffold (residual 0). Training grows the
residual so synonyms drift together and E/N/C separates — meaning forms around
the char structure. Unseen words keep residual 0 and fall back to char position.

Trainable: meaning residual + collapse engine + class anchors. Char scaffold frozen.

Run from collapse_retrain/:  python3 train_nli_meaning_forms_symbols.py
Needs torch + char_typer_symbols.pt + vector_collapse.py.
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_collapse import VectorCollapseEngine
from paths import SNLI_TRAIN

NLI_CKPT = "model_nli_v1/nli_epoch20.pt"
NLI_LABEL_TO_IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


def find_symbol_ckpt():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in ["char_typer_symbols.pt",
              os.path.join(here, "char_typer_symbols.pt"),
              os.path.join(here, "..", "char_typer_symbols.pt")]:
        if os.path.exists(c):
            return c
    return "char_typer_symbols.pt"


def load_vocab():
    try:
        data = torch.load(NLI_CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(NLI_CKPT, map_location="cpu")
    v = data["vocab"]
    return v["idx2word"], {w: i for i, w in enumerate(v["idx2word"])}, v["pad_idx"], v["unk_idx"]


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


def encode_line(line, w2i, unk):
    return [w2i.get(t, unk) for t in line.strip().split() if t] or [unk]


def collate(batch, pad):
    pms, hys, ys = zip(*batch)
    mp = max(len(x) for x in pms); mh = max(len(x) for x in hys)
    f = lambda seqs, m: torch.tensor([s + [pad] * (m - len(s)) for s in seqs], dtype=torch.long)
    return f(pms, mp), f(hys, mh), torch.tensor(ys, dtype=torch.long)


def build_char_scaffold(idx2word, pad_idx):
    """Frozen word positions = mean of the SYMBOL anchors over every char in the word."""
    ck = torch.load(find_symbol_ckpt(), map_location="cpu")
    anchors = F.normalize(ck["symbol_anchors"], dim=-1)
    stoi = ck["stoi"]
    dim = anchors.size(1)
    table = torch.zeros(len(idx2word), dim)
    covered = 0
    for i, w in enumerate(idx2word):
        if i == pad_idx:
            continue
        ids = [stoi[c] for c in w.lower() if c in stoi]   # keeps punctuation & digits
        if ids:
            table[i] = anchors[torch.tensor(ids)].mean(0)
            covered += 1
    print(f"symbol scaffold: {covered}/{len(idx2word)} words placed by their chars  (dim {dim})")
    return table, dim


class MeaningForms(nn.Module):
    def __init__(self, scaffold, pad_idx, dim):
        super().__init__()
        self.char = nn.Embedding.from_pretrained(scaffold, freeze=True, padding_idx=pad_idx)
        self.residual = nn.Embedding(scaffold.size(0), dim, padding_idx=pad_idx)
        nn.init.zeros_(self.residual.weight)            # meaning starts at nothing
        self.pad_idx = pad_idx
        self.engine = VectorCollapseEngine(dim=dim, num_layers=4)

    def pool(self, ids):
        e = self.char(ids) + self.residual(ids)         # scaffold + grown meaning
        m = (ids != self.pad_idx).float().unsqueeze(-1)
        return (e * m).sum(1) / m.sum(1).clamp(min=1.0)

    def anchors(self):
        return torch.stack([
            F.normalize(self.engine.anchor_entail, dim=0),
            F.normalize(self.engine.anchor_neutral, dim=0),
            F.normalize(self.engine.anchor_contra, dim=0),
        ])

    def forward(self, prem, hyp, temp=0.1):
        pair, _ = self.engine(self.pool(prem) - self.pool(hyp))
        return (F.normalize(pair, dim=-1) @ self.anchors().t()) / temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default=SNLI_TRAIN)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.02)
    ap.add_argument("--residual-reg", type=float, default=1e-3,
                    help="L2 pull keeping each word's meaning residual near its char anchor "
                         "(0 = untethered, residual free to wander = can memorize).")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    idx2word, w2i, pad_idx, unk_idx = load_vocab()
    scaffold, dim = build_char_scaffold(idx2word, pad_idx)
    print(f"vocab {len(idx2word)}  dim {dim}  device {device}")
    print("SYMBOL char scaffold FROZEN; meaning residual (init 0) + engine TRAINABLE\n")

    model = MeaningForms(scaffold, pad_idx, dim).to(device)
    data = read_nli_jsonl(args.nli_path, args.max_lines)
    data = [(encode_line(s1, w2i, unk_idx), encode_line(s2, w2i, unk_idx), y) for s1, s2, y in data]
    random.seed(0); random.shuffle(data)
    n_dev = int(len(data) * args.dev_frac)
    dev, train = data[:n_dev], data[n_dev:]
    print(f"train {len(train)}   dev {len(dev)}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def acc(d):
        model.eval(); ok = tot = 0
        with torch.no_grad():
            for i in range(0, len(d), args.batch_size):
                p, h, y = collate(d[i:i + args.batch_size], pad_idx)
                ok += (model(p.to(device), h.to(device)).argmax(-1).cpu() == y).sum().item()
                tot += len(y)
        return ok / max(1, tot)

    best_dev = 0.0
    best_ep = 0
    for ep in range(1, args.epochs + 1):
        model.train(); random.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            p, h, y = collate(train[i:i + args.batch_size], pad_idx)
            logits = model(p.to(device), h.to(device))
            a = model.anchors()
            sep = sum(torch.relu((a[x] * a[z]).sum()) for x, z in [(0, 1), (1, 2), (0, 2)])
            # tether meaning to geometry: penalize how far each residual strays from its anchor (0)
            reg = args.residual_reg * model.residual.weight.pow(2).sum(-1).mean()
            loss = F.cross_entropy(logits, y.to(device)) + sep + reg
            opt.zero_grad(); loss.backward(); opt.step()
            if (i // args.batch_size) % 200 == 0:
                print(f"  ep{ep} step {i//args.batch_size:4d}  ce {loss.item():.4f}")
        tr = acc(train[:20000]); dv = acc(dev)
        rnorm = model.residual.weight.norm(dim=-1).mean().item()   # how far meaning strayed
        flag = ""
        if dv > best_dev:
            best_dev, best_ep = dv, ep
            torch.save({"model": model.state_dict(), "dim": dim,
                        "epoch": ep, "dev": dv}, "nli_meaning_forms_symbols.pt")
            flag = "  <- best, saved"
        print(f"epoch {ep}: train {tr*100:.2f}%   dev {dv*100:.2f}%   resid|.| {rnorm:.3f}{flag}")

    print(f"\nbest dev {best_dev*100:.2f}% at epoch {best_ep}  -> nli_meaning_forms_symbols.pt")


if __name__ == "__main__":
    main()
