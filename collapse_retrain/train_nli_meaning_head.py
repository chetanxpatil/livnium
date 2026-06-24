"""
train_nli_meaning_head.py — STAGE 2: attach MEANING to the frozen typer.

Stage 1 (train_snli_typer.py) taught CharCollapse the STRUCTURE of words
(letters + order) with no meaning. This stage freezes that substrate and learns
meaning on top, supervised by SNLI E/N/C labels.

    word --(FROZEN CharCollapse)--> spelling-vector   [fixed]
                                          |
                                   MeaningHead (trainable)  <-- learns meaning
                                          |
            premise pool u , hypothesis pool v  ->  pair = u - v
                                          |
                          VectorCollapseEngine (trainable, fresh)
                                          |
                            cosine to E / N / C anchors  ->  logits  ->  CE

The frozen typer never moves: spelling structure + zero-OOV stay intact. Only
the meaning head, the collapse engine, and the E/N/C anchors learn. This is the
"learn structure first, attach meaning later, by supervised learning" plan.

Usage:
    python3 train_nli_meaning_head.py --nli-path /path/to/snli_1.0_train.jsonl \\
                                      --dev-path  /path/to/snli_1.0_dev.jsonl

The vocab + tokenization are taken from the trained NLI checkpoint so they match
your existing pipeline. Needs torch + char_typer.pt from stage 1.
"""

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from char_collapse import CharCollapse
from vector_collapse import VectorCollapseEngine

NLI_CKPT = "model_nli_v1/nli_epoch20.pt"
TYPER_CKPT = "char_typer.pt"
NLI_LABEL_TO_IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


# --- vocab / data (mirrors train_collapse_embeddings.py) ----------------------

def load_vocab():
    try:
        data = torch.load(NLI_CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(NLI_CKPT, map_location="cpu")
    v = data["vocab"]
    idx2word = v["idx2word"]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return idx2word, word2idx, v["pad_idx"], v["unk_idx"]


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


def encode_line(line, word2idx, unk_idx):
    return [word2idx.get(t, unk_idx) for t in line.strip().split() if t] or [unk_idx]


def collate(batch, pad_idx):
    pms, hys, ys = zip(*batch)
    mp = max(len(x) for x in pms)
    mh = max(len(x) for x in hys)
    pad = lambda seqs, m: torch.tensor([s + [pad_idx] * (m - len(s)) for s in seqs], dtype=torch.long)
    return pad(pms, mp), pad(hys, mh), torch.tensor(ys, dtype=torch.long)


# --- build the FROZEN typer-derived word vectors ------------------------------

def build_frozen_embeddings(idx2word, pad_idx, device):
    """Run the frozen CharCollapse on every vocab word ONCE to make a fixed
    embedding table. The table literally IS the typer's output."""
    ck = torch.load(TYPER_CKPT, map_location="cpu")
    cfg = ck["config"]
    enc = CharCollapse(dim=cfg["dim"], max_len=cfg["max_len"])
    enc.load_state_dict(ck["char_collapse"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.to(device)

    alpha = set("abcdefghijklmnopqrstuvwxyz")
    dim = cfg["dim"]
    table = torch.zeros(len(idx2word), dim)
    B = 1024
    buf_words, buf_rows = [], []

    def flush():
        if not buf_words:
            return
        ids = enc.vocab.encode_batch(buf_words, cfg["max_len"]).to(device)
        with torch.no_grad():
            _, _, fused, _ = enc.encode(ids)
        for r, vec in zip(buf_rows, fused.cpu()):
            table[r] = vec
        buf_words.clear()
        buf_rows.clear()

    for i, w in enumerate(idx2word):
        lw = w.lower()
        clean = "".join(c for c in lw if c in alpha)
        if i == pad_idx or not clean:
            continue  # pad + non-alphabetic tokens stay zero (learned by head bias)
        buf_words.append(clean)
        buf_rows.append(i)
        if len(buf_words) >= B:
            flush()
    flush()
    return table, dim


# --- model --------------------------------------------------------------------

class MeaningNLI(nn.Module):
    def __init__(self, frozen_table, pad_idx, dim):
        super().__init__()
        # frozen spelling vectors from the typer (never updated)
        self.embed = nn.Embedding.from_pretrained(frozen_table, freeze=True, padding_idx=pad_idx)
        self.pad_idx = pad_idx
        # trainable meaning head: maps spelling-space -> meaning-space
        self.meaning = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim)
        )
        # fresh, trainable collapse engine + E/N/C anchors
        self.engine = VectorCollapseEngine(dim=dim, num_layers=4)

    def pool(self, ids):
        e = self.meaning(self.embed(ids))                 # attach meaning per word
        mask = (ids != self.pad_idx).float().unsqueeze(-1)
        return (e * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def anchors(self):
        return torch.stack([
            F.normalize(self.engine.anchor_entail, dim=0),
            F.normalize(self.engine.anchor_neutral, dim=0),
            F.normalize(self.engine.anchor_contra, dim=0),
        ])

    def forward(self, prem, hyp, temp=0.1):
        u, v = self.pool(prem), self.pool(hyp)
        pair, _ = self.engine(u - v)
        pair_n = F.normalize(pair, dim=-1)
        return (pair_n @ self.anchors().t()) / temp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", required=True, help="SNLI/MultiNLI train .jsonl")
    ap.add_argument("--dev-path", default="", help="SNLI/MultiNLI dev .jsonl (optional)")
    ap.add_argument("--dev-frac", type=float, default=0.02,
                    help="If no --dev-path, hold out this fraction of train as dev.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--lambda-sep", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    idx2word, word2idx, pad_idx, unk_idx = load_vocab()
    print(f"vocab: {len(idx2word)} tokens   device: {device}")

    table, dim = build_frozen_embeddings(idx2word, pad_idx, device)
    nonzero = int((table.abs().sum(1) > 0).sum())
    print(f"frozen typer embeddings built: {nonzero}/{len(idx2word)} words have char-built vectors")

    model = MeaningNLI(table, pad_idx, dim).to(device)
    # sanity: typer embeddings must be frozen, everything else trainable
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"trainable modules: meaning head + collapse engine ({len(trainable)} tensors); "
          f"typer embeddings frozen\n")

    import random
    train = read_nli_jsonl(args.nli_path, args.max_lines)
    train = [(encode_line(s1, word2idx, unk_idx), encode_line(s2, word2idx, unk_idx), y)
             for s1, s2, y in train]
    dev = []
    if args.dev_path:
        dev = read_nli_jsonl(args.dev_path)
        dev = [(encode_line(s1, word2idx, unk_idx), encode_line(s2, word2idx, unk_idx), y)
               for s1, s2, y in dev]
    elif args.dev_frac > 0:
        # No dev file: hold out a slice of train so we still get a held-out number.
        random.seed(0)
        random.shuffle(train)
        n_dev = int(len(train) * args.dev_frac)
        dev, train = train[:n_dev], train[n_dev:]
        print(f"no dev file -> held out {args.dev_frac*100:.0f}% of train as dev")
    print(f"train examples: {len(train)}")
    if dev:
        print(f"dev examples  : {len(dev)}")

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    def run_eval(data):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for i in range(0, len(data), args.batch_size):
                p, h, y = collate(data[i:i + args.batch_size], pad_idx)
                logits = model(p.to(device), h.to(device))
                correct += (logits.argmax(-1).cpu() == y).sum().item()
                total += len(y)
        return correct / max(1, total)

    for ep in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), args.batch_size):
            p, h, y = collate(train[i:i + args.batch_size], pad_idx)
            logits = model(p.to(device), h.to(device))
            ce = F.cross_entropy(logits, y.to(device))
            a = model.anchors()
            sep = torch.relu((a[0] * a[1]).sum()) + torch.relu((a[1] * a[2]).sum()) + \
                  torch.relu((a[0] * a[2]).sum())
            loss = ce + args.lambda_sep * sep
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += ce.item()
            if (i // args.batch_size) % 100 == 0:
                print(f"  ep{ep} step {i//args.batch_size:4d}  ce {ce.item():.4f}")
        tr_acc = run_eval(train[:20000])
        msg = f"epoch {ep}: train-acc {tr_acc*100:.2f}%"
        if dev:
            msg += f"   dev-acc {run_eval(dev)*100:.2f}%"
        print(msg)

    torch.save({"model": model.state_dict(), "dim": dim}, "nli_meaning_head.pt")
    print("\nsaved -> nli_meaning_head.pt")


if __name__ == "__main__":
    main()
