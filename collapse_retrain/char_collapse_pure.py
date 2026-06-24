"""
char_collapse_pure.py — CharCollapse with NOTHING but Livnium geometry.

No MLP. No GRU. No learned readout matrix. The ONLY learnable things are:
    - the letter anchors          (the gravity wells, geometry)
    - a start state               (where every word's trajectory begins)
    - one scalar `strength`        (how hard the wells pull)
    - one scalar `temp`            (sharpness of the cosine readout)

ENCODE (pure collapse, one attraction step per letter):
    h <- h - strength * (1 - cos(h, A_c)) * normalize(h - A_c)
This is exactly your VectorCollapseEngine force law with delta = 0 (no MLP warp).

DECODE (pure geometry, no network):
    logits at position i = cos(state_i, EVERY anchor) / temp   -> pick nearest.
The classifier IS the geometry — the same anchors used to pull are the anchors
used to read back. There is no separate decoder.

So this answers one question honestly: with ONLY the geometry, how well can the
collapse trajectory be read back into letters? Compare its number to the
MLP+GRU typer (~85%) — the gap is exactly what the neural parts were buying.

Trained on SNLI words. Needs torch. CPU is fine (no recurrent decoder).
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

NLI_CKPT = "model_nli_v1/nli_epoch20.pt"
SEED = 0
DIM = 64
MAX_WORD = 18             # longest word kept
MAXLEN = 20               # word + EOS, then padded
STEPS_PER_CHAR = 1        # 1 = mirror CharCollapse exactly; raise to let it settle
STEPS = 3000
BATCH = 256
LR = 5e-3
HELDOUT = 1500

LETTERS = "abcdefghijklmnopqrstuvwxyz"
STOI = {c: i + 1 for i, c in enumerate(LETTERS)}   # 0 = PAD
ITOS = {i + 1: c for i, c in enumerate(LETTERS)}
PAD = 0
EOS = 27                  # dedicated STOP well (its own anchor)
N_CHARS = 28
ALPHA = set(LETTERS)


def decode_pred(row):
    """Read letters until the trajectory hits the EOS well (or padding)."""
    out = []
    for t in row:
        t = int(t)
        if t == EOS or t == PAD:
            break
        out.append(ITOS.get(t, "?"))
    return "".join(out)


def load_snli_words():
    try:
        data = torch.load(NLI_CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(NLI_CKPT, map_location="cpu")
    words, seen = [], set()
    for w in data["vocab"]["idx2word"]:
        lw = w.lower()
        if lw and all(c in ALPHA for c in lw) and len(lw) <= MAX_WORD and lw not in seen:
            seen.add(lw)
            words.append(lw)
    return words


def encode_batch(words):
    out = []
    for w in words:
        ids = [STOI[c] for c in w][:MAX_WORD] + [EOS]   # append the STOP symbol
        ids += [PAD] * (MAXLEN - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)


class CharCollapsePure(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.letter_anchors = nn.Parameter(torch.randn(N_CHARS, dim))
        self.start = nn.Parameter(torch.randn(dim) * 0.05)
        self.log_strength = nn.Parameter(torch.tensor(0.0))   # -> sigmoid ~0.5
        self.log_temp = nn.Parameter(torch.tensor(0.0))       # softplus

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
        anchors = F.normalize(self.letter_anchors, dim=-1)     # (N_CHARS, dim)
        mask = (char_ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        states = []
        s = self.strength
        for i in range(L):
            target = anchors[char_ids[:, i]]                   # (B, dim) this letter's well
            m = mask[:, i].float().unsqueeze(-1)
            for _ in range(STEPS_PER_CHAR):
                h_n = F.normalize(h, dim=-1)
                align = (h_n * target).sum(-1)                 # (B,)
                div = 1.0 - align                              # pure attraction
                away = F.normalize(h - target, dim=-1)         # anchor -> h
                step = -s * div.unsqueeze(-1) * away           # NO delta / MLP
                h = h + m * step
                n = h.norm(dim=-1, keepdim=True)
                h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
        return torch.stack(states, dim=1), mask                # (B, L, dim)

    def logits(self, states):
        """Pure-geometry readout: cosine of each state to every anchor."""
        anchors = F.normalize(self.letter_anchors, dim=-1)     # (N_CHARS, dim)
        sn = F.normalize(states, dim=-1)                       # (B, L, dim)
        return (sn @ anchors.t()) / self.temp                  # (B, L, N_CHARS)

    def loss(self, char_ids):
        states, _ = self.encode(char_ids)
        logits = self.logits(states)
        return F.cross_entropy(
            logits.reshape(-1, N_CHARS), char_ids.reshape(-1), ignore_index=PAD
        )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    words = load_snli_words()
    random.shuffle(words)
    test_words = words[:HELDOUT]
    train_words = words[HELDOUT:]
    print(f"alphabetic SNLI words : {len(words)}")
    print(f"train / held-out      : {len(train_words)} / {len(test_words)}")
    print(f"steps per character   : {STEPS_PER_CHAR}   (MLP: NONE, GRU: NONE)\n")

    model = CharCollapsePure()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_ids = encode_batch(train_words)

    model.train()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(train_words), (BATCH,))
        loss = model.loss(train_ids[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  strength {model.strength.item():.3f}")

    model.eval()

    def scores(word_list, cap=3000):
        s = word_list[:cap]
        ids = encode_batch(s)
        with torch.no_grad():
            states, mask = model.encode(ids)
            pred = model.logits(states).argmax(-1)             # (B, L)
        char_ok = ((pred == ids) & mask).sum().item() / mask.sum().item()
        exact = 0
        for k, w in enumerate(s):
            if decode_pred(pred[k]) == w:
                exact += 1
        return char_ok, exact / len(s)

    tr_c, tr_w = scores(train_words)
    te_c, te_w = scores(test_words)
    print("\n--- PURE GEOMETRY (no MLP, no GRU) ---")
    print(f"  trained: per-letter {tr_c*100:5.1f}%   exact-word {tr_w*100:5.1f}%")
    print(f"  held-out: per-letter {te_c*100:5.1f}%   exact-word {te_w*100:5.1f}%")
    print(f"  learned strength {model.strength.item():.3f}   temp {model.temp.item():.3f}")

    print("\n--- per-position decode examples (held-out) ---")
    ids = encode_batch(test_words[:10])
    with torch.no_grad():
        pred = model.logits(model.encode(ids)[0]).argmax(-1)
    for k, w in enumerate(test_words[:10]):
        got = decode_pred(pred[k])
        mark = "OK " if got == w else "XX "
        print(f"  {mark}{w:>14s} -> {got}")

    # The single-point limit: a contraction cannot hold a sequence in one vector.
    print("\n--- single word-vector (mean of trajectory) decoded to ONE anchor ---")
    ids = encode_batch(test_words[:6])
    with torch.no_grad():
        states, mask = model.encode(ids)
        fused = (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        anchors = F.normalize(model.letter_anchors, dim=-1)
        top = (F.normalize(fused, dim=-1) @ anchors.t()).argmax(-1)
    for k, w in enumerate(test_words[:6]):
        print(f"  {w:>14s}  -> nearest single letter: '{ITOS.get(int(top[k]), '?')}'  "
              f"(one point can't unroll a word)")

    # ---- THE ENTIRE LEARNED MODEL: 28 anchor directions + start + 2 scalars ----
    import numpy as np
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- where the knowledge lives ({n_params} numbers total) ---")
    print(f"  letter_anchors {tuple(model.letter_anchors.shape)} = {model.letter_anchors.numel()} numbers  <- ~all of it")
    print(f"  start {tuple(model.start.shape)} = {model.start.numel()}   strength 1   temp 1")
    print(f"  MLP params: 0    GRU params: 0    word table: none")

    A = F.normalize(model.letter_anchors.detach(), dim=-1)
    idxs = list(range(1, 27)) + [EOS]                 # a..z + eos
    labels = list(LETTERS) + ["eos"]
    sub = A[idxs].numpy()
    cosM = np.clip(sub @ sub.T, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosM))
    off = ~np.eye(len(idxs), dtype=bool)
    print("\n--- the learned geometry = angles between the 27 wells ---")
    print(f"  pairwise angle  min {ang[off].min():.1f}deg   mean {ang[off].mean():.1f}deg   max {ang[off].max():.1f}deg")
    pairs = sorted((ang[i, j], labels[i], labels[j])
                   for i in range(len(idxs)) for j in range(i + 1, len(idxs)))
    print("  closest (most confusable) wells:")
    for a, x, y in pairs[:6]:
        print(f"    {x} <-> {y} : {a:.1f}deg")

    torch.save({"letter_anchors": model.letter_anchors.detach(),
                "start": model.start.detach(),
                "strength": model.strength.item(),
                "temp": model.temp.item(),
                "config": {"dim": DIM}}, "char_typer_pure.pt")
    print("\nsaved -> char_typer_pure.pt")

    # 2D map of the wells (the whole model, drawn)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mean = sub.mean(0)
        _, _, Vt = np.linalg.svd(sub - mean, full_matrices=False)
        P = (sub - mean) @ Vt[:2].T
        fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=130)
        fig.patch.set_facecolor("white")
        for (x, y), lab in zip(P, labels):
            eos = lab == "eos"
            ax.scatter([x], [y], s=560, c="#ffe8cc" if eos else "#eef2ff",
                       edgecolors="#d9480f" if eos else "#4f5bd5", linewidths=1.4, zorder=2)
            ax.text(x, y, lab, ha="center", va="center",
                    fontsize=11 if eos else 13, fontweight="bold",
                    color="#d9480f" if eos else "#2a2e6e", zorder=3)
        ax.set_title("Pure CharCollapse — the whole model: 27 learned wells (64d → 2D)",
                     fontsize=11, color="#222")
        ax.set_aspect("equal"); ax.grid(True, ls=":", alpha=0.4)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        fig.tight_layout()
        out = "../docs/images/char_pure_anchor_map.png"
        fig.savefig(out, facecolor="white"); plt.close(fig)
        print(f"saved map -> {out}")
    except ImportError:
        print("(matplotlib not installed — skipped the map; numbers above are the geometry)")


if __name__ == "__main__":
    main()
