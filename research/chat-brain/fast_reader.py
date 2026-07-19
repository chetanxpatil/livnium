"""
fast_reader.py — a SMALL collapse-mimic module, trained in a loop on real data.

The finding (verified numerically): the exact collapse reader is ADDITIVE in
practice — the state at position i is ≈ a geometric-decay weighted sum of the
last few word wells (fitted profile ~ [0.72, 0.20, 0.06, 0.02, ...], dead by
~5 words). A weighted sum is parallel: one causal convolution replaces the
sequential walk.

This script makes that a real trained module:

    student(state_i)  =  sum_k  alpha_k * well(word_{i-k})        (K taps)
    optional refine   :  one exact collapse step from that sum

    teacher           =  the exact sequential collapse (chat_typer geometry,
                         the typer's trained start/strength)

Trained in a LOOP (Adam) on real sentences from data/chat_sentences.txt, real
wells from model/chat_typer.pt. Punished by cosine distance to the teacher's
states at EVERY position (not just the endpoint). Then evaluated honestly:

    * cosine(student state, teacher state) per position, held-out
    * nearest-well decode agreement (do they read the same word back?)
    * order sensitivity (reversed sentences must NOT collapse to the same state)

Output: model/fast_reader.pt  { alpha, taps, refine, metrics }
`chat_reply.py --fast-reader` can then read contexts in parallel.

STANDALONE by default: runs on synthetic geometry (random wells + random
sequences — the exact setup of the verified numpy test), needs no checkpoint
and no data files. Pass --real to fit against model/chat_typer.pt + real
sentences instead.

Usage:
    python3 fast_reader.py                       # rough run, synthetic, self-contained
    python3 fast_reader.py --taps 16 --steps 2000
    python3 fast_reader.py --real                # later: fit on trained wells + real data
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from paths import data_path, model_path

TYPER_CKPT = model_path("chat_typer.pt")
OUT = model_path("fast_reader.pt")
SEED = 0


def load_typer(device):
    ck = torch.load(TYPER_CKPT, map_location=device)
    A = F.normalize(ck["word_anchors"].to(device), dim=-1)
    A[0] = 0.0                                        # PAD contributes nothing
    return (A, ck["start"].to(device), float(ck["strength"]),
            ck["stoi"], ck["config"]["dim"])


def teacher_states(ids, A, start, s):
    """The exact sequential collapse — ground truth. ids (B, L)."""
    B, L = ids.shape
    mask = (ids != 0)
    h = start.expand(B, -1).contiguous()
    states = []
    for i in range(L):
        t = A[ids[:, i]]
        m = mask[:, i].float().unsqueeze(-1)
        align = (F.normalize(h, dim=-1) * t).sum(-1)
        away = F.normalize(h - t, dim=-1)
        h = h + m * (-s * (1.0 - align).unsqueeze(-1) * away)
        n = h.norm(dim=-1, keepdim=True)
        h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        states.append(h)
    return torch.stack(states, dim=1), mask            # (B, L, D), (B, L)


class FastReader(nn.Module):
    """The small module: K trainable taps + optional one collapse refine step."""

    def __init__(self, taps=8, refine=False, strength=0.898):
        super().__init__()
        # init at the collapse's own geometry: alpha_k ~ s * (1-s)^k
        s = strength
        init = torch.tensor([s * (1 - s) ** k for k in range(taps)])
        self.alpha = nn.Parameter(init)
        self.taps, self.refine, self.s = taps, refine, strength

    def forward(self, wells):
        """wells (B, L, D) = A[ids] (PAD rows are zero). Causal conv, parallel."""
        B, L, D = wells.shape
        x = wells.transpose(1, 2)                          # (B, D, L)
        x = F.pad(x, (self.taps - 1, 0))                   # causal left-pad
        w = self.alpha.flip(0).view(1, 1, self.taps).expand(D, 1, self.taps)
        out = F.conv1d(x, w, groups=D).transpose(1, 2)     # (B, L, D)
        if self.refine:                                    # one exact collapse step
            t = F.normalize(wells + 1e-8, dim=-1)
            align = (F.normalize(out + 1e-8, dim=-1) * t).sum(-1)
            away = F.normalize(out - t + 1e-8, dim=-1)
            out = out - self.s * (1.0 - align).unsqueeze(-1) * away
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="fit against model/chat_typer.pt + data/chat_sentences.txt "
                         "(default: synthetic, self-contained)")
    ap.add_argument("--data", default=data_path("chat_sentences.txt"))
    ap.add_argument("--taps", type=int, default=8)
    ap.add_argument("--refine", action="store_true",
                    help="add one exact collapse step after the additive sum")
    ap.add_argument("--n-sents", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=5000,
                    help="hard cap; the loop normally exits on plateau or --seconds")
    ap.add_argument("--seconds", type=float, default=60,
                    help="wall-clock training budget (0 = unlimited, cap = --steps)")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="plateau tolerance on the loss EMA, checked every 50 steps")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    random.seed(SEED); torch.manual_seed(SEED)
    device = (torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    if args.real:
        A, start, s, stoi, dim = load_typer(device)
        print(f"REAL: wells {A.shape[0]}   dim {dim}   teacher strength {s:.3f}   device {device}")
        from chat_typer import encode_batch  # noqa: E402
        unk = len(stoi) + 1; eos = len(stoi) + 2
        sents = []
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                sents.append(line.strip())
                if len(sents) >= args.n_sents:
                    break
        random.shuffle(sents)
        ids = encode_batch(sents, stoi, unk, eos).to(device)
    else:
        # SYNTHETIC: the exact setup of the verified test — random wells,
        # random sequences. Self-contained, no files needed.
        V, dim, s = 5000, 256, 0.898
        A = F.normalize(torch.randn(V, dim, device=device) / dim ** 0.5, dim=-1)
        A[0] = 0.0                                   # PAD
        start = torch.randn(dim, device=device) * 0.05
        maxlen = 34
        ids = torch.zeros(args.n_sents, maxlen, dtype=torch.long, device=device)
        for r in range(args.n_sents):
            L = int(torch.randint(4, maxlen + 1, (1,)))
            ids[r, :L] = torch.randint(1, V, (L,), device=device)
        print(f"SYNTHETIC: wells {V}   dim {dim}   strength {s}   device {device}")
    n_te = max(500, ids.shape[0] // 10)
    te_ids, tr_ids = ids[:n_te], ids[n_te:]
    print(f"sequences: {tr_ids.shape[0]} train / {n_te} held-out")

    model = FastReader(args.taps, args.refine, s).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"module size: {sum(p.numel() for p in model.parameters())} parameters "
          f"({args.taps} taps{', +refine' if args.refine else ''})\n")

    # PRECOMPUTE the teacher once: it is FROZEN (fixed wells/start/strength),
    # so re-walking the sequential collapse every batch just recomputes a
    # constant. One pass here, then every step is only the K-tap conv on 8
    # params — the loop itself costs almost nothing.
    print("precomputing teacher states (one pass) ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        T_all = torch.empty(*tr_ids.shape, A.shape[1], device=device)
        for i in range(0, tr_ids.shape[0], 512):
            T_all[i:i + 512] = F.normalize(
                teacher_states(tr_ids[i:i + 512], A, start, s)[0], dim=-1)
    M_all = (tr_ids != 0)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    # THE LOOP: time-bounded + plateau-stopped; --steps is only a hard cap
    t0 = time.time()
    ema, best_ema, flat, step = None, float("inf"), 0, 0
    while step < args.steps:
        step += 1
        idx = torch.randint(0, tr_ids.shape[0], (args.batch,), device=device)
        S = model(A[tr_ids[idx]])
        cos = (F.normalize(S, dim=-1) * T_all[idx]).sum(-1)               # (B, L)
        m = M_all[idx].float()
        loss = ((1.0 - cos) * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        l = loss.item()
        ema = l if ema is None else 0.9 * ema + 0.1 * l
        if step % 50 == 0 or step == 1:
            print(f"step {step:5d}  1-cos {l:.5f}  ema {ema:.5f}  "
                  f"alpha[:4] {[round(float(a), 3) for a in model.alpha.detach()[:4]]}  "
                  f"| {time.time() - t0:5.1f}s", flush=True)
        if step % 50 == 0:
            if ema < best_ema - args.tol:
                best_ema, flat = ema, 0
            else:
                flat += 1
                if flat >= 4:
                    print(f"  [plateau: ema flat for 200 steps -> stop at step {step}]", flush=True)
                    break
        if args.seconds and time.time() - t0 > args.seconds:
            print(f"  [time budget {args.seconds:.0f}s reached at step {step}]", flush=True)
            break

    # honest eval on held-out
    model.eval()
    with torch.no_grad():
        T, mask = teacher_states(te_ids, A, start, s)
        S = model(A[te_ids])
        cos = (F.normalize(S, dim=-1) * F.normalize(T, dim=-1)).sum(-1)[mask]
        # decode agreement: nearest well of student state vs teacher state.
        # CHUNKED: (B*L, D) @ (D, V) in one shot is ~25 GB at V=100k — do 32
        # sequences at a time, masked positions only.
        agree_n = agree_d = 0
        for i in range(0, te_ids.shape[0], 32):
            mk = mask[i:i + 32]
            mT = F.normalize(T[i:i + 32][mk], dim=-1)
            mS = F.normalize(S[i:i + 32][mk], dim=-1)
            near_T = (mT @ A.t()).argmax(-1)
            near_S = (mS @ A.t()).argmax(-1)
            agree_n += (near_T == near_S).sum().item(); agree_d += near_T.numel()
        agree = agree_n / max(agree_d, 1)
        # order sensitivity: reverse each held-out sentence, endpoints must differ.
        # measured for the STUDENT and the TEACHER — the student can only be as
        # order-sensitive as the collapse it mimics, so judge it against that bar.
        lens = (te_ids != 0).sum(1)
        rev = torch.zeros_like(te_ids)
        for r in range(te_ids.shape[0]):
            L = int(lens[r]) - 1                       # keep EOS at the end
            rev[r, :L] = te_ids[r, :L].flip(0); rev[r, L] = te_ids[r, L]
        Sr = model(A[rev])
        Tr, _ = teacher_states(rev, A, start, s)
        pick = torch.arange(te_ids.shape[0], device=device)
        e1 = F.normalize(S[pick, lens - 1], dim=-1)
        e2 = F.normalize(Sr[pick, lens - 1], dim=-1)
        order_cos = (e1 * e2).sum(-1).mean().item()
        t1 = F.normalize(T[pick, lens - 1], dim=-1)
        t2 = F.normalize(Tr[pick, lens - 1], dim=-1)
        order_cos_teacher = (t1 * t2).sum(-1).mean().item()

    print(f"\n--- FAST READER vs exact collapse (held-out) ---")
    print(f"  state cosine : mean {cos.mean().item():.4f}   min {cos.min().item():.4f}   >0.95: {(cos > 0.95).float().mean().item()*100:.1f}%")
    print(f"  decode agree : {agree*100:.1f}%  (same nearest well as teacher)")
    print(f"  order check  : student reversed-endpoint cos {order_cos:.3f}   "
          f"teacher {order_cos_teacher:.3f}  (student should match the teacher's)")
    print(f"  learned taps : {[round(float(a), 4) for a in model.alpha.detach()]}")

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save({"alpha": model.alpha.detach().cpu(), "taps": args.taps,
                "refine": args.refine, "strength": s,
                "metrics": {"cos_mean": cos.mean().item(), "cos_min": cos.min().item(),
                            "decode_agree": agree, "order_cos": order_cos,
                            "order_cos_teacher": order_cos_teacher}}, OUT)
    print(f"\nsaved -> {OUT}")
    print("verdict: if cosine ~0.99+ and decode agree ~100%, the additive module "
          "faithfully mimics the collapse reader. (Standalone experiment — "
          "integration is a separate decision.)")


if __name__ == "__main__":
    main()
