"""
noun_collapse_pure.py — noun embeddings from raw text, PURE vector collapse.

The occurrence-based task, executed by the one engine instead of PPMI+SVD:
for EVERY OCCURRENCE of a noun, collapse a state through its context words'
wells, and the final state must point at the right noun's well.

    ENCODE (pure collapse, one attraction step per context word):
        h <- h - strength * (1 - cos(h, W_ctx)) * norm(h - W_ctx)
    READOUT (pure geometry, no network):
        logits = cos(h, noun wells) / temp   -> CE against the true noun

Learnable things, in full (the char_collapse_pure.py discipline):
    - one well per vocab word     (context and noun wells are THE SAME table)
    - a start state
    - one scalar strength, one scalar temp
    MLP: none.  SVD: none.  Readout matrix: none.

vs word2vec CBOW: same data signal (co-occurrence), but the context is read
as an ordered trajectory, not a bag — "dog bites man" and "man bites dog"
collapse to different states. Whatever meaning appears in the wells came from
prediction pressure through the geometry alone.

Streaming: examples are built and trained on chunk by chunk — the corpus is
never held in memory. Works on the raw wiki .xml.bz2 via noun_embed's reader.

Usage:
    python3 noun_embed.py --data ... (not needed first; this is standalone)
    python3 noun_collapse_pure.py \
        --data ~/Desktop/test/data-bank/enwiki-latest-pages-articles-multistream.xml.bz2 \
        --max-lines 20000000
    python3 noun_collapse_pure.py --probe cat physics war india

Output: model/noun_collapse_pure.pt { wells, stoi, noun_ids, start, strength, temp }
"""

import argparse
import os
import sys
from collections import Counter

from noun_embed import iter_lines, noun_set
from prep_chat_context import clean

OUT = "model/noun_collapse_pure.pt"
PAD = 0


# ------------------------------------------------------------ data plumbing
# torch-free on purpose: testable without the heavy import.

def build_vocab(args, nouns):
    print("pass 1/2: word frequencies ...", flush=True)
    freq = Counter()
    for line in iter_lines(args.data, args.max_lines, args.sample_parts):
        freq.update(clean(line).split())
    keep = [w for w, c in freq.most_common(args.vocab) if c >= args.min_count]
    stoi = {w: i + 1 for i, w in enumerate(keep)}              # 0 = PAD
    noun_ids = sorted(stoi[w] for w in keep
                      if w in nouns and freq[w] >= args.min_noun_count)
    noun_ids = noun_ids[:args.max_nouns]
    print(f"  vocab {len(stoi):,}   nouns {len(noun_ids):,} "
          f"(min-count {args.min_count}/{args.min_noun_count})", flush=True)
    return stoi, noun_ids, freq


def windows(args, stoi, is_noun, slot=0):
    """Stream (ctx_ids, target_noun_id) for every noun occurrence.
    ctx = W words each side, in order, target replaced by the SLOT well
    (slot=0 disables: old behaviour, hole unmarked)."""
    W = args.window
    for line in iter_lines(args.data, args.max_lines, args.sample_parts):
        ids = [stoi.get(t, 0) for t in clean(line).split()]    # OOV -> PAD (skipped)
        for i, t in enumerate(ids):
            if not is_noun(t):
                continue
            left = [c for c in ids[max(0, i - W):i] if c != PAD]
            right = [c for c in ids[i + 1:i + W + 1] if c != PAD]
            if len(left) + len(right) < args.min_ctx:
                continue
            ctx = left + ([slot] if slot else []) + right
            yield ctx + [PAD] * (2 * W + 1 - len(ctx)), t


# ------------------------------------------------------------ the model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="txt file, folder of .txt, or wiki .xml.bz2")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--max-nouns", type=int, default=50000)
    ap.add_argument("--min-count", type=int, default=10)
    ap.add_argument("--min-noun-count", type=int, default=50,
                    help="a noun needs this many occurrences to earn a target slot")
    ap.add_argument("--min-ctx", type=int, default=3)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--sample-parts", type=int, default=0,
                    help="random-access sampling: probe this many evenly "
                         "spaced multistream blocks in random order (0 = "
                         "front-to-back). Uniform domain coverage.")
    ap.add_argument("--max-occ", type=int, default=0,
                    help="stop training after this many noun occurrences "
                         "(e.g. 100000000 = the 100M budget)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--neg", type=int, default=512,
                    help="sampled-softmax noun negatives (0 = all nouns)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=5000,
                    help="checkpoint every N steps (0 = only at the end)")
    ap.add_argument("--probe", nargs="*", default=None)
    ap.add_argument("--resume", default=None,
                    help="continue training this checkpoint: keeps its vocab, "
                         "wells, start, strength, temp — skips pass 1")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    device = (torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    torch.manual_seed(0)

    # -- probe an existing model and exit
    if args.probe is not None:
        ck = torch.load(args.out, map_location="cpu")
        A = F.normalize(ck["wells"], dim=-1)
        stoi, noun_ids = ck["stoi"], ck["noun_ids"]
        itos = {i: w for w, i in stoi.items()}
        nid = torch.tensor(noun_ids)
        AN = A[nid]
        for w in args.probe:
            if w not in stoi or stoi[w] not in set(noun_ids):
                print(f"  {w:14s} (not a trained noun)")
                continue
            sims = AN @ A[stoi[w]]
            sims[noun_ids.index(stoi[w])] = -1e9
            top = sims.topk(8)
            print(f"  {w:14s} -> " + "  ".join(
                f"{itos[int(nid[i])]}({float(s):.2f})" for s, i in zip(top.values, top.indices)))
        return
    if not args.data:
        sys.exit("need --data (or --probe)")

    # -- vocab + noun targets (resume: reuse the checkpoint's, skip pass 1)
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        stoi, noun_ids = ck["stoi"], ck["noun_ids"]
        print(f"resumed vocab from {args.resume}: {len(stoi):,} words, "
              f"{len(noun_ids):,} nouns", flush=True)
    else:
        nouns = noun_set()
        stoi, noun_ids, freq = build_vocab(args, nouns)
    SLOT = len(stoi) + 1          # the hole gets its own learned well
    V = len(stoi) + 2
    noun_ids_t = torch.tensor(noun_ids, device=device)
    noun_slot = torch.full((V,), -1, dtype=torch.long, device=device)
    noun_slot[noun_ids_t] = torch.arange(len(noun_ids), device=device)
    noun_mask = [False] * V
    for i in noun_ids:
        noun_mask[i] = True

    # -- the whole model: wells + start + 2 scalars (nothing else learns)
    wells = torch.nn.Parameter(torch.randn(V, args.dim, device=device) / args.dim ** 0.5)
    start = torch.nn.Parameter(torch.randn(args.dim, device=device) * 0.05)
    log_strength = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_temp = torch.nn.Parameter(torch.tensor(0.0, device=device))
    if args.resume:
        with torch.no_grad():
            W_old = ck["wells"].to(device)
            wells[:W_old.size(0)].copy_(W_old)   # old run may lack the SLOT row
            start.copy_(ck["start"].to(device))
            s = min(max(ck["strength"], 1e-3), 1 - 1e-3)
            log_strength.copy_(torch.logit(torch.tensor(s, device=device)))
            log_temp.copy_(torch.log(torch.expm1(torch.tensor(
                max(ck["temp"] - 1e-3, 1e-4), device=device))))
    params = [wells, start, log_strength, log_temp]
    print(f"pure model: wells {V:,} x {args.dim}  + start + strength + temp   "
          f"({sum(p.numel() for p in params):,} numbers)   device {device}", flush=True)

    def encode(ctx_ids):
        """Pure collapse through the ordered context. ctx_ids (B, L)."""
        A = F.normalize(wells, dim=-1)
        mask = ctx_ids != PAD
        h = start.expand(ctx_ids.size(0), -1).contiguous()
        s = torch.sigmoid(log_strength)
        for i in range(ctx_ids.size(1)):
            t = A[ctx_ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            align = (F.normalize(h, dim=-1) * t).sum(-1)
            away = F.normalize(h - t, dim=-1)
            h = h + m * (-s * (1.0 - align).unsqueeze(-1) * away)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
        return h

    def loss_fn(ctx_ids, tgt_ids):
        h = F.normalize(encode(ctx_ids), dim=-1)
        A = F.normalize(wells, dim=-1)
        temp = F.softplus(log_temp) + 1e-3
        if args.neg > 0:                       # 1 true noun + K noun negatives
            pos = (h * A[tgt_ids]).sum(-1, keepdim=True) / temp
            neg = noun_ids_t[torch.randint(0, len(noun_ids), (args.neg,), device=device)]
            ng = (h @ A[neg].t()) / temp
            # mask false negatives: a sampled negative that equals the true
            # target must not be penalized as a negative
            ng = ng.masked_fill(neg.unsqueeze(0) == tgt_ids.unsqueeze(1), float("-inf"))
            cand = torch.cat([pos, ng], dim=1)
            return F.cross_entropy(cand, torch.zeros(cand.size(0), dtype=torch.long,
                                                     device=device))
        logits = (h @ A[noun_ids_t].t()) / temp     # full noun-vocab CE
        return F.cross_entropy(logits, noun_slot[tgt_ids])

    # -- streaming train: one pass, chunk by chunk
    def save():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"wells": wells.detach().cpu(), "stoi": stoi, "noun_ids": noun_ids,
                    "start": start.detach().cpu(),
                    "strength": torch.sigmoid(log_strength).item(),
                    "temp": (F.softplus(log_temp) + 1e-3).item(),
                    "config": {"dim": args.dim, "window": args.window,
                               "slot": SLOT}}, args.out)

    opt = torch.optim.Adam(params, lr=args.lr)
    is_noun = lambda t: noun_mask[t]  # noqa: E731
    cbuf, tbuf, step, seen = [], [], 0, 0
    import time
    t0 = time.time()
    for ctx, tgt in windows(args, stoi, is_noun, slot=SLOT):
        cbuf.append(ctx); tbuf.append(tgt); seen += 1
        if len(cbuf) < args.batch:
            continue
        ctx_ids = torch.tensor(cbuf, dtype=torch.long, device=device)
        tgt_ids = torch.tensor(tbuf, dtype=torch.long, device=device)
        cbuf, tbuf = [], []
        loss = loss_fn(ctx_ids, tgt_ids)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:6d}  loss {loss.item():.4f}  "
                  f"strength {torch.sigmoid(log_strength).item():.3f}  "
                  f"occurrences {seen:,}  | {time.time() - t0:.0f}s", flush=True)
        if args.save_every and step % args.save_every == 0:
            save()
            print(f"  [checkpoint -> {args.out}]", flush=True)
        if args.max_occ and seen >= args.max_occ:
            print(f"  [occurrence budget {args.max_occ:,} reached]", flush=True)
            break
    print(f"done: {seen:,} noun occurrences, {step:,} steps", flush=True)
    save()
    print(f"saved -> {args.out}")
    print("probe it:  python3 noun_collapse_pure.py --probe cat physics war india")


if __name__ == "__main__":
    main()
