"""train_collapse.py — collapse embeddings on the frozen corpus, seeded.

Same model as chat/noun_collapse_pure.py (wells + start + 2 scalars, v1 chord
step, sampled-softmax over noun targets) with three harness additions:

  --variant v1|v2   v1 = legacy, NO false-negative mask (matches the published
                    checkpoint's objective); v2 = masked negatives (current).
                    v1 and v2 results are kept separate all the way through.
  --seed N          seeds init AND negative sampling (the published run was
                    seed 0 only).
  resume            checkpoints carry the occurrence counter + torch RNG state;
                    on resume the deterministic window stream is fast-forwarded
                    past the already-trained occurrences.

Output: work/models/collapse_{variant}_seed{K}.npz  {words, vectors} + lineage.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "research", "embeddings", "noun-collapse"))
from common import WORK, caffeinate, corpus_lines, load_vocab, save_json, stamp  # noqa: E402


def windows(corpus, stoi, is_noun, W, min_ctx, slot):
    """(ctx_ids, target) per noun occurrence — same shape as noun_collapse_pure."""
    for toks in corpus_lines(corpus):
        ids = [stoi.get(t, 0) for t in toks]
        for i, t in enumerate(ids):
            if not is_noun(t):
                continue
            left = [c for c in ids[max(0, i - W):i] if c]
            right = [c for c in ids[i + 1:i + W + 1] if c]
            if len(left) + len(right) < min_ctx:
                continue
            ctx = left + [slot] + right
            yield ctx + [0] * (2 * W + 1 - len(ctx)), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v1", "v2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-ctx", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--neg", type=int, default=512)
    ap.add_argument("--max-occ", type=int, default=0, help="occurrence budget (0 = one full pass)")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()
    caffeinate()

    import torch
    import torch.nn.functional as F

    device = (torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    voc = load_vocab()
    stoi, noun_ids = voc["stoi"], voc["noun_ids"]
    corpus = os.path.join(WORK, "corpus.txt")
    tag = f"collapse_{args.variant}_seed{args.seed}"
    out_npz = os.path.join(WORK, "models", f"{tag}.npz")
    ck_path = os.path.join(WORK, "models", f"{tag}.ckpt.pt")
    os.makedirs(os.path.join(WORK, "models"), exist_ok=True)
    if os.path.exists(out_npz):
        print(f"{out_npz} exists — done (delete to retrain)")
        return

    SLOT = len(stoi) + 1
    V = len(stoi) + 2
    torch.manual_seed(args.seed)
    wells = torch.nn.Parameter(torch.randn(V, args.dim, device=device) / args.dim ** 0.5)
    start = torch.nn.Parameter(torch.randn(args.dim, device=device) * 0.05)
    log_strength = torch.nn.Parameter(torch.tensor(0.0, device=device))
    log_temp = torch.nn.Parameter(torch.tensor(0.0, device=device))
    params = [wells, start, log_strength, log_temp]
    opt = torch.optim.Adam(params, lr=args.lr)
    noun_ids_t = torch.tensor(noun_ids, device=device)
    noun_mask = [False] * V
    for i in noun_ids:
        noun_mask[i] = True

    done_occ = 0
    if os.path.exists(ck_path):                     # ---- resume
        ck = torch.load(ck_path, map_location=device)
        with torch.no_grad():
            wells.copy_(ck["wells"]); start.copy_(ck["start"])
            log_strength.copy_(ck["log_strength"]); log_temp.copy_(ck["log_temp"])
        opt.load_state_dict(ck["opt"])
        torch.set_rng_state(ck["rng"].cpu())
        done_occ = ck["seen"]
        print(f"resumed at {done_occ:,} occurrences", flush=True)

    def encode(ctx_ids):
        A = F.normalize(wells, dim=-1)
        mask = ctx_ids != 0
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
        pos = (h * A[tgt_ids]).sum(-1, keepdim=True) / temp
        neg = noun_ids_t[torch.randint(0, len(noun_ids), (args.neg,), device=device)]
        ng = (h @ A[neg].t()) / temp
        if args.variant == "v2":                    # masked false negatives
            ng = ng.masked_fill(neg.unsqueeze(0) == tgt_ids.unsqueeze(1), float("-inf"))
        cand = torch.cat([pos, ng], dim=1)
        return F.cross_entropy(cand, torch.zeros(cand.size(0), dtype=torch.long, device=device))

    def save_ckpt(seen):
        torch.save({"wells": wells.detach(), "start": start.detach(),
                    "log_strength": log_strength.detach(), "log_temp": log_temp.detach(),
                    "opt": opt.state_dict(), "rng": torch.get_rng_state(),
                    "seen": seen}, ck_path)

    is_noun = lambda t: noun_mask[t]  # noqa: E731
    cbuf, tbuf, step, seen = [], [], 0, 0
    import time
    t0 = time.time()
    for ctx, tgt in windows(corpus, stoi, is_noun, args.window, args.min_ctx, SLOT):
        seen += 1
        if seen <= done_occ:                        # fast-forward on resume
            continue
        cbuf.append(ctx); tbuf.append(tgt)
        if len(cbuf) < args.batch:
            continue
        ctx_ids = torch.tensor(cbuf, dtype=torch.long, device=device)
        tgt_ids = torch.tensor(tbuf, dtype=torch.long, device=device)
        cbuf, tbuf = [], []
        loss = loss_fn(ctx_ids, tgt_ids)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            print(f"{tag}  step {step:6d}  loss {loss.item():.4f}  "
                  f"occ {seen:,}  | {time.time() - t0:.0f}s", flush=True)
        if args.save_every and step % args.save_every == 0:
            save_ckpt(seen)
        if args.max_occ and seen >= args.max_occ:
            break

    # ---- export {words, vectors} for the shared evaluator
    import numpy as np
    itos = sorted(stoi, key=stoi.get)
    W = wells.detach().cpu().numpy()[1:len(stoi) + 1]           # drop PAD + SLOT rows
    np.savez_compressed(out_npz, words=np.array(itos), vectors=W.astype(np.float32))
    save_json(out_npz.replace(".npz", ".meta.json"), {
        **stamp(), "model": "collapse", "variant": args.variant, "seed": args.seed,
        "dim": args.dim, "window": args.window, "neg": args.neg, "lr": args.lr,
        "occurrences_trained": seen, "occurrence_budget": args.max_occ,
        "false_negative_mask": args.variant == "v2",
    })
    if os.path.exists(ck_path):
        os.remove(ck_path)
    print(f"saved -> {out_npz}  ({seen:,} occurrences)")


if __name__ == "__main__":
    main()
