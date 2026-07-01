"""
chat_reply.py — the GENERATOR: your message -> TYPE the reply, on the typer's wells.

This is premise_from_hyp.py ported to your chats, with the same simplifications
that worked for the typer:
  * NO labels. NLI had (hypothesis + label) -> premise; chat has just
    (your message) -> reply. label_emb, the margin loss, and the generative
    classifier are gone.
  * NO char stage. Word wells warm-start from model/chat_typer.pt — the wells
    that already type your chats at ~100% clean — and keep training.
  * ALIGN is ON by default. On SNLI, per-step attention over the input words was
    the best model (+8.6 points). Here it is the 'reasoning': every reply word is
    picked while looking at a specific word of YOUR message, and the trace of
    what it looked at is printed as 'thinking'. --no-align for the baseline.

Mechanism (unchanged collapse engine):
    z      = think(meanpool(message word wells))            # the thought
    h0     = z
    step t : ctx   = attend(h -> message words)             # what it's looking at
             query = brain([h ; z ; ctx]) -> cos(query, wells)/temp -> next word
             punish with CE vs the true reply word
             h <- collapse(h, well[true reply word])        # teacher forcing

Context (v2, session-aware): trains on data/chat_context.tsv from
prep_chat_context.py — each example is the last K turns of a real conversation,
speaker-tagged <you>/<me> (fresh trainable wells), tail-truncated so the newest
words always survive. --chat is multi-turn: it feeds the live conversation back
as context, the same shape it trained on.

Usage:
    python3 prep_chat_context.py                   # once: raw export -> data/chat_context.tsv
    python3 chat_reply.py                          # train (align on, warm wells)
    python3 chat_reply.py --chat                   # talk to it (multi-turn)
    python3 chat_reply.py --no-align               # baseline: pooled thought only
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from chat_typer import encode_batch, MAXLEN, PAD, SEED

TYPER_CKPT = "model/chat_typer.pt"
CKPT_OUT = "model/chat_reply.pt"
SPECIALS = ["<you>", "<me>"]      # speaker wells: minted fresh, trained like any word
CTX_WORDS = 48                    # context budget (oldest words drop first)


class ReplyBrain(nn.Module):
    """Generates a reply from your message via collapse-typing. No labels, no char.

    READER (v3): the context is read the same way sentences are written — by
    COLLAPSING through it word by word (the typer's encode). The trajectory of
    states is the sentence-aware memory: state_i = 'the conversation up to word
    i'. z = think(final state), and alignment attends over the trajectory, not
    over static word wells. reader='meanpool' keeps the old bag-of-words."""

    def __init__(self, n_words, dim, eos, warm=None, hidden=512, align=True,
                 reader="collapse", warm_start=None, warm_strength=None):
        super().__init__()
        self.eos, self.dim, self.n_words = eos, dim, n_words
        self.align = align
        self.reader = reader
        self.word_anchors = nn.Parameter(torch.randn(n_words, dim) * (1.0 / dim ** 0.5))
        if warm is not None:
            with torch.no_grad():
                self.word_anchors.copy_(warm); self.word_anchors[PAD].zero_()
        # reader params: the typer's start vector + its learned collapse strength
        self.start = nn.Parameter(torch.randn(dim) * 0.05)
        if warm_start is not None:
            with torch.no_grad():
                self.start.copy_(warm_start)
        s0 = 2.2 if warm_strength is None else float(
            torch.logit(torch.tensor(min(max(warm_strength, 1e-3), 1 - 1e-3))))
        self.log_strength_read = nn.Parameter(torch.tensor(s0))
        self.think = nn.Linear(dim, dim)                 # read state -> thought
        n_parts = 2 + (1 if align else 0)                # [h ; z] (+ aligned ctx)
        self.brain = nn.Sequential(nn.Linear(n_parts * dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, dim))
        if align:
            self.att_key = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
            self.att_query = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.neg_samples = 0        # >0 => sampled-softmax CE in TRAINING only

    @property
    def strength(self):
        return torch.sigmoid(self.log_strength)

    @property
    def temp(self):
        return F.softplus(self.log_temp) + 1e-3

    @property
    def strength_read(self):
        return torch.sigmoid(self.log_strength_read)

    def meanpool(self, ids, A):
        m = (ids != PAD).float().unsqueeze(-1)
        return (A[ids] * m).sum(1) / m.sum(1).clamp(min=1.0)

    def read(self, msg_ids, A):
        """Read the context by collapsing through it (the typer's encode).
        Returns (normalized trajectory states, mask, final state)."""
        B, L = msg_ids.shape
        mask = (msg_ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        s = self.strength_read
        states = []
        for i in range(L):
            target = A[msg_ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            align = (F.normalize(h, dim=-1) * target).sum(-1)
            away = F.normalize(h - target, dim=-1)
            h = h + m * (-s * (1.0 - align).unsqueeze(-1) * away)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
        S = torch.stack(states, dim=1)                 # (B, L, dim) the trajectory
        return F.normalize(S, dim=-1), mask, h

    def read_context(self, msg_ids, A):
        """Both readers behind one door: returns (memory, mask, thought-input).
        collapse: memory = trajectory states (order-aware)
        meanpool: memory = static word wells   (bag-of-words baseline)"""
        if self.reader == "collapse":
            S, mask, hT = self.read(msg_ids, A)
            return S, mask, hT
        return A[msg_ids], (msg_ids != PAD), self.meanpool(msg_ids, A)

    def collapse_step(self, h, target):
        align = (F.normalize(h, dim=-1) * target).sum(-1)
        away = F.normalize(h - target, dim=-1)
        h = h - self.strength * (1.0 - align).unsqueeze(-1) * away
        n = h.norm(dim=-1, keepdim=True)
        return torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)

    def reply_nll(self, msg_ids, rep_ids, reduce_mean=True):
        """Teacher-forced NLL of the reply. This is the punishment.
        SELF-ATTEND: each typed word's collapse state joins the memory, so the
        writer can look back at its own words, not just the context."""
        A = F.normalize(self.word_anchors, dim=-1)
        EM, mmask, hread = self.read_context(msg_ids, A)   # memory = trajectory (or wells)
        z = self.think(hread)
        h = z
        if self.align:
            K = self.att_key(EM)
            mem, kmem, vmask = EM, K, mmask                # growing memory
        B, L = rep_ids.shape
        tok_nll = torch.zeros(B, device=rep_ids.device)
        tok_cnt = torch.zeros(B, device=rep_ids.device)
        sampled = self.training and self.neg_samples > 0
        for t in range(L):
            hn = F.normalize(h, dim=-1)
            parts = [hn, z]
            if self.align:
                q = self.att_query(hn).unsqueeze(1)                        # (B,1,dim)
                scores = torch.bmm(q, kmem.transpose(1, 2)).squeeze(1)
                scores = scores.masked_fill(~vmask, -1e9)
                attn = torch.softmax(scores, dim=-1).unsqueeze(1)
                parts.append(torch.bmm(attn, mem).squeeze(1))              # aligned ctx
            query = F.normalize(self.brain(torch.cat(parts, dim=-1)), dim=-1)
            tgt = rep_ids[:, t]
            if sampled:
                pos = (query * A[tgt]).sum(-1, keepdim=True) / self.temp
                neg_ids = torch.randint(1, self.n_words, (self.neg_samples,),
                                        device=rep_ids.device)
                neg = (query @ A[neg_ids].t()) / self.temp
                cand = torch.cat([pos, neg], dim=1)
                ll = F.cross_entropy(cand, torch.zeros(cand.size(0), dtype=torch.long,
                                                       device=rep_ids.device),
                                     reduction="none")
            else:
                logits = (query @ A.t()) / self.temp
                ll = F.cross_entropy(logits, tgt, reduction="none")
            mask = (tgt != PAD).float()
            tok_nll += ll * mask; tok_cnt += mask
            h = self.collapse_step(h, A[tgt])                              # teacher forcing
            if self.align:                                 # the word just typed becomes
                hs = F.normalize(h, dim=-1).unsqueeze(1)   # part of the memory
                mem = torch.cat([mem, hs], dim=1)
                kmem = torch.cat([kmem, self.att_key(hs)], dim=1)
                vmask = torch.cat([vmask, (tgt != PAD).unsqueeze(1)], dim=1)
        if reduce_mean:
            return (tok_nll / tok_cnt.clamp(min=1)).mean()
        return tok_nll, tok_cnt

    @torch.no_grad()
    def generate(self, msg_ids, max_len, unk=None, ban=()):
        """Free-running greedy decode: the model types its own reply, feeding each
        choice back into the collapse. Returns token ids and, with align, the
        message-word index it leaned on at each step (the 'thinking')."""
        A = F.normalize(self.word_anchors, dim=-1)
        EM, mmask, hread = self.read_context(msg_ids, A)   # memory = trajectory (or wells)
        z = self.think(hread)
        h = z
        if self.align:
            mem, kmem, vmask = EM, self.att_key(EM), mmask
        B = msg_ids.size(0)
        toks, attn = [], []
        done = torch.zeros(B, dtype=torch.bool, device=msg_ids.device)
        for _ in range(max_len):
            hn = F.normalize(h, dim=-1)
            parts = [hn, z]
            if self.align:
                q = self.att_query(hn).unsqueeze(1)
                scores = torch.bmm(q, kmem.transpose(1, 2)).squeeze(1).masked_fill(~vmask, -1e9)
                a = torch.softmax(scores, dim=-1)
                attn.append(a.argmax(-1))
                parts.append(torch.bmm(a.unsqueeze(1), mem).squeeze(1))
            query = self.brain(torch.cat(parts, dim=-1))
            logits = (F.normalize(query, dim=-1) @ A.t()) / self.temp
            logits[:, PAD] = float("-inf")
            if unk is not None:
                logits[:, unk] = float("-inf")
            for b in ban:                      # e.g. never type <you>/<me>
                logits[:, b] = float("-inf")
            nxt = logits.argmax(-1)
            toks.append(nxt)
            h = self.collapse_step(h, A[nxt])
            if self.align:                                 # own words join the memory
                hs = F.normalize(h, dim=-1).unsqueeze(1)
                mem = torch.cat([mem, hs], dim=1)
                kmem = torch.cat([kmem, self.att_key(hs)], dim=1)
                vmask = torch.cat([vmask, (~done).unsqueeze(1)], dim=1)
            done = done | (nxt == self.eos)
            if bool(done.all()):
                break
        T = torch.stack(toks, dim=1)
        att = torch.stack(attn, dim=1) if (self.align and attn) else None
        return T, att


def load_wells(device, path=TYPER_CKPT):
    """Typer wells + fresh wells for the speaker tokens (<you>, <me>)."""
    ck = torch.load(path, map_location=device)
    warm, stoi, itos = ck["word_anchors"].to(device), dict(ck["stoi"]), dict(ck["itos"])
    n_words, dim = ck["config"]["n_words"], ck["config"]["dim"]
    for tok in SPECIALS:
        stoi[tok] = n_words; itos[n_words] = tok
        warm = torch.cat([warm, torch.randn(1, dim, device=device) / dim ** 0.5], dim=0)
        n_words += 1
    extras = {"start": ck.get("start"), "strength": ck.get("strength")}
    return warm, stoi, itos, ck["unk"], ck["eos"], n_words, dim, extras


def encode_ctx(sents, stoi, unk, eos, ctx_words=CTX_WORDS):
    """Context encoder: keep the LAST ctx_words tokens (newest survive), + EOS."""
    maxlen = ctx_words + 2
    out = []
    for s in sents:
        ids = [stoi.get(t, unk) for t in s.split()][-ctx_words:] + [eos]
        ids += [PAD] * (maxlen - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)


def read_pairs(path, max_lines=0):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_lines and len(out) >= max_lines:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                out.append((parts[0], parts[1]))
    return out


def trace_str(reply_words, att_row, ctx_words, Lm):
    """The thinking trace. Each typed word shows what it looked at:
    a context word, <eos>, or ~word = a word the model itself already typed."""
    out = []
    for j, w in enumerate(reply_words):
        hi = int(att_row[j])
        if hi < len(ctx_words):
            look = ctx_words[hi]
        elif hi < Lm:
            look = "<eos>"
        else:
            k = hi - Lm
            look = "~" + (reply_words[k] if 0 <= k < j else "?")
        out.append(f"{w}<-{look}")
    return " ".join(out)


def decode(ids, itos, eos):
    out = []
    for t in ids.tolist():
        if t == eos or t == PAD:
            break
        out.append(itos.get(t, "?"))
    return " ".join(out) if out else "(empty)"


def chat_loop(args, device):
    """--chat: talk to the trained generator, MULTI-TURN. The conversation so far
    is fed back as <you>/<me>-tagged context — the same shape it trained on."""
    ck = torch.load(args.ckpt, map_location=device)
    cfg = ck["config"]
    stoi, itos, unk, eos = ck["stoi"], ck["itos"], ck["unk"], ck["eos"]
    ctx_words = cfg.get("ctx_words", CTX_WORDS)
    model = ReplyBrain(cfg["n_words"], cfg["dim"], eos, align=cfg["align"],
                       reader=cfg.get("reader", "meanpool")).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ban = [stoi[t] for t in SPECIALS if t in stoi]
    print(f"loaded {args.ckpt}   align={cfg['align']}   ctx {ctx_words} words   device {device}")
    print("multi-turn: it remembers this conversation. :reset to wipe, :q to quit\n")
    import re
    clean = re.compile(r"[^a-z0-9' ]+")
    history = []                                   # [(tag, text), ...] this session
    while True:
        try:
            line = input("you   > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line == ":q":
            break
        if line == ":reset":
            history = []; print("  (context wiped)\n"); continue
        line = re.sub(r"\s+", " ", clean.sub(" ", line)).strip()
        if not line:
            continue
        history.append(("<you>", line))
        toks = []
        for tag, text in history:
            toks.append(tag); toks += text.split()
        ctx = " ".join(toks[-ctx_words:])
        ids = encode_ctx([ctx], stoi, unk, eos, ctx_words).to(device)
        gen, att = model.generate(ids, MAXLEN, unk=unk, ban=ban)
        reply = decode(gen[0], itos, eos)
        history.append(("<me>", reply))
        print(f"model > {reply}")
        if att is not None:
            print(f"  thinking: {trace_str(reply.split(), att[0], ctx.split(), ids.shape[1])}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/chat_context.tsv")
    ap.add_argument("--ctx-words", type=int, default=CTX_WORDS,
                    help="context budget; must match prep_chat_context.py --ctx-words")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--no-align", action="store_true",
                    help="baseline: pooled thought only, no per-step attention")
    ap.add_argument("--meanpool", action="store_true",
                    help="baseline reader: bag-of-words meanpool instead of the "
                         "order-aware collapse trajectory")
    ap.add_argument("--neg-samples", type=int, default=0,
                    help="sampled-softmax negatives in training (0=off, full vocab)")
    ap.add_argument("--ckpt", default=CKPT_OUT)
    ap.add_argument("--chat", action="store_true", help="talk to the trained model")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()
    align = not args.no_align

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    random.seed(SEED); torch.manual_seed(SEED)

    if args.chat:
        chat_loop(args, device)
        return

    reader = "meanpool" if args.meanpool else "collapse"
    print("loading typer wells ...", flush=True)
    warm, stoi, itos, unk, eos, n_words, dim, extras = load_wells(device)
    model = ReplyBrain(n_words, dim, eos, warm=warm, align=align, reader=reader,
                       warm_start=extras["start"].to(device) if extras["start"] is not None else None,
                       warm_strength=extras["strength"]).to(device)
    model.neg_samples = args.neg_samples
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"dim {dim}   device {device}   vocab {n_words}   params {n_par:,}")
    print(f"mode: {'ALIGNED (per-step attention over your message = the reasoning)' if align else 'BASELINE (pooled thought)'}")
    print(f"reader: {'COLLAPSE trajectory (order-aware, warm-started from the typer)' if reader == 'collapse' else 'MEANPOOL (bag-of-words baseline)'}")
    print(f"task: (your message) -> TYPE the reply   |   punish: per-word cross-entropy\n", flush=True)

    data = read_pairs(args.data, args.max_lines)
    random.shuffle(data)
    n_dev = max(1, int(len(data) * args.dev_frac))
    dev, train = data[:n_dev], data[n_dev:]
    print(f"train {len(train)}   dev {len(dev)}\n", flush=True)

    def pre_encode(chunk):
        msg = encode_ctx([m for m, r in chunk], stoi, unk, eos, args.ctx_words)
        rep = encode_batch([r for m, r in chunk], stoi, unk, eos)
        return msg, rep

    train_msg, train_rep = pre_encode(train)
    dev_msg, dev_rep = pre_encode(dev)

    def batches(msg_t, rep_t, shuffle=False):
        n = msg_t.size(0)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, args.batch_size):
            idx = order[i:i + args.batch_size]
            m, r = msg_t[idx], rep_t[idx]
            lm = int((m != 0).sum(1).max()); lr_ = int((r != 0).sum(1).max())
            yield m[:, :max(lm, 1)].to(device), r[:, :max(lr_, 1)].to(device)

    show = dev[:3]

    ban = [stoi[t] for t in SPECIALS]

    def show_samples():
        model.eval()
        msg = encode_ctx([m for m, r in show], stoi, unk, eos, args.ctx_words).to(device)
        gen, att = model.generate(msg, MAXLEN, unk=unk, ban=ban)
        print("  --- 3 dev examples ---")
        for k, (m, r) in enumerate(show):
            print(f"   you        : {m}")
            print(f"   real reply : {r}")
            print(f"   ai reply   : {decode(gen[k], itos, eos)}")
            if att is not None:
                print(f"   thinking   : {trace_str(decode(gen[k], itos, eos).split(), att[k], m.split()[-args.ctx_words:], msg.shape[1])}")
            print(flush=True)

    def evaluate():
        model.eval()
        tot_nll = tot_cnt = 0.0
        with torch.no_grad():
            for msg, rep in batches(dev_msg, dev_rep):
                tn, tc = model.reply_nll(msg, rep, reduce_mean=False)
                tot_nll += tn.sum().item(); tot_cnt += tc.sum().item()
        return tot_nll / max(tot_cnt, 1)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_nll, best_epoch = float("inf"), 0

    import time
    for ep in range(1, args.epochs + 1):
        model.train(); nb = 0
        run_t = torch.zeros((), device=device)
        t_mark = time.time()
        for msg, rep in batches(train_msg, train_rep, shuffle=True):
            loss = model.reply_nll(msg, rep)
            opt.zero_grad(); loss.backward(); opt.step()
            run_t += loss.detach(); nb += 1
            if nb % 100 == 0:
                now = time.time(); sps = 100 / (now - t_mark); t_mark = now
                print(f"  ep{ep} step {nb:4d}  loss {(run_t/nb).item():.4f}  "
                      f"strength {model.strength.item():.3f}  temp {model.temp.item():.3f}  "
                      f"| {sps:4.1f} steps/s", flush=True)
        nll = evaluate()
        print(f"epoch {ep}: dev reply-nll/word {nll:.4f}", flush=True)
        show_samples()
        if nll < best_nll:
            best_nll, best_epoch = nll, ep
            os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "stoi": stoi, "itos": itos,
                        "unk": unk, "eos": eos,
                        "config": {"dim": dim, "n_words": n_words, "align": align,
                                   "ctx_words": args.ctx_words, "reader": reader},
                        "best": {"epoch": ep, "nll": nll}}, args.ckpt)
            print(f"  [BEST so far (nll {nll:.4f}) -> saved {args.ckpt}]", flush=True)
        else:
            print(f"  [no improvement -> kept epoch {best_epoch} (nll {best_nll:.4f})]", flush=True)

    print(f"\nbest model = epoch {best_epoch} (nll {best_nll:.4f})  ->  {args.ckpt}")
    print(f"talk to it:  python3 chat_reply.py --chat")


if __name__ == "__main__":
    main()
