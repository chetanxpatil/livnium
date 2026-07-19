"""
premise_from_hyp.py — flip NLI into GENERATION: (hypothesis + label) -> premise.

Normal NLI is discriminative:  premise + hypothesis -> label.
Here we invert it: give the model the HYPOTHESIS and the LABEL, and make it
TYPE THE PREMISE, one word at a time. It is punished (cross-entropy) for every
premise word it gets wrong. To do that well it must learn what the label MEANS
generatively — what "contradiction" / "entailment" / "neutral" actually do to a
sentence — not just memorize a class index.

This is the brain-generator: a 'thought' (hypothesis + label) is unrolled into a
typed sequence (the premise) through the collapse geometry. A small controller
picks the next word, the collapse executes it.

    z      = brain_in([ meanpool(hypothesis) ; label_vec ])     # the thought
    h0     = z
    step t : query = brain([h ; z]) -> cos(query, word wells)/temp -> next word
             punish with CE vs the true premise word
             h <- collapse(h, well[true premise word])          # teacher forcing

BEST-MODEL IDEA — cross-sentence alignment (--align). The baseline above crushes
the whole hypothesis into ONE static vector z, so every premise word is typed
without ever looking at individual hypothesis words. On the discriminative task,
adding Parikh-style word-to-word alignment BEFORE the collapse bought +8.6 dev
points (66.1% -> 74.7% dev / 74.4% test, see SNLI_BASELINES.md). Here we port the same lever to
GENERATION: at each step the controller runs a label-conditioned attention over
the hypothesis WORD wells and feeds the aligned content into the query, so each
premise word is chosen with per-word hypothesis correspondence in hand:

    step t : ctx   = attend([h ; label] -> hypothesis words)    # soft alignment
             query = brain([h ; z ; ctx]) -> next word          # interaction first
             h     <- collapse(h, well[true premise word])      # teacher forcing

The collapse engine is unchanged; alignment just supplies per-word hypothesis
correspondence before the pick instead of a single pooled summary. The label
conditions the attention, so 'entailment' / 'contradiction' / 'neutral' can pull
different hypothesis content. Off by default; pass --align to switch it on.

Caveat by design: one (hypothesis, label) has MANY valid premises but only one
target, so EXACT premise reproduction is not the goal. We measure per-word loss
(perplexity) and the free bonus below.

FREE CLASSIFIER: at eval, score the real premise under all 3 labels and pick the
one that explains it best -> argmax_label P(premise | hypothesis, label). That is
a GENERATIVE NLI classifier; we report its accuracy next to the gold labels.

Usage:  python3 premise_from_hyp.py
Needs torch + model/sentence_typer.pt (word wells + vocab).
"""

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_typer import encode_batch, MAXLEN, SEED
# read_nli_jsonl is imported lazily inside the training path (see __main__) so the
# chat demo runs without the training script present.

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
SENT_CKPT = os.path.join(MODEL_DIR, "sentence_typer.pt")
DEFAULT_NLI = os.path.join(HERE, "data", "snli_1.0_train.jsonl")
N_LABELS = 3   # 0=entail, 1=neutral, 2=contradiction


class PremiseBrain(nn.Module):
    """Generates a premise from (hypothesis, label) via collapse-typing."""

    def __init__(self, n_words, dim, pad_idx, eos, warm=None, hidden=512, align=False,
                 label_every=False):
        super().__init__()
        self.pad_idx, self.eos, self.dim, self.n_words = pad_idx, eos, dim, n_words
        self.align = align
        self.label_every = label_every          # feed the label into the controller at EVERY step
        self.word_anchors = nn.Parameter(torch.randn(n_words, dim) * (1.0 / dim ** 0.5))
        if warm is not None:
            with torch.no_grad():
                self.word_anchors.copy_(warm); self.word_anchors[pad_idx].zero_()
        self.label_emb = nn.Parameter(torch.randn(N_LABELS, dim) * 0.1)   # what each label MEANS
        self.think = nn.Linear(2 * dim, dim)                              # [hyp ; label] -> thought
        # controller input parts: [h ; z] (+ aligned-ctx if --align) (+ label if label_every)
        n_parts = 2 + (1 if align else 0) + (1 if label_every else 0)
        self.brain = nn.Sequential(nn.Linear(n_parts * dim, hidden), nn.Tanh(), nn.Linear(hidden, dim))
        if align:
            # Parikh-style attend FFNs: key over hypothesis word wells, query from
            # the current collapse state conditioned on the label.
            self.att_key = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
            self.att_query = nn.Sequential(nn.Linear(2 * dim, dim), nn.Tanh())
        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.neg_samples = 0          # >0 => sampled-softmax CE during TRAINING only (set in main)

    @property
    def strength(self):
        return torch.sigmoid(self.log_strength)

    @property
    def temp(self):
        return F.softplus(self.log_temp) + 1e-3

    def meanpool(self, ids, A):
        m = (ids != self.pad_idx).float().unsqueeze(-1)
        return (A[ids] * m).sum(1) / m.sum(1).clamp(min=1.0)

    def thought(self, hyp_ids, labels, A):
        v = self.meanpool(hyp_ids, A)                 # (B, dim) hypothesis meaning
        lab = self.label_emb[labels]                  # (B, dim) the label's meaning
        return self.think(torch.cat([v, lab], dim=-1))   # (B, dim) the conditioned thought

    def collapse_step(self, h, target):
        align = (F.normalize(h, dim=-1) * target).sum(-1)
        away = F.normalize(h - target, dim=-1)
        h = h - self.strength * (1.0 - align).unsqueeze(-1) * away
        n = h.norm(dim=-1, keepdim=True)
        return torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)

    def align_context(self, hn, lab, EH, K, hmask):
        """Label-conditioned attention over hypothesis word wells -> aligned content.
        hn  (B,dim) normalized state, lab (B,dim) label meaning,
        EH  (B,Lh,dim) hyp word wells, K (B,Lh,dim) their keys, hmask (B,Lh) bool."""
        q = self.att_query(torch.cat([hn, lab], dim=-1)).unsqueeze(1)     # (B,1,dim)
        scores = torch.bmm(q, K.transpose(1, 2)).squeeze(1)              # (B,Lh)
        scores = scores.masked_fill(~hmask, -1e9)
        attn = torch.softmax(scores, dim=-1).unsqueeze(1)                # (B,1,Lh)
        return torch.bmm(attn, EH).squeeze(1)                            # (B,dim) aligned hyp content

    def premise_nll(self, hyp_ids, prem_ids, labels, reduce_mean=True):
        """Teacher-forced negative log-likelihood of the premise. This is the punishment."""
        A = F.normalize(self.word_anchors, dim=-1)
        lab = self.label_emb[labels]                  # (B, dim) the label's meaning
        z = self.think(torch.cat([self.meanpool(hyp_ids, A), lab], dim=-1))   # (B, dim)
        h = z
        if self.align:                                # precompute hyp word wells + keys once
            EH = A[hyp_ids]                           # (B, Lh, dim) hypothesis word wells
            hmask = (hyp_ids != self.pad_idx)         # (B, Lh)
            K = self.att_key(EH)                      # (B, Lh, dim)
        # batches already trim premise/hyp to the batch's real max length on the CPU,
        # so we loop over prem_ids.shape[1] directly — no per-step host sync here (#2).
        B, L = prem_ids.shape
        tok_nll = torch.zeros(B, device=prem_ids.device)
        tok_cnt = torch.zeros(B, device=prem_ids.device)
        sampled = self.training and self.neg_samples > 0   # bypass full-vocab matmul in training
        for t in range(L):
            hn = F.normalize(h, dim=-1)
            parts = [hn, z]
            if self.align:
                parts.append(self.align_context(hn, lab, EH, K, hmask))  # (B, dim) aligned ctx
            if self.label_every:
                parts.append(lab)                                        # label at every step
            query = F.normalize(self.brain(torch.cat(parts, dim=-1)), dim=-1)
            tgt = prem_ids[:, t]
            if sampled:
                # score the TRUE word + K random negatives only (not all n_words)
                pos = (query * A[tgt]).sum(-1, keepdim=True) / self.temp          # (B, 1)
                neg_ids = torch.randint(1, self.n_words, (self.neg_samples,),
                                        device=prem_ids.device)                  # skip PAD(0)
                neg = (query @ A[neg_ids].t()) / self.temp                       # (B, K)
                # mask false negatives: sampled ids equal to the true target.
                # NOTE: published checkpoints predate this mask (see CHECKPOINTS.md);
                # retraining will not exactly reproduce them.
                neg = neg.masked_fill(neg_ids.unsqueeze(0) == tgt.unsqueeze(1), float("-inf"))
                cand = torch.cat([pos, neg], dim=1)                              # (B, 1+K), true=col 0
                ll = F.cross_entropy(cand, torch.zeros(cand.size(0), dtype=torch.long,
                                                       device=prem_ids.device), reduction="none")
            else:
                logits = (query @ A.t()) / self.temp                            # (B, n_words) full
                ll = F.cross_entropy(logits, tgt, reduction="none")
            mask = (tgt != self.pad_idx).float()
            tok_nll += ll * mask; tok_cnt += mask
            h = self.collapse_step(h, A[tgt])                             # teacher forcing
        if reduce_mean:
            return (tok_nll / tok_cnt.clamp(min=1)).mean()
        return tok_nll, tok_cnt                                          # per-example sums

    @torch.no_grad()
    def generate(self, hyp_ids, labels, max_len, unk=None):
        """Free-running greedy decode (NOT teacher forced): the model types its own
        premise word by word, feeding each choice back into the collapse. Returns the
        generated token ids and, in --align mode, the hypothesis-word index it leaned
        on most at each step (the 'thinking' — what it was looking at while typing)."""
        A = F.normalize(self.word_anchors, dim=-1)
        lab = self.label_emb[labels]
        z = self.think(torch.cat([self.meanpool(hyp_ids, A), lab], dim=-1))
        h = z
        if self.align:
            EH = A[hyp_ids]; hmask = (hyp_ids != self.pad_idx); K = self.att_key(EH)
        B = hyp_ids.size(0)
        toks, attn = [], []
        done = torch.zeros(B, dtype=torch.bool, device=hyp_ids.device)
        for _ in range(max_len):
            hn = F.normalize(h, dim=-1)
            parts = [hn, z]
            if self.align:
                q = self.att_query(torch.cat([hn, lab], dim=-1)).unsqueeze(1)
                scores = torch.bmm(q, K.transpose(1, 2)).squeeze(1).masked_fill(~hmask, -1e9)
                a = torch.softmax(scores, dim=-1)
                attn.append(a.argmax(-1))
                parts.append(torch.bmm(a.unsqueeze(1), EH).squeeze(1))
            if self.label_every:
                parts.append(lab)
            query = self.brain(torch.cat(parts, dim=-1))
            logits = (F.normalize(query, dim=-1) @ A.t()) / self.temp
            logits[:, self.pad_idx] = float("-inf")          # never type PAD
            if unk is not None:
                logits[:, unk] = float("-inf")               # never type <unk>
            nxt = logits.argmax(-1)
            toks.append(nxt)
            h = self.collapse_step(h, A[nxt])                # feed own choice back
            done = done | (nxt == self.eos)
            if bool(done.all()):
                break
        T = torch.stack(toks, dim=1)                          # (B, t)
        att = torch.stack(attn, dim=1) if (self.align and attn) else None
        return T, att


def load_wells(device):
    ck = torch.load(SENT_CKPT, map_location=device)
    return (ck["word_anchors"].to(device), ck["stoi"], ck["unk"], ck["eos"],
            ck["config"]["n_words"], ck["config"]["dim"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default=DEFAULT_NLI)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.02)
    ap.add_argument("--align", action="store_true",
                    help="best-model idea: per-step label-conditioned alignment over "
                         "hypothesis words before each premise word (cross-sentence "
                         "interaction, the generative analog of Parikh attention)")
    ap.add_argument("--best-metric", default="nll", choices=["nll", "acc"],
                    help="which dev metric decides the 'best' checkpoint: nll (lower "
                         "better, default — tracks generation quality) or acc (higher "
                         "better — the generative-classifier accuracy)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="Fix 1: contrastive label-margin (nats/word). >0 turns it on; "
                         "pushes the gold label's premise-nll below each wrong label by "
                         "this margin. Directly attacks the classifier plateau. Try 0.2.")
    ap.add_argument("--margin-weight", type=float, default=1.0,
                    help="weight on the margin term relative to the gold cross-entropy")
    ap.add_argument("--margin-every", type=int, default=4,
                    help="apply the contrastive margin every K steps (gold-CE only on the "
                         "rest). The margin costs 3x decode; K=4 cuts the overhead ~75%% "
                         "for ~the same separation. Set 1 for max separation, higher for speed.")
    ap.add_argument("--neg-samples", type=int, default=0,
                    help="sampled-softmax: score the true word + this many random negatives "
                         "instead of all 20003 wells (TRAINING only; eval/generate stay "
                         "full-vocab). Bypasses the dominant matmul. Try 512. 0=off.")
    ap.add_argument("--label-every-step", action="store_true",
                    help="Fix 2: feed the label into the controller at every step (auto-on "
                         "when --margin>0, so the margin can reach late words)")
    args = ap.parse_args()
    label_every = args.label_every_step or (args.margin > 0)   # Fix 2 enables Fix 1's reach

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    random.seed(SEED); torch.manual_seed(SEED)

    print("starting up, loading word wells ...", flush=True)
    warm, stoi, unk, eos, n_words, dim = load_wells(device)
    model = PremiseBrain(n_words, dim, 0, eos, warm=warm, align=args.align,
                         label_every=label_every).to(device)
    model.neg_samples = args.neg_samples       # sampled-softmax in training (eval uses full)

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"dim {dim}   device {device}   vocab {n_words}", flush=True)
    print(f"mode: {'ALIGNED (cross-sentence attention before collapse)' if args.align else 'BASELINE (static mean-pooled thought)'}", flush=True)
    if args.margin > 0:
        print(f"Fix 1: contrastive margin {args.margin} (w={args.margin_weight})  |  "
              f"Fix 2: label-every-step ON", flush=True)
    if args.neg_samples > 0:
        print(f"sampled-softmax: {args.neg_samples} negatives in training "
              f"(eval/generate full-vocab)", flush=True)
    print(f"task: (hypothesis + label) -> TYPE the premise   |   punish: per-word cross-entropy", flush=True)
    print(f"trainable params: {n_par:,}", flush=True)

    print("loading SNLI (this takes a few seconds) ...", flush=True)
    from train_nli_from_sentence import read_nli_jsonl   # (premise, hypothesis, label_idx)
    data = read_nli_jsonl(args.nli_path, args.max_lines)   # (premise, hypothesis, label)
    random.shuffle(data)
    n_dev = int(len(data) * args.dev_frac)
    dev, train = data[:n_dev], data[n_dev:]
    print(f"train {len(train)}   dev {len(dev)}\n")

    # speedup #3: tokenize the whole corpus ONCE into CPU tensors (was re-encoding every
    # batch every epoch). Batches just index + .to(device).
    def pre_encode(chunk):
        prem = encode_batch([p for p, h, y in chunk], stoi, unk, eos)
        hyp = encode_batch([h for p, h, y in chunk], stoi, unk, eos)
        y = torch.tensor([y for p, h, y in chunk], dtype=torch.long)
        return prem, hyp, y

    train_prem, train_hyp, train_y = pre_encode(train)
    dev_prem, dev_hyp, dev_y = pre_encode(dev)

    itos = {i: w for w, i in stoi.items()}; itos[unk] = "<unk>"; itos[eos] = "<eos>"
    LABEL_NAMES = ["entail", "neutral", "contra"]

    def decode(ids):
        out = []
        for t in ids.tolist():
            if t == eos or t == 0:
                break
            out.append(itos.get(t, "?"))
        return " ".join(out) if out else "(empty)"

    # 3 fixed dev examples (one of each gold label if possible) shown every epoch
    show = []
    for lab in range(N_LABELS):
        for ex in dev:
            if ex[2] == lab:
                show.append(ex); break
    show = (show + dev)[:3]

    def show_samples():
        model.eval()
        prem, hyp, y = batchify(show)
        gen, att = model.generate(hyp, y, MAXLEN, unk=unk)
        print("  --- 3 dev examples (generated under the GOLD label) ---")
        for k, (p, hh, yy) in enumerate(show):
            print(f"   [{LABEL_NAMES[yy]}]  hypothesis: {hh}")
            print(f"            gold premise: {p}")
            print(f"            ai premise  : {decode(gen[k])}")
            if att is not None:                       # the 'thinking': hyp word it leaned on per typed word
                gw = decode(gen[k]).split()
                hyp_words = hh.split()
                trace = []
                for j, w in enumerate(gw):
                    hi = att[k, j].item()
                    look = hyp_words[hi] if hi < len(hyp_words) else "<eos>"
                    trace.append(f"{w}<-{look}")
                print(f"            thinking    : {' '.join(trace)}")
            print(flush=True)

    def batchify(chunk):
        prem = encode_batch([p for p, h, y in chunk], stoi, unk, eos).to(device)
        hyp = encode_batch([h for p, h, y in chunk], stoi, unk, eos).to(device)
        y = torch.tensor([y for p, h, y in chunk], dtype=torch.long, device=device)
        return prem, hyp, y

    def batches(prem_t, hyp_t, y_t, shuffle=False):
        n = y_t.size(0)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, args.batch_size):
            idx = order[i:i + args.batch_size]
            p, h = prem_t[idx], hyp_t[idx]
            # trim to this batch's real max length ON CPU (no GPU stall) + transfer less data
            lp = int((p != 0).sum(1).max()); lh = int((h != 0).sum(1).max())
            yield (p[:, :max(lp, 1)].to(device), h[:, :max(lh, 1)].to(device),
                   y_t[idx].to(device))

    def nll_all_labels(hyp, prem):
        """speedup #1: score the premise under ALL 3 labels in a SINGLE decode loop
        (tile the batch x3) instead of three separate calls. Returns (B, 3) per-word nll,
        differentiable — used by both the margin loss and the generative classifier."""
        B = hyp.size(0)
        hyp3 = hyp.repeat_interleave(N_LABELS, dim=0)               # [h0,h0,h0, h1,h1,h1, ...]
        prem3 = prem.repeat_interleave(N_LABELS, dim=0)
        labs3 = torch.arange(N_LABELS, device=hyp.device).repeat(B)  # [0,1,2, 0,1,2, ...]
        tn, tc = model.premise_nll(hyp3, prem3, labs3, reduce_mean=False)
        return (tn / tc.clamp(min=1)).view(B, N_LABELS)             # row i = [nll_E, nll_N, nll_C]

    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    os.makedirs(MODEL_DIR, exist_ok=True)
    out_name = "premise_from_hyp_align.pt" if args.align else "premise_from_hyp.pt"
    out_ckpt = os.path.join(MODEL_DIR, out_name)

    def save_ckpt(epoch, nll, acc):
        torch.save({"state_dict": model.state_dict(), "stoi": stoi, "unk": unk, "eos": eos,
                    "config": {"dim": dim, "n_words": n_words, "align": args.align,
                               "label_every": label_every},
                    "best": {"epoch": epoch, "metric": args.best_metric, "nll": nll, "acc": acc}},
                   out_ckpt)

    # best-checkpoint tracking: keep ONLY the best epoch, overwrite when beaten,
    # never let a worse later epoch clobber it.
    lower_better = (args.best_metric == "nll")
    best_score = float("inf") if lower_better else float("-inf")
    best_epoch = 0

    def evaluate(prem_t, hyp_t, y_t):
        """Per-word NLL with the TRUE label and the argmax-label generative classifier
        accuracy. Uses the single-loop all-labels scorer (speedup #1)."""
        model.eval()
        true_nll = tot = 0.0
        gen_correct = seen = 0
        with torch.no_grad():
            for prem, hyp, y in batches(prem_t, hyp_t, y_t):
                B = y.size(0)
                P = nll_all_labels(hyp, prem)                    # (B, 3) in one decode loop
                pred = P.argmin(dim=1)                           # label that explains premise best
                gen_correct += (pred == y).sum().item(); seen += B
                true_nll += P[torch.arange(B, device=device), y].sum().item(); tot += B
        return true_nll / max(tot, 1), gen_correct / max(seen, 1)

    def train_loss(prem, hyp, y, apply_margin):
        """Gold cross-entropy, plus (when apply_margin) the contrastive label-margin (Fix 1):
        push the GOLD label's per-word nll below each WRONG label's by `margin`. The margin
        is throttled to every K steps because scoring all 3 labels costs 3x decode."""
        if args.margin <= 0 or not apply_margin:
            return model.premise_nll(hyp, prem, y), 0.0          # cheap: gold CE only (1 decode)
        B = y.size(0)
        P = nll_all_labels(hyp, prem)                            # (B, 3) in ONE decode loop (#1)
        ar = torch.arange(B, device=device)
        gold = P[ar, y]                                          # (B,) gold-label nll
        diff = P - gold.unsqueeze(1)                             # wrong - gold; want >= margin
        hinge = F.relu(args.margin - diff)                       # (B, 3) penalty where gap too small
        mask = torch.ones_like(P); mask[ar, y] = 0.0             # ignore the gold column
        margin = (hinge * mask).sum(1) / (N_LABELS - 1)          # (B,) mean over wrong labels
        mbar = margin.mean()
        return gold.mean() + args.margin_weight * mbar, mbar      # tensor (accumulated on-device)

    for ep in range(1, args.epochs + 1):
        model.train(); nb = 0; nm = 0
        run_t = torch.zeros((), device=device)      # accumulate on-device (no per-step sync)
        runm_t = torch.zeros((), device=device)
        for prem, hyp, y in batches(train_prem, train_hyp, train_y, shuffle=True):
            apply_margin = (args.margin > 0) and (nb % args.margin_every == 0)
            loss, mval = train_loss(prem, hyp, y, apply_margin)
            opt.zero_grad(); loss.backward(); opt.step()
            run_t += loss.detach(); nb += 1
            if apply_margin:
                runm_t += mval.detach(); nm += 1
            if nb % 100 == 0:                        # one host sync per 100 steps, not per step
                extra = f"  margin {(runm_t/max(nm,1)).item():.4f}" if args.margin > 0 else ""
                print(f"  ep{ep} step {nb:4d}  loss {(run_t/nb).item():.4f}{extra}  "
                      f"strength {model.strength.item():.3f}  temp {model.temp.item():.3f}", flush=True)
        nll, gen_acc = evaluate(dev_prem, dev_hyp, dev_y)
        print(f"epoch {ep}: dev premise-nll/word {nll:.4f}   "
              f"generative-classifier acc {gen_acc*100:.2f}%", flush=True)
        show_samples()
        score = nll if lower_better else gen_acc
        improved = (score < best_score) if lower_better else (score > best_score)
        if improved:
            best_score, best_epoch = score, ep
            save_ckpt(ep, nll, gen_acc)                      # overwrite -> only the best survives
            tag = f"nll {nll:.4f}" if lower_better else f"acc {gen_acc*100:.2f}%"
            print(f"  [BEST so far ({tag}) -> saved {out_ckpt}]", flush=True)
        else:
            keep = f"nll {best_score:.4f}" if lower_better else f"acc {best_score*100:.2f}%"
            print(f"  [no improvement -> kept epoch {best_epoch} ({keep})]", flush=True)

    # final: show the label actually helps (true label vs each fixed wrong label)
    print("\n--- does the label carry generative meaning? (lower nll = better fit) ---")
    model.eval()
    with torch.no_grad():
        agg = torch.zeros(N_LABELS); n = 0
        gold_nll = 0.0
        for prem, hyp, y in batches(dev_prem, dev_hyp, dev_y):
            B = y.size(0)
            P = nll_all_labels(hyp, prem)                    # (B, 3) one decode loop (#1)
            agg += P.sum(0).cpu()
            gold_nll += P[torch.arange(B, device=device), y].sum().item(); n += B
        print(f"  premise-nll/word with GOLD label   : {gold_nll/n:.4f}")
        for lab, name in enumerate(["entail", "neutral", "contra"]):
            print(f"  premise-nll/word forced {name:8s}   : {agg[lab].item()/n:.4f}")

    metric_str = f"nll {best_score:.4f}" if lower_better else f"acc {best_score*100:.2f}%"
    print(f"\nbest model = epoch {best_epoch} ({metric_str})  ->  {out_ckpt}")


if __name__ == "__main__":
    main()
