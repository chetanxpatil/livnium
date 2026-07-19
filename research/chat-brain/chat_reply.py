"""
chat_reply.py — the GENERATOR: (your last turns) -> TYPE the reply.

One mechanism, used three times on the same word wells:
    READ    collapse through the context words -> a trajectory of states.
    THINK   z = linear(final state); the writing state h starts at z.
    WRITE   step t: attend over [trajectory + own typed words], build a
            query, cos(query, wells)/temp picks the next word. Punish with
            per-word CE vs the true reply; collapse h onto the word walked.

Everything else is a training lever — all default-ON, each with an off switch:
    fast reader      distilled 8-tap conv = the sequential read   --no-fast-reader
    vocab cut        wells only for words seen >=2x in train      --min-freq 1
    char wells       OOV words READ through minted char wells     --no-char-wells
    meaning weights  rare (content) words punish harder           --meaning-w 0
    pos scaffold     "where am I in the reply", annealed to 0     --pos-anneal 0
    sched sampling   sometimes walk on own picks, ramped up       --sched-sample 0
    sampled softmax  512 negatives instead of the full vocab      --neg-samples 0
    early stop       patience on dev NLL                          --patience N

The char layer (char_fingerprint.py) is READ-side only: an unseen word gets a
stable well minted from its letters so the reader never collapses through
<unk> mush — but the writer's vocabulary stays trained-words-only. Spelling
earns a word the right to be HEARD; only training earns the right to be SAID.

Data: data/chat_context.tsv from prep_chat_context.py — the last K turns of a
real conversation, speaker-tagged <you>/<me>, tail-truncated (newest words
survive). --chat feeds the live conversation back in the same shape.

Usage:
    python3 prep_chat_context.py     # once: raw export -> data/chat_context.tsv
    python3 chat_reply.py            # train
    python3 chat_reply.py --chat     # talk to it (multi-turn)
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from chat_typer import encode_batch, MAXLEN, PAD, SEED
from char_fingerprint import letter_anchors, char_fingerprint
from paths import NOUN_CHECKPOINT, data_path, model_path

TYPER_CKPT = model_path("chat_typer.pt")
FAST_CKPT = model_path("fast_reader.pt")
CKPT_OUT = model_path("chat_reply.pt")
SPECIALS = ["<you>", "<me>"]      # speaker wells: minted fresh, trained like words
CTX_WORDS = 256                   # context budget (oldest words drop first —
                                  # big enough that whole questions go in)


# ---------------------------------------------------------------- the model

class ReplyBrain(nn.Module):
    """(context ids) -> type the reply, word by word, on the shared wells."""

    def __init__(self, n_words, dim, eos, warm=None, hidden=512, pos=False,
                 warm_start=None, warm_strength=None, pure=False, pos_well=True,
                 align=False):
        super().__init__()
        self.eos, self.dim, self.n_words = eos, dim, n_words
        self.pos = pos
        self.pure = pure          # pure collapse writer: no attention/MLP/memory
        # positional WELL: one anchor per reply position, ADDED to the query.
        # pure geometry (just a vector add) — no MLP, no attention — but it tells
        # the writer WHERE it is, which is what stops the pure bigram loop.
        self.pos_well = bool(pos_well and pure)
        # pure cosine-ALIGNMENT: each step scores the query against the CONTEXT
        # word wells by raw cosine (no learned Q/K/V — same op as the decode),
        # softmaxes, and adds the aligned context to the query. Tells the writer
        # WHAT to look at, per word. Only new param is one temperature scalar.
        self.align = bool(align and pure)
        # training levers, set from outside (see main):
        self.pos_w = 0.0          # positional scaffold weight, annealed 1 -> 0
        self.sample_p = 0.0       # scheduled sampling: prob of walking on own pick
        self.neg_samples = 0      # sampled-softmax negatives (0 = full vocab CE)
        self.word_w = None        # per-word loss weights (meaning weighting)
        self.fast_alpha = None    # distilled reader taps (None = exact walk)
        self.oov_wells = None     # frozen char wells for OOV words (READ only)

        # the wells: one vector per word, warm-started from the typer
        self.word_anchors = nn.Parameter(torch.randn(n_words, dim) / dim ** 0.5)
        if warm is not None:
            with torch.no_grad():
                self.word_anchors.copy_(warm); self.word_anchors[PAD].zero_()

        # reader: the typer's start vector + its learned collapse strength
        self.start = nn.Parameter(torch.randn(dim) * 0.05)
        if warm_start is not None:
            with torch.no_grad():
                self.start.copy_(warm_start)
        s0 = 2.2 if warm_strength is None else float(
            torch.logit(torch.tensor(min(max(warm_strength, 1e-3), 1 - 1e-3))))
        self.log_strength_read = nn.Parameter(torch.tensor(s0))

        # writer: thought + attention + brain + write-collapse
        if self.pos_well:
            self.pos_anchor = nn.Parameter(torch.randn(MAXLEN + 2, dim) * 0.05)
        if self.align:
            self.log_align_temp = nn.Parameter(torch.tensor(0.0))   # only align param

        if not self.pure:
            self.think = nn.Linear(dim, dim)
            if pos:
                self.pos_emb = nn.Parameter(torch.randn(MAXLEN + 2, dim) * 0.05)
            n_parts = 3 + (1 if pos else 0)               # [h ; z ; ctx] (+pos)
            self.brain = nn.Sequential(nn.Linear(n_parts * dim, hidden), nn.Tanh(),
                                       nn.Linear(hidden, dim))
            self.att_key = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
            self.att_query = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())

        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def strength(self):
        return torch.sigmoid(self.log_strength)

    @property
    def temp(self):
        return F.softplus(self.log_temp) + 1e-3

    @property
    def strength_read(self):
        return torch.sigmoid(self.log_strength_read)

    @property
    def align_temp(self):
        return F.softplus(self.log_align_temp) + 1e-3

    # -- reading ------------------------------------------------------------

    def read(self, ids, A):
        """EXACT reader: collapse through the context sequentially.
        Returns (normalized trajectory, mask, final state)."""
        B, L = ids.shape
        mask = (ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        s = self.strength_read
        states = []
        for i in range(L):
            target = A[ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            h_norm = h.norm(dim=-1, keepdim=True)
            h_n = h / (h_norm + 1e-8)
            align = (h_n * target).sum(-1, keepdim=True)
            # Analytical energy gradient: grad = -(target - h_n * align) / (h_norm + 1e-8)
            grad = -(target - h_n * align) / (h_norm + 1e-8)
            h = h + m * (-s * grad)
            n = h.norm(dim=-1, keepdim=True)
            h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
        S = torch.stack(states, dim=1)
        return F.normalize(S, dim=-1), mask, h

    def read_fast(self, ids, A):
        """DISTILLED reader (fast_reader.py): the same trajectory as a causal
        K-tap conv over the wells — one parallel op, no sequential walk."""
        mask = (ids != PAD)
        wells = A[ids] * mask.unsqueeze(-1)               # PAD contributes zero
        B, L, D = wells.shape
        K = self.fast_alpha.numel()
        x = F.pad(wells.transpose(1, 2), (K - 1, 0))      # causal left-pad
        w = self.fast_alpha.flip(0).view(1, 1, K).expand(D, 1, K)
        S = F.conv1d(x, w, groups=D).transpose(1, 2)
        lens = mask.sum(1).clamp(min=1) - 1
        hT = S[torch.arange(B, device=ids.device), lens]  # last real state
        return F.normalize(S + 1e-8, dim=-1), mask, hT

    def read_context(self, ids, A):
        if self.fast_alpha is not None:
            return self.read_fast(ids, A)
        return self.read(ids, A)

    def read_table(self, A):
        """READING sees [trained wells | frozen char wells]. WRITING sees A
        only — context ids may point past n_words, reply targets never do."""
        if self.oov_wells is None or self.oov_wells.numel() == 0:
            return A
        return torch.cat([A, self.oov_wells], dim=0)

    # -- writing ------------------------------------------------------------

    def collapse_step(self, h, target):
        h_norm = h.norm(dim=-1, keepdim=True)
        h_n = h / (h_norm + 1e-8)
        align = (h_n * target).sum(-1, keepdim=True)
        # Analytical energy gradient: grad = -(target - h_n * align) / (h_norm + 1e-8)
        grad = -(target - h_n * align) / (h_norm + 1e-8)
        h = h - self.strength * grad
        n = h.norm(dim=-1, keepdim=True)
        return torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)

    def reply_nll(self, msg_ids, rep_ids, reduce_mean=True):
        """Teacher-forced NLL of the reply — the punishment.

        PURE mode (self.pure): no attention, no MLP, no growing memory. The
        writer is just collapse — h starts at the thought z, each step picks
        the nearest well to (h+z) and collapses onto it. O(L), fast, same
        mechanism as the noun/char models.

        FULL mode: cross-attention over a memory that GROWS each step (each
        typed word joins it). More expressive, but O(L^2) and slow — growth
        uses torch.cat so autograd keeps the old memory for backward."""
        A = F.normalize(self.word_anchors, dim=-1)
        AT = self.read_table(A)
        if self.pure:
            _, _, hread = self.read_context(msg_ids, AT)
            if self.align:
                Cwells = AT[msg_ids]; cmask = (msg_ids != PAD)   # context word wells
            z = hread # Pure mode bypasses self.think linear layer
        else:
            mem, vmask, hread = self.read_context(msg_ids, AT)
            kmem = self.att_key(mem)
            z = self.think(hread)
        h = z
        B, L = rep_ids.shape
        tok_nll = torch.zeros(B, device=rep_ids.device)
        tok_cnt = torch.zeros(B, device=rep_ids.device)
        sampled = self.training and self.neg_samples > 0
        for t in range(L):
            hn = F.normalize(h, dim=-1)
            if self.pure:                                 # state + thought (+ position + aligned context)
                q = hn + z
                if self.pos_well:                         # WHERE am I — pure vector add, no MLP
                    q = q + self.pos_anchor[min(t, self.pos_anchor.size(0) - 1)]
                if self.align:                            # WHAT to look at — cosine over context wells, no learned Q/K
                    probe = F.normalize(hn + z, dim=-1).unsqueeze(1)
                    sc = torch.bmm(probe, Cwells.transpose(1, 2)).squeeze(1) / self.align_temp
                    a = torch.softmax(sc.masked_fill(~cmask, float("-inf")), dim=-1)
                    q = q + torch.bmm(a.unsqueeze(1), Cwells).squeeze(1)
                query = F.normalize(q, dim=-1)
            else:
                q = self.att_query(hn).unsqueeze(1)
                scores = torch.bmm(q, kmem.transpose(1, 2)).squeeze(1)
                attn = torch.softmax(scores.masked_fill(~vmask, -1e9), dim=-1)
                ctx = torch.bmm(attn.unsqueeze(1), mem).squeeze(1)
                parts = [hn, z, ctx]
                if self.pos:                              # annealed scaffold
                    parts.append(self.pos_w * self.pos_emb[t].unsqueeze(0).expand(B, -1))
                query = F.normalize(self.brain(torch.cat(parts, dim=-1)), dim=-1)
            tgt = rep_ids[:, t]
            if sampled:                                   # 1 pos + K random negs
                pos = (query * A[tgt]).sum(-1, keepdim=True) / self.temp
                neg_ids = torch.randint(1, self.n_words, (self.neg_samples,),
                                        device=rep_ids.device)
                neg = (query @ A[neg_ids].t()) / self.temp
                cand = torch.cat([pos, neg], dim=1)
                ll = F.cross_entropy(cand, torch.zeros(cand.size(0), dtype=torch.long,
                                                       device=rep_ids.device),
                                     reduction="none")
            else:                                         # full-vocab CE
                ll = F.cross_entropy((query @ A.t()) / self.temp, tgt,
                                     reduction="none")
            mask = (tgt != PAD).float()
            if self.training and self.word_w is not None:
                # meaning weighting: "the" is cheap, "scaffold" is expensive —
                # the reply must earn its content words, not just its skeleton
                mask = mask * self.word_w[tgt]
            tok_nll += ll * mask; tok_cnt += mask
            walk = tgt                                    # teacher forcing
            if self.training and self.sample_p > 0:
                # scheduled sampling: with prob p walk on the model's own pick.
                # full-vocab argmax only for the rows that flipped heads.
                coin = (torch.rand(B, device=rep_ids.device) < self.sample_p) & (tgt != PAD)
                if bool(coin.any()):
                    with torch.no_grad():
                        lg = query[coin] @ A.t()
                        lg[:, PAD] = float("-inf")
                        walk = tgt.clone()
                        walk[coin] = lg.argmax(-1)
            h = self.collapse_step(h, A[walk])
            if not self.pure:                             # grow the self-attend memory
                hs = F.normalize(h, dim=-1).unsqueeze(1)  # typed word joins memory
                mem = torch.cat([mem, hs], dim=1)
                kmem = torch.cat([kmem, self.att_key(hs)], dim=1)
                vmask = torch.cat([vmask, (tgt != PAD).unsqueeze(1)], dim=1)
        if reduce_mean:
            return (tok_nll / tok_cnt.clamp(min=1)).mean()
        return tok_nll, tok_cnt

    @torch.no_grad()
    def generate(self, msg_ids, max_len, unk=None, ban=(),
                 rep_penalty=1.0, no_repeat_bigram=False, temperature=0.0):
        """Free-running decode. Returns token ids and, per step, the memory
        index it attended to (the 'thinking' trace).

        Anti-loop levers (all off by default = plain greedy):
          rep_penalty >1     : divide the logit of any already-said word (calms
                               'do you do you' by making repeats progressively
                               costlier). 1.1-1.3 is gentle, 1.5+ is firm.
          no_repeat_bigram   : hard-block any (prev, next) pair already produced
                               — kills 2-word cycles outright.
          temperature >0     : sample instead of argmax (adds variety; 0 = greedy)."""
        A = F.normalize(self.word_anchors, dim=-1)
        AT = self.read_table(A)
        traj, tmask, hread = self.read_context(msg_ids, AT)
        z = hread if self.pure else self.think(hread)
        h = z
        if self.pure and self.align:
            Cwells = AT[msg_ids]; cmask = tmask               # context word wells
        B, Lm = msg_ids.size(0), traj.size(1)
        if not self.pure:
            # preallocated memory (no_grad => in-place writes are safe here)
            mem = traj.new_zeros(B, Lm + max_len, self.dim); mem[:, :Lm] = traj
            kmem = traj.new_zeros(B, Lm + max_len, self.dim); kmem[:, :Lm] = self.att_key(traj)
            vmask = torch.zeros(B, Lm + max_len, dtype=torch.bool, device=msg_ids.device)
            vmask[:, :Lm] = tmask
            mlen = Lm
        toks, attn = [], []
        done = torch.zeros(B, dtype=torch.bool, device=msg_ids.device)
        prev = None                                   # repeat-killer: last word
        seen_bigrams = [set() for _ in range(B)]      # (prev, next) already made
        ar = torch.arange(B, device=msg_ids.device)
        for t in range(max_len):
            hn = F.normalize(h, dim=-1)
            if self.pure:                             # pure collapse: state + thought (+ position + aligned context)
                q = hn + z
                if self.pos_well:
                    q = q + self.pos_anchor[min(t, self.pos_anchor.size(0) - 1)]
                if self.align:
                    probe = F.normalize(hn + z, dim=-1).unsqueeze(1)
                    sc = torch.bmm(probe, Cwells.transpose(1, 2)).squeeze(1) / self.align_temp
                    a = torch.softmax(sc.masked_fill(~cmask, float("-inf")), dim=-1)
                    q = q + torch.bmm(a.unsqueeze(1), Cwells).squeeze(1)
                query = F.normalize(q, dim=-1)
            else:
                q = self.att_query(hn).unsqueeze(1)
                scores = torch.bmm(q, kmem[:, :mlen].transpose(1, 2)).squeeze(1)
                a = torch.softmax(scores.masked_fill(~vmask[:, :mlen], -1e9), dim=-1)
                attn.append(a.argmax(-1))
                ctx = torch.bmm(a.unsqueeze(1), mem[:, :mlen]).squeeze(1)
                parts = [hn, z, ctx]
                if self.pos:
                    parts.append(self.pos_w * self.pos_emb[t].unsqueeze(0).expand(B, -1))
                query = F.normalize(self.brain(torch.cat(parts, dim=-1)), dim=-1)
            logits = (query @ A.t()) / self.temp
            if rep_penalty != 1.0 and toks:           # penalize everything said
                hist = torch.stack(toks, dim=1)       # (B, t)
                g = torch.gather(logits, 1, hist)
                g = torch.where(g > 0, g / rep_penalty, g * rep_penalty)
                logits.scatter_(1, hist, g)
            logits[:, PAD] = float("-inf")
            if unk is not None:
                logits[:, unk] = float("-inf")
            for b in ban:                             # e.g. never type <you>/<me>
                logits[:, b] = float("-inf")
            if prev is not None:                      # no same word twice in a row
                logits[ar, prev] = float("-inf")
            if no_repeat_bigram and prev is not None:  # block (prev, y) reuse
                for b in range(B):
                    for (p, y) in seen_bigrams[b]:
                        if p == int(prev[b]):
                            logits[b, y] = float("-inf")
            if temperature > 0:                       # sample, calmer variety
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            if prev is not None:
                for b in range(B):
                    seen_bigrams[b].add((int(prev[b]), int(nxt[b])))
            prev = nxt
            toks.append(nxt)
            h = self.collapse_step(h, A[nxt])
            if not self.pure:                         # own word joins the memory
                mem[:, mlen] = F.normalize(h, dim=-1)
                kmem[:, mlen] = self.att_key(mem[:, mlen])
                vmask[:, mlen] = ~done
                mlen += 1
            done = done | (nxt == self.eos)
            if bool(done.all()):
                break
        T = torch.stack(toks, dim=1)
        return T, (torch.stack(attn, dim=1) if attn else None)


# ---------------------------------------------------------------- data

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


def shrink_vocab(train_pairs, warm, stoi, itos, unk, eos, min_freq):
    """Cut the wells to words seen >= min_freq times in TRAIN. 100k wells on
    ~18k pairs dilutes the negatives with words that can never be targets.
    Kept typer words keep their warm wells. Frequent words the typer NEVER
    saw (punctuation, merged contractions, new-corpus words) get a fresh
    TRAINABLE well, char-fingerprint initialized — they earn their place in
    training instead of drowning as <unk>. Rare words still fall to <unk>.
    Layout preserved: PAD(0), words, unk, eos, specials."""
    from collections import Counter
    cnt = Counter()
    for m, r in train_pairs:
        cnt.update(m.split()); cnt.update(r.split())
    keep = sorted((w for w, c in cnt.items()
                   if c >= min_freq and w not in SPECIALS),
                  key=lambda w: (-cnt[w], w))
    anchors = letter_anchors(warm.size(1), device="cpu")       # mint on CPU
    rows, minted = [warm[PAD]], 0
    for w in keep:
        if w in stoi:
            rows.append(warm[stoi[w]])
        else:
            rows.append(char_fingerprint(w, anchors).to(warm.device))
            minted += 1
    rows += [warm[unk], warm[eos]]
    new_stoi = {w: i + 1 for i, w in enumerate(keep)}          # 0 stays PAD
    new_unk, new_eos = len(keep) + 1, len(keep) + 2
    new_itos = {i + 1: w for i, w in enumerate(keep)}
    new_itos[new_unk] = "<unk>"; new_itos[new_eos] = "<eos>"
    n = new_eos + 1
    for tok in SPECIALS:
        new_stoi[tok] = n; new_itos[n] = tok
        rows.append(warm[stoi[tok]]); n += 1
    return torch.stack(rows), new_stoi, new_itos, new_unk, new_eos, n, minted


def semantic_init(model, stoi, path, device):
    """Semantic initialization layer: overwrite wells with Wikipedia-trained
    collapse geometry (noun_collapse_pure.pt) for every shared word. Warm
    start only — training still moves them; dialogue usage wins over
    encyclopedia usage wherever they disagree."""
    ck = torch.load(path, map_location=device)
    wells = F.normalize(ck["wells"].to(device), dim=-1)
    sstoi = ck["stoi"]
    n = 0
    with torch.no_grad():
        for w, i in stoi.items():
            j = sstoi.get(w)
            if j is not None:
                model.word_anchors[i].copy_(wells[j]); n += 1
    print(f"semantic init: {n:,}/{len(stoi):,} wells <- {path}")


def build_two_view_wells(warm, stoi, noun_path, device):
    """Option B (two views): each word well = [typer 256 | noun 256] concatenated
    into 512-d. The two spaces stay SEPARATE halves — never summed — so their
    different bases don't fight. Cosine in 512-d is just the average of the two
    per-view cosines, so the writer must match a word in BOTH its writing
    geometry and its meaning geometry. Words the noun model never saw get the
    typer view duplicated (they simply have one view). forward re-normalizes."""
    ck = torch.load(noun_path, map_location=device)
    nwells = F.normalize(ck["wells"].to(device), dim=-1)
    nstoi = ck["stoi"]
    typer = F.normalize(warm, dim=-1)
    noun = typer.clone()                        # default: the one view we have
    hits = 0
    with torch.no_grad():
        for w, i in stoi.items():
            j = nstoi.get(w)
            if j is not None:
                noun[i] = nwells[j]; hits += 1
    cat = torch.cat([typer, noun], dim=1)       # (n_words, 512)
    cat[PAD] = 0.0
    return cat, hits


def meaning_weights(train_rep, n_words, alpha):
    """Rarity = meaning, no POS tagger needed: function words are frequent,
    content words are rare. weight = surprisal^alpha, occurrence-mean 1 so
    the loss scale (and lr) stays put. Returns (weights, most-common id)."""
    ids_u, cnts = torch.unique(train_rep, return_counts=True)
    seen = ids_u != PAD
    ids_u, cnts = ids_u[seen], cnts[seen].float()
    W = torch.ones(n_words)
    W[ids_u] = torch.log1p(cnts.sum() / cnts) ** alpha
    W[ids_u] = W[ids_u] / ((W[ids_u] * cnts).sum() / cnts.sum())
    return W, int(ids_u[cnts.argmax()])


class CharMinter:
    """The char layer's door into the reply model: OOV word -> stable well
    minted from its LETTERS (char_fingerprint), id past the trained vocab.
    Deterministic — same word, same well, every run. Read-side only."""

    def __init__(self, dim, n_words, device):
        # mint on CPU: fingerprinting is thousands of tiny ops per word —
        # on MPS that queues a million micro-kernels and looks like a hang.
        # the finished table moves to the device ONCE in table().
        self.anchors = letter_anchors(dim, device="cpu")
        self.base, self.device = n_words, device
        self.ids, self.rows = {}, []

    def id_for(self, word):
        if word not in self.ids:
            self.ids[word] = self.base + len(self.rows)
            self.rows.append(char_fingerprint(word, self.anchors))
        return self.ids[word]

    def table(self):
        if not self.rows:
            return torch.zeros(0, self.anchors.size(1), device=self.device)
        return torch.stack(self.rows).to(self.device)


class WordMinter:
    """OOV context word -> its REAL well from a trained WORD model (e.g.
    noun_collapse_pure.pt), which lives in the same geometry as the reply
    wells. Only words that model has never seen fall back to a char
    fingerprint. Read-side only; ids sit past the trained vocab, exactly like
    CharMinter, so the writer's vocabulary is unchanged."""

    def __init__(self, out_dim, n_words, device, word_path):
        ck = torch.load(word_path, map_location=device)
        self.wells = F.normalize(ck["wells"].to(device), dim=-1).cpu()
        self.src = self.wells.size(1)                      # source width (256)
        self.wstoi = dict(ck["stoi"])
        self.anchors = letter_anchors(self.src, device="cpu")   # char fallback
        self.out_dim, self.reps = out_dim, max(1, out_dim // self.src)
        self.base, self.device = n_words, device
        self.ids, self.rows = {}, []
        self.hits = self.miss = 0

    def id_for(self, word):
        if word not in self.ids:
            self.ids[word] = self.base + len(self.rows)
            j = self.wstoi.get(word)
            if j is not None:
                v = self.wells[j]; self.hits += 1
            else:
                v = char_fingerprint(word, self.anchors); self.miss += 1
            self.rows.append(v.repeat(self.reps) if self.reps > 1 else v)  # tile to out_dim
        return self.ids[word]

    def table(self):
        if not self.rows:
            return torch.zeros(0, self.out_dim, device=self.device)
        return torch.stack(self.rows).to(self.device)


def encode_ctx(sents, stoi, unk, eos, ctx_words=CTX_WORDS, minter=None):
    """Keep the LAST ctx_words tokens (newest survive), + EOS, padded.
    With a minter, OOV words get char-well ids instead of <unk>."""
    maxlen = ctx_words + 2
    out = []
    for s in sents:
        toks = s.split()[-ctx_words:]
        ids = [stoi[t] if t in stoi else
               (minter.id_for(t) if minter is not None else unk) for t in toks]
        ids += [eos] + [PAD] * (maxlen - len(ids) - 1)
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


def decode(ids, itos, eos):
    out = []
    for t in ids.tolist():
        if t == eos or t == PAD:
            break
        out.append(itos.get(t, "?"))
    return " ".join(out) if out else "(empty)"


def trace_str(reply_words, att_row, ctx_words, Lm):
    """The thinking trace: word<-what it looked at. ~word = its own typed word."""
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


# ---------------------------------------------------------------- chat

def chat_loop(args, device):
    """--chat: multi-turn. The conversation so far is fed back as
    <you>/<me>-tagged context — the same shape it trained on."""
    ck = torch.load(args.ckpt, map_location=device)
    cfg, stoi, itos = ck["config"], ck["stoi"], ck["itos"]
    unk, eos = ck["unk"], ck["eos"]
    ctx_words = cfg.get("ctx_words", CTX_WORDS)
    model = ReplyBrain(cfg["n_words"], cfg["dim"], eos,
                       pos=cfg.get("pos", False),
                       pure=cfg.get("pure", False),
                       pos_well=cfg.get("pos_well", False),
                       align=cfg.get("align", False)).to(device)
    model.load_state_dict(ck["state_dict"], strict=False)
    model.pos_w = ck.get("pos_w", 0.0)
    if ck.get("fast_alpha") is not None:
        model.fast_alpha = ck["fast_alpha"].to(device)
    model.eval()
    ban = [stoi[t] for t in SPECIALS if t in stoi]
    oov_src = getattr(args, "oov_words", None) or str(NOUN_CHECKPOINT)
    if os.path.exists(oov_src):
        minter = WordMinter(cfg["dim"], cfg["n_words"], device, oov_src)  # real word wells
    else:
        minter = CharMinter(cfg["dim"], cfg["n_words"], device)           # char fallback
    print(f"loaded {args.ckpt}   ctx {ctx_words} words   device {device}")
    print("multi-turn: it remembers this conversation. :reset to wipe, :q to quit\n")
    from prep_chat_context import clean as clean_text   # same tokenizer as training
    history = []
    while True:
        try:
            line = input("you   > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line == ":q":
            break
        if line == ":reset":
            history = []; print("  (context wiped)\n"); continue
        line = clean_text(line)
        if not line:
            continue
        history.append(("<you>", line))
        toks = []
        for tag, text in history:
            toks.append(tag); toks += text.split()
        ctx = " ".join(toks[-ctx_words:])
        ids = encode_ctx([ctx], stoi, unk, eos, ctx_words, minter=minter).to(device)
        model.oov_wells = minter.table()          # new words -> new read-wells
        gen, att = model.generate(ids, MAXLEN, unk=unk, ban=ban,
                                  rep_penalty=args.rep_penalty,
                                  no_repeat_bigram=args.no_repeat_bigram,
                                  temperature=args.temperature)
        reply = decode(gen[0], itos, eos)
        history.append(("<me>", reply))
        print(f"model > {reply}")
        if att is not None:
            print(f"  thinking: {trace_str(reply.split(), att[0], ctx.split(), ids.shape[1])}")
        print()


# ---------------------------------------------------------------- training

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data", default=data_path("chat_context.tsv"))
    ap.add_argument("--ctx-words", type=int, default=CTX_WORDS,
                    help="context budget; must match prep_chat_context.py")
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.05)
    # optimization
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256,
                    help="256 fits the 256-word context in memory; the growing "
                         "attention memory is held for backward at every reply "
                         "step, so batch x ctx is the real memory knob")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=6,
                    help="early stop after this many epochs without dev improvement")
    # levers (all default-on; 0 / --no-* turns each off)
    ap.add_argument("--min-freq", type=int, default=2,
                    help="cut vocab to words seen >= this often in train (1 = keep all)")
    ap.add_argument("--meaning-w", type=float, default=1.0,
                    help="rare-word loss boost = surprisal^this (0 = uniform CE; "
                         "dev NLL stays unweighted either way)")
    ap.add_argument("--neg-samples", type=int, default=512,
                    help="sampled-softmax negatives in training (0 = full vocab)")
    ap.add_argument("--pos-anneal", type=int, default=20,
                    help="'where am I in the reply' scaffold, faded over this "
                         "many epochs (0 = off)")
    ap.add_argument("--sched-sample", type=float, default=0.25,
                    help="max prob of walking on own picks instead of the truth")
    ap.add_argument("--sched-anneal", type=int, default=10,
                    help="epochs to ramp scheduled sampling from 0 up to max")
    ap.add_argument("--no-fast-reader", action="store_true",
                    help="use the exact sequential read (fast reader is on "
                         "whenever model/fast_reader.pt exists)")
    ap.add_argument("--no-char-wells", action="store_true",
                    help="OOV context words become <unk> instead of reading "
                         "through char-fingerprint wells")
    ap.add_argument("--oov-words", default=None,
                    help="mint OOV context read-wells from this WORD model "
                         "(real in-geometry wells; char fingerprint only for "
                         "words it never saw). Defaults to --semantic-init.")
    # two-stage: general pretrain -> personal fine-tune (see prep_dailydialog.py)
    ap.add_argument("--extra-vocab", default=None,
                    help="second tsv whose words also count in the vocab cut, "
                         "so a later --resume fine-tune shares this vocab")
    ap.add_argument("--resume", default=None,
                    help="continue training from this reply checkpoint — keeps "
                         "its vocab and wells")
    ap.add_argument("--semantic-init", default=None,
                    help="warm shared wells from a Wikipedia collapse model "
                         "(defaults to models/noun-collapse/model/) before training")
    # io
    ap.add_argument("--ckpt", default=CKPT_OUT)
    ap.add_argument("--chat", action="store_true", help="talk to the trained model")
    ap.add_argument("--pure", action="store_true",
                    help="PURE collapse writer: no attention, no MLP, no growing "
                         "memory — O(L) and fast, same engine as the noun model")
    ap.add_argument("--two-views", action="store_true",
                    help="each word well = [typer 256 | noun 256] concatenated "
                         "(512-d). Keeps BOTH representations instead of "
                         "overwriting one with the other.")
    ap.add_argument("--no-pos-well", dest="pos_well", action="store_false",
                    help="disable the positional well in --pure mode (on by "
                         "default; it's what stops the pure bigram loop)")
    ap.set_defaults(pos_well=True)
    ap.add_argument("--align", action="store_true",
                    help="pure cosine-alignment in --pure mode: each step attends "
                         "over the CONTEXT word wells by raw cosine (no learned "
                         "Q/K) and adds the aligned content to the query. Per-word "
                         "context lookup, still wells+cosine. Breaks generic replies.")
    # anti-loop decode levers (chat only; defaults = plain greedy)
    ap.add_argument("--rep-penalty", type=float, default=1.3,
                    help="divide the logit of any already-said word (calms "
                         "loops; 1.0 = off, 1.1-1.3 gentle, 1.5+ firm)")
    ap.add_argument("--no-repeat-bigram", action="store_true", default=True,
                    help="hard-block any 2-word pair from repeating (kills "
                         "'do you do you'); --no-no-repeat-bigram to disable")
    ap.add_argument("--no-no-repeat-bigram", dest="no_repeat_bigram",
                    action="store_false")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sample instead of argmax (0 = greedy; 0.7-1.0 varied)")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    device = (torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    random.seed(SEED); torch.manual_seed(SEED)

    if args.chat:
        chat_loop(args, device)
        return

    # -- data
    data = read_pairs(args.data, args.max_lines)
    random.shuffle(data)
    n_dev = max(1, int(len(data) * args.dev_frac))
    dev, train = data[:n_dev], data[n_dev:]

    # -- model: resumed from a reply ckpt (its vocab + wells), or fresh
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        cfg = ck["config"]
        stoi, itos, unk, eos = ck["stoi"], ck["itos"], ck["unk"], ck["eos"]
        n_words, dim = cfg["n_words"], cfg["dim"]
        model = ReplyBrain(n_words, dim, eos, pos=cfg.get("pos", False),
                           pure=cfg.get("pure", False),
                           pos_well=cfg.get("pos_well", False),
                           align=cfg.get("align", False)).to(device)
        model.load_state_dict(ck["state_dict"], strict=False)
        print(f"resumed {args.resume} (epoch {ck['best']['epoch']}, "
              f"nll {ck['best']['nll']:.4f}) — vocab + wells carried over")
    else:
        print("loading typer wells ...", flush=True)
        warm, stoi, itos, unk, eos, n_words, dim, extras = load_wells(device)
        if args.min_freq > 1:
            vocab_src = train + (read_pairs(args.extra_vocab)
                                 if args.extra_vocab else [])
            full = n_words
            warm, stoi, itos, unk, eos, n_words, minted = shrink_vocab(
                vocab_src, warm, stoi, itos, unk, eos, args.min_freq)
            note = f" + {args.extra_vocab}" if args.extra_vocab else ""
            print(f"vocab cut: {full:,} -> {n_words:,} wells "
                  f"(min-freq {args.min_freq}{note}; {minted:,} fresh wells "
                  f"for words the typer never saw)")
        if args.two_views:
            noun_path = args.oov_words or args.semantic_init or str(NOUN_CHECKPOINT)
            warm, nhits = build_two_view_wells(warm, stoi, noun_path, device)
            dim = warm.size(1)                        # 256 -> 512
            extras["start"] = None                    # start is now 512-d, fresh
            print(f"two views: well = [typer | noun] = {dim}d  "
                  f"({nhits:,} noun-seeded, rest typer-duplicated)")
        model = ReplyBrain(n_words, dim, eos, warm=warm,
                           pos=(args.pos_anneal > 0 and not args.pure),
                           warm_start=(extras["start"].to(device)
                                       if extras["start"] is not None else None),
                           warm_strength=extras["strength"], pure=args.pure,
                           pos_well=args.pos_well, align=args.align).to(device)
    if args.semantic_init and not args.two_views:     # two-views already baked noun in
        semantic_init(model, stoi, args.semantic_init, device)
    model.neg_samples = args.neg_samples
    use_fast = not args.no_fast_reader and os.path.exists(FAST_CKPT)
    if use_fast:
        fr = torch.load(FAST_CKPT, map_location=device)
        model.fast_alpha = fr["alpha"].to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"dim {dim}   device {device}   vocab {n_words:,}   params {n_par:,}")
    rd = (f"FAST ADDITIVE (distilled {model.fast_alpha.numel()} taps)"
          if use_fast else "EXACT collapse walk")
    print(f"reader: {rd}")
    print(f"scaffolds: pos-anneal {args.pos_anneal}   sched-sample {args.sched_sample} "
          f"over {args.sched_anneal} ep   neg-samples {args.neg_samples}", flush=True)

    # -- tensors (context reads through OOV wells; replies stay <unk>)
    oov_src = args.oov_words or args.semantic_init
    if args.no_char_wells:
        minter = None
    elif oov_src:
        minter = WordMinter(dim, n_words, device, oov_src)   # real word wells
    else:
        minter = CharMinter(dim, n_words, device)            # char fallback

    def pre_encode(chunk):
        msg = encode_ctx([m for m, r in chunk], stoi, unk, eos, args.ctx_words,
                         minter=minter)
        rep = encode_batch([r for m, r in chunk], stoi, unk, eos)
        return msg, rep

    train_msg, train_rep = pre_encode(train)
    dev_msg, dev_rep = pre_encode(dev)
    if minter is not None:
        model.oov_wells = minter.table()
        if isinstance(minter, WordMinter):
            print(f"oov wells: {minter.hits:,} real wells <- {oov_src}   "
                  f"{minter.miss:,} char fallback")
        else:
            print(f"char wells: minted {len(minter.rows):,} read-wells for OOV context words")
    print(f"train {len(train)}   dev {len(dev)}\n", flush=True)

    if args.meaning_w > 0:
        W, top = meaning_weights(train_rep, n_words, args.meaning_w)
        model.word_w = W.to(device)
        print(f"meaning weights: '{itos[top]}' x{W[top]:.2f}  vs rarest "
              f"x{W.max():.2f}  (alpha {args.meaning_w})")

    def batches(msg_t, rep_t, shuffle=False):
        n = msg_t.size(0)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, args.batch_size):
            idx = order[i:i + args.batch_size]
            m, r = msg_t[idx], rep_t[idx]
            lm = int((m != 0).sum(1).max()); lr_ = int((r != 0).sum(1).max())
            yield m[:, :max(lm, 1)].to(device), r[:, :max(lr_, 1)].to(device)

    # -- eval helpers
    show = dev[:3]
    ban = [stoi[t] for t in SPECIALS]

    def show_samples():
        model.eval()
        msg = encode_ctx([m for m, r in show], stoi, unk, eos, args.ctx_words,
                         minter=minter).to(device)
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

    # -- the loop
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_nll, best_epoch = float("inf"), 0
    for ep in range(1, args.epochs + 1):
        model.pos_w = (max(0.0, 1.0 - (ep - 1) / args.pos_anneal)
                       if args.pos_anneal > 0 else 0.0)
        model.sample_p = args.sched_sample * min(1.0, (ep - 1) / max(args.sched_anneal, 1))
        model.train()
        run_t = torch.zeros((), device=device); nb = 0
        n_batches = (train_msg.size(0) + args.batch_size - 1) // args.batch_size
        t_mark = t_hb = time.time(); hb_nb = 0
        for msg, rep in batches(train_msg, train_rep, shuffle=True):
            loss = model.reply_nll(msg, rep)
            opt.zero_grad(); loss.backward(); opt.step()
            run_t += loss.detach(); nb += 1
            if nb == 1 or nb % 10 == 0:           # heartbeat: never look stuck
                now = time.time()
                sps = (nb - hb_nb) / max(now - t_hb, 1e-9)
                print(f"  ep{ep} {nb}/{n_batches}  loss {(run_t / nb).item():.4f}"
                      f"  | {sps:.1f} steps/s", flush=True)
                t_hb = now; hb_nb = nb
        nll = evaluate()
        print(f"epoch {ep}: dev reply-nll/word {nll:.4f}   "
              f"[train {(run_t / nb).item():.4f}  pos_w {model.pos_w:.2f}  "
              f"sample_p {model.sample_p:.2f}  {time.time() - t_mark:.0f}s]", flush=True)
        show_samples()
        if nll < best_nll:
            best_nll, best_epoch = nll, ep
            os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "stoi": stoi, "itos": itos,
                        "unk": unk, "eos": eos,
                        "fast_alpha": (model.fast_alpha.cpu()
                                       if model.fast_alpha is not None else None),
                        "config": {"dim": dim, "n_words": n_words,
                                   "ctx_words": args.ctx_words, "pos_well": model.pos_well,
                                   "align": model.align,
                                   "pos": model.pos, "pure": model.pure},
                        "pos_w": model.pos_w,
                        "best": {"epoch": ep, "nll": nll}}, args.ckpt)
            print(f"  [BEST (nll {nll:.4f}) -> saved {args.ckpt}]", flush=True)
        else:
            print(f"  [no improvement -> kept epoch {best_epoch} (nll {best_nll:.4f})]", flush=True)
        if ep - best_epoch >= args.patience:
            print(f"  [early stop: {args.patience} epochs without improvement]", flush=True)
            break

    print(f"\nbest model = epoch {best_epoch} (nll {best_nll:.4f})  ->  {args.ckpt}")
    print("talk to it:  python3 chat_reply.py --chat")


if __name__ == "__main__":
    main()
