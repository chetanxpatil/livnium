"""
sentence_typer.py — the char typer, one level up, STACKED on char geometry.

WORDS are wells, the SENTENCE is the trajectory. No NLI. No labels. Nothing but
learning to WRITE the sentence back from the word-collapse path.

THE LADDER (char step retained as INITIALIZATION, then learned):
    char_collapse_pure.py : letters are wells -> letter anchors (loaded frozen)
    this file             : each word's WELL is SEEDED from the char geometry,
                            then trained to write sentences back.

Each word well starts at its ORDER-AWARE char fingerprint — a position-weighted
sum of its letters' char anchors (from char_typer_pure.pt), so anagrams don't
collide. That seeding is the retained char step. The wells are then TRAINABLE
and move under the reconstruction loss:

    word well  W_word  : init = char_seq(word)   then learned
    also learnable     : start state, strength scalar, temp scalar

So the char geometry decides where every word STARTS; training decides where it
ends up. Nothing is an MLP/GRU/pooling — still pure wells + cosine.

ENCODE (one attraction step per word, in order):
    h <- h - strength * (1 - cos(h, W_word)) * normalize(h - W_word)
A state is kept per word position so the sentence can be unrolled.

DECODE (pure geometry, no network):
    at position i: cos(state_i, EVERY word well) -> pick nearest word.
    type words, joining them, and STOP at the EOS well.

Trained only on SNLI *sentences* (premises + hypotheses), labels ignored.

Run from reached/pure/:  python3 sentence_typer.py
Needs torch + model/char_typer_pure.pt (run char_collapse_pure.py first).
"""

import argparse
import json
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import char_collapse_pure as cp   # the frozen char stage (letters -> word)

SEED = 0
DIM = 256                 # MUST match the char model (char endpoints inherit this dim)
MAX_WORDS = 32            # longest sentence kept (words), EOS appended after
MAXLEN = 34               # words + EOS, then padded
STEPS_PER_WORD = 1        # 1 = mirror the char typer exactly
STEPS = 6000
BATCH = 128
LR = 5e-3
HELDOUT = 2000
MAX_VOCAB = 20000         # keep the most frequent words as wells; rest -> <unk>

PAD = 0
CHAR_CKPT = "model/char_typer_pure.pt"

LETTERS = set("abcdefghijklmnopqrstuvwxyz")


def read_sentences(path, max_lines=0):
    """Just the sentences. Labels are thrown away — this is unsupervised typing."""
    sents, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if max_lines and len(sents) >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for key in ("sentence1", "sentence2"):
                s = (d.get(key) or "").lower().strip()
                if s and s not in seen:
                    seen.add(s)
                    sents.append(s)
    return sents


def build_word_vocab(sents, max_vocab):
    """One well per word (the most frequent ones), plus PAD(0), <unk>, EOS."""
    cnt = Counter(t for s in sents for t in s.split())
    keep = [w for w, _ in cnt.most_common(max_vocab)]
    stoi = {w: i + 1 for i, w in enumerate(keep)}     # 0 = PAD
    unk = len(keep) + 1
    eos = len(keep) + 2
    itos = {i: w for w, i in stoi.items()}
    itos[unk] = "<unk>"; itos[eos] = "<eos>"
    n_words = len(keep) + 3                            # PAD + words + UNK + EOS
    return stoi, itos, unk, eos, n_words


def encode_batch(sents, stoi, unk, eos):
    out = []
    for s in sents:
        ids = [stoi.get(t, unk) for t in s.split()][:MAX_WORDS] + [eos]
        ids += [PAD] * (MAXLEN - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)


def decode_pred(row, itos, eos):
    out = []
    for t in row:
        t = int(t)
        if t == eos or t == PAD:
            break
        out.append(itos.get(t, "?"))
    return " ".join(out)


# --------------------------------------------------------------------------- #
#  The retained char step: build every word's well from char_typer_pure.pt    #
# --------------------------------------------------------------------------- #
def load_char_model(device):
    """Rebuild the FROZEN pure char typer from its checkpoint."""
    ck = torch.load(CHAR_CKPT, map_location=device)
    dim = ck["config"]["dim"]
    assert dim == DIM, f"char model dim {dim} != sentence DIM {DIM}"
    char = cp.CharCollapsePure(dim).to(device)
    with torch.no_grad():
        char.letter_anchors.copy_(ck["letter_anchors"].to(device))
        char.start.copy_(ck["start"].to(device))
        char.log_strength.fill_(torch.logit(torch.tensor(ck["strength"])))
        char.log_temp.fill_(torch.log(torch.expm1(torch.tensor(ck["temp"]))))
    for p in char.parameters():
        p.requires_grad_(False)
    char.eval()
    return char


# Fixed, distinct per-position weights (deterministic = pure). Because each slot
# has a different weight, two words with the SAME letters in a DIFFERENT order
# (anagrams) get different vectors — the seed is order-aware.
_POSW = torch.linspace(1.2, 0.4, cp.MAX_WORD)


def char_seq(words, char, device, bind="weighted"):
    """Order-aware char fingerprint = bound sum of letter anchors.

    bind="weighted" (original): word_vec = sum_i POSW[i] * normalize(anchor[c_i])
        each position carries a distinct scalar weight. Weak order coding —
        anagrams stay almost collinear (~0.99 cos).

    bind="roll" (dimensional binding): word_vec = sum_i ROLL_i(normalize(anchor[c_i]))
        each position i cyclically rotates the letter anchor by i*stride dims, so
        position acts as a near-orthogonal ROLE (permutation binding, VSA-style).
        Same letters in different slots get rotated by different amounts and
        decorrelate, so anagrams ('listen' vs 'silent') separate strongly.

    Used as the INITIALIZATION (and frozen tie target) for the trainable wells.
    """
    A = F.normalize(char.letter_anchors.detach(), dim=-1)   # (N_CHARS, dim)
    posw = _POSW.to(device)
    stride = max(1, A.size(1) // cp.MAX_WORD)               # spread shifts across the dim
    out = torch.zeros(len(words), A.size(1), device=device)
    for k, w in enumerate(words):
        ids = [cp.STOI[c] for c in w if c in LETTERS][:cp.MAX_WORD]
        if ids:
            idx = torch.tensor(ids, device=device)
            vecs = A[idx]                                   # (L, dim) normalized
            if bind == "roll":
                bound = torch.stack(
                    [torch.roll(vecs[i], shifts=i * stride, dims=-1)
                     for i in range(len(ids))], dim=0)
                out[k] = bound.sum(0)
            else:
                out[k] = (posw[:len(ids)].unsqueeze(-1) * vecs).sum(0)
    return out


def build_word_init(itos, stoi, unk, eos, n_words, device, bs=2048, bind="weighted"):
    """Initial (n_words, dim) well table seeded from the char geometry.

    real word -> its order-aware char fingerprint (position-weighted anchors)
    EOS       -> the char model's own stop well (letter_anchors[cp.EOS])
    UNK       -> opposite the word centroid (far from every real word)
    PAD       -> zeros (masked out everywhere)

    These are only the STARTING positions — the wells are trainable and move.
    """
    char = load_char_model(device)
    wells = torch.zeros(n_words, DIM, device=device)

    words = [(i, itos[i]) for i in range(n_words) if i in itos and i not in (unk, eos)]
    for j in range(0, len(words), bs):
        chunk = words[j:j + bs]
        vecs = char_seq([w for _, w in chunk], char, device, bind=bind)
        for (i, _), v in zip(chunk, vecs):
            wells[i] = v
    wells[eos] = F.normalize(char.letter_anchors[cp.EOS].detach(), dim=-1)
    real = wells[1:unk]                       # rows 1..unk-1 are the real words
    wells[unk] = -F.normalize(real.mean(0), dim=-1)
    return wells


def anagram_report(device, bind="weighted"):
    """Prove the order-aware seed gives anagrams DIFFERENT birth positions.

    cos = 1.0 would mean identical seeds (the old order-blind bag); we want < 1.
    """
    char = load_char_model(device)
    groups = [["listen", "silent", "tinsel"],
              ["stop", "tops", "pots", "spot"],
              ["act", "cat"]]
    print(f"--- anagram birth-position check (bind={bind}) ---")
    for g in groups:
        V = F.normalize(char_seq(g, char, device, bind=bind), dim=-1)
        sims = V @ V.t()
        pairs = [f"{g[i]}~{g[j]} {sims[i, j]:.3f}"
                 for i in range(len(g)) for j in range(i + 1, len(g))]
        print("  " + "   ".join(pairs))
    print("  (1.000 = collide; lower = separated at birth)\n")


class SentenceTyper(nn.Module):
    """Word wells TRAINABLE (sentence rule) but tied to char geometry (char rule).

    word_anchors : trainable, seeded from char geometry in main()
    char_init    : FROZEN reference (the char positions); the tie penalty pulls
                   word_anchors back toward it so BOTH rules hold at once.
    """

    def __init__(self, n_words, dim=DIM):
        super().__init__()
        self.n_words = n_words
        # TRAINABLE word wells — seeded from char geometry in main(), then learned.
        self.word_anchors = nn.Parameter(torch.randn(n_words, dim) * (1.0 / dim ** 0.5))
        # FROZEN char reference — what the char rule wants the wells to stay near.
        self.register_buffer("char_init", torch.zeros(n_words, dim))
        self.start = nn.Parameter(torch.randn(dim) * 0.05)
        self.log_strength = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    @property
    def strength(self):
        return torch.sigmoid(self.log_strength)

    @property
    def temp(self):
        return F.softplus(self.log_temp) + 1e-3

    def encode(self, word_ids):
        if word_ids.dim() == 1:
            word_ids = word_ids.unsqueeze(0)
        B, L = word_ids.shape
        anchors = F.normalize(self.word_anchors, dim=-1)
        mask = (word_ids != PAD)
        h = self.start.expand(B, -1).contiguous()
        states = []
        s = self.strength
        for i in range(L):
            target = anchors[word_ids[:, i]]
            m = mask[:, i].float().unsqueeze(-1)
            for _ in range(STEPS_PER_WORD):
                align = (F.normalize(h, dim=-1) * target).sum(-1)
                div = 1.0 - align
                away = F.normalize(h - target, dim=-1)
                step = -s * div.unsqueeze(-1) * away
                h = h + m * step
                n = h.norm(dim=-1, keepdim=True)
                h = torch.where(n > 10.0, h * (10.0 / (n + 1e-8)), h)
            states.append(h)
        return torch.stack(states, dim=1), mask

    def logits(self, states):
        anchors = F.normalize(self.word_anchors, dim=-1)
        sn = F.normalize(states, dim=-1)
        return (sn @ anchors.t()) / self.temp

    def char_pull(self, word_ids):
        """The char rule: how far the wells used in this batch have drifted
        (in direction) from their char-geometry positions. 0 = perfectly faithful."""
        uniq = word_ids[word_ids != PAD].unique()
        if uniq.numel() == 0:
            return self.word_anchors.new_zeros(())
        return (1.0 - F.cosine_similarity(self.word_anchors[uniq],
                                          self.char_init[uniq], dim=-1)).mean()

    def loss(self, word_ids, lam=0.0):
        """sentence rule (reconstruct) + lam * char rule (stay near char geometry)."""
        states, _ = self.encode(word_ids)
        logits = self.logits(states)
        ce = F.cross_entropy(
            logits.reshape(-1, self.n_words), word_ids.reshape(-1), ignore_index=PAD
        )
        pull = self.char_pull(word_ids)
        return ce + lam * pull, ce, pull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default="/Users/chetanpatil/Desktop/test/data-bank/snli_1.0_train.jsonl")
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--max-vocab", type=int, default=MAX_VOCAB)
    ap.add_argument("--tie", type=float, default=1.0,
                    help="char-rule weight: 0 = pure alive (drift freely), "
                         "high = follow char geometry tightly")
    ap.add_argument("--bind", choices=["weighted", "roll"], default="weighted",
                    help="char->word binding: weighted = scalar position weights "
                         "(original); roll = permutation binding (VSA, anagram-safe)")
    args = ap.parse_args()

    random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    sents = read_sentences(args.nli_path, args.max_lines)
    random.shuffle(sents)
    sents = [s for s in sents if len(s.split()) <= MAX_WORDS]

    test_sents = sents[:HELDOUT]
    train_sents = sents[HELDOUT:]

    # Build vocab from train_sents only to prevent held-out leakage
    stoi, itos, unk, eos, n_words = build_word_vocab(train_sents, args.max_vocab)
    print(f"unique SNLI sentences   : {len(sents)}")
    print(f"train / held-out        : {len(train_sents)} / {len(test_sents)}")
    print(f"word wells (TRAINABLE, tied to char geometry): {n_words-3} words (+PAD+UNK+EOS = {n_words})")
    print(f"dim {DIM}   steps/word {STEPS_PER_WORD}   tie {args.tie}   bind {args.bind}   device {device}   (MLP: NONE, GRU: NONE)\n")

    model = SentenceTyper(n_words).to(device)
    with torch.no_grad():
        init = build_word_init(itos, stoi, unk, eos, n_words, device, bind=args.bind)
        model.char_init.copy_(init)        # frozen char reference (char rule)
        model.word_anchors.copy_(init)     # trainable wells start here (sentence rule)
    print("word wells SEEDED from char geometry; both rules active "
          f"(sentence reconstruct + char tie x{args.tie}, bind={args.bind}).\n")
    anagram_report(device, bind=args.bind)

    # EVERYTHING learns: word wells + start + strength + temp. char_init stays frozen.
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_ids = encode_batch(train_sents, stoi, unk, eos).to(device)

    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train_sents), (args.batch,), device=device)
        loss, ce, pull = model.loss(train_ids[idx], lam=args.tie)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  ce {ce.item():.4f}  "
                  f"char-drift {pull.item():.4f}  strength {model.strength.item():.3f}")

    model.eval()

    def scores(sent_list, cap=3000, bs=256, shuffle_words=False):
        s = sent_list[:cap]
        if shuffle_words:
            rng = random.Random(SEED)            # reproducible, isolated from global stream
            shuffled_s = []
            for text in s:
                words = text.split()
                rng.shuffle(words)
                shuffled_s.append(" ".join(words))
            s = shuffled_s

        word_hits = word_tot = exact = 0
        exact_clean = clean_tot = 0

        with torch.no_grad():
            for j in range(0, len(s), bs):
                chunk = s[j:j + bs]
                ids = encode_batch(chunk, stoi, unk, eos).to(device)
                states, mask = model.encode(ids)
                lg = model.logits(states)
                lg[..., PAD] = float("-inf"); lg[..., unk] = float("-inf")  # never type junk
                pred = lg.argmax(-1)

                word_hits += ((pred == ids) & mask).sum().item()
                word_tot += mask.sum().item()

                for k in range(len(chunk)):
                    got = decode_pred(pred[k], itos, eos)
                    tgt = decode_pred(ids[k], itos, eos)

                    if got == tgt:
                        exact += 1

                    # Track clean accuracy for sentences without OOV tokens
                    if not (ids[k] == unk).any().item():
                        clean_tot += 1
                        if got == tgt:
                            exact_clean += 1

        return (word_hits / max(1, word_tot),
                exact / len(s),
                exact_clean / max(1, clean_tot))

    tr_w, tr_s, tr_clean = scores(train_sents)
    te_w, te_s, te_clean = scores(test_sents)
    shuf_w, shuf_s, shuf_clean = scores(test_sents, shuffle_words=True)

    print("\n--- PURE GEOMETRY, char-derived wells — writing sentences ---")
    print(f"  trained : per-word {tr_w*100:5.1f}%   exact-sentence {tr_s*100:5.1f}%   (clean OOV-free: {tr_clean*100:5.1f}%)")
    print(f"  held-out: per-word {te_w*100:5.1f}%   exact-sentence {te_s*100:5.1f}%   (clean OOV-free: {te_clean*100:5.1f}%)")
    print(f"  shuffled: per-word {shuf_w*100:5.1f}%   exact-sentence {shuf_s*100:5.1f}%   <- order-dependency baseline")
    print(f"  learned strength {model.strength.item():.3f}   temp {model.temp.item():.3f}")

    print("\n--- example sentences typed back (held-out) ---")
    show = test_sents[:8]
    ids = encode_batch(show, stoi, unk, eos).to(device)
    with torch.no_grad():
        lg = model.logits(model.encode(ids)[0])
        lg[..., PAD] = float("-inf"); lg[..., unk] = float("-inf")  # never type junk
        pred = lg.argmax(-1)
    for k in range(len(show)):
        got = decode_pred(pred[k], itos, eos)
        tgt = decode_pred(ids[k], itos, eos)
        mark = "OK " if got == tgt else "XX "
        print(f"  {mark}{tgt}")
        if got != tgt:
            print(f"     got -> {got}")

    # how well BOTH rules ended up holding
    with torch.no_grad():
        faith = F.cosine_similarity(model.word_anchors[1:unk],
                                    model.char_init[1:unk], dim=-1).mean().item()
    print("\n--- both rules at the end ---")
    print(f"  sentence rule: per-word reconstruction (above)")
    print(f"  char rule    : mean cos(word well, char position) = {faith:.3f}   (1.0 = fully faithful)")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- where the knowledge lives ({n_train} trainable numbers) ---")
    print(f"  word_anchors {tuple(model.word_anchors.shape)} = TRAINABLE, tied to char (order-aware seed)")
    print(f"  + start + strength + temp     MLP: 0   GRU: 0")

    torch.save({"word_anchors": model.word_anchors.detach().cpu(),
                "start": model.start.detach().cpu(),
                "strength": model.strength.item(), "temp": model.temp.item(),
                "stoi": stoi, "itos": itos, "unk": unk, "eos": eos,
                "config": {"dim": DIM, "n_words": n_words, "bind": args.bind}}, "model/sentence_typer.pt")
    print("\nsaved -> model/sentence_typer.pt")


if __name__ == "__main__":
    main()
