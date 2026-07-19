"""
train_char_collapse.py — proof harness for the position-aware CharCollapse.

It trains CharCollapse as a character autoencoder (encode a word into a single
SPLIT vector via position-aware collapse, then decode the characters back) and
checks three claims:

  1. COMPOSITIONALITY / ZERO-OOV
     The decoder reconstructs words it has NEVER seen during training. Because a
     word is built from shared letter anchors, unseen words are just new routes
     through known wells.

  2. POSITION-AWARENESS (the whole point)
     For a word and a shuffled anagram of it:
        - the FULL view (bag of letters) is IDENTICAL  (cos ~ 1.0) — order-blind,
        - the SPLIT view (the trajectory) is DIFFERENT (cos well below 1.0).
     This proves the sequential collapse encodes order, not just letter content.

  3. STABILITY
     The collapse trajectory stays bounded (norm clamp never fights the
     dynamics into the wall on every step).

Runs on CPU in well under a minute. No external data needed.
"""

import random

import torch

from char_collapse import CharCollapse

SEED = 0
DIM = 64
MAX_LEN = 12
N_TRAIN = 2000
N_TEST = 400
STEPS = 400
BATCH = 128
LR = 3e-3


def make_word(rng: random.Random) -> str:
    """A pronounceable-ish random word: alternating-ish consonant/vowel."""
    vowels = "aeiou"
    cons = "bcdfghjklmnpqrstvwxyz"
    n = rng.randint(3, MAX_LEN)
    out = []
    for i in range(n):
        out.append(rng.choice(vowels if i % 2 else cons))
    return "".join(out)


def make_vocab(rng: random.Random, n: int) -> list:
    seen = set()
    words = []
    while len(words) < n:
        w = make_word(rng)
        if w not in seen:
            seen.add(w)
            words.append(w)
    return words


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    # Disjoint train/test vocabularies — test words are never seen in training.
    all_words = make_vocab(rng, N_TRAIN + N_TEST)
    train_words = all_words[:N_TRAIN]
    test_words = all_words[N_TRAIN:]

    model = CharCollapse(dim=DIM, max_len=MAX_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_ids = model.vocab.encode_batch(train_words, MAX_LEN)

    print(f"alphabet size      : {len(model.vocab)} (incl. PAD)")
    print(f"train / test words : {len(train_words)} / {len(test_words)} (disjoint)")
    print(f"dim                : {DIM}\n")

    model.train()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(train_words), (BATCH,))
        batch = train_ids[idx]
        loss, correct = model.reconstruction_loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            acc = correct.sum().item() / max(1.0, (batch != model.pad_idx).sum().item())
            print(f"step {step:4d}  loss {loss.item():.4f}  train char-acc {acc*100:5.1f}%")

    # ---- claim 1: reconstruction on UNSEEN words -----------------------------
    model.eval()
    with torch.no_grad():
        def char_accuracy(words):
            ids = model.vocab.encode_batch(words, MAX_LEN)
            _, c = model.reconstruction_loss(ids)
            return c.sum().item() / max(1.0, (ids != model.pad_idx).sum().item())

        def exact_word_accuracy(words):
            ids = model.vocab.encode_batch(words, MAX_LEN)
            z, _, _, _ = model.encode(ids)
            logits = model.decode(z, ids.size(1))
            pred = logits.argmax(-1)
            hits = 0
            for k, w in enumerate(words):
                if model.vocab.decode(pred[k].tolist()) == w:
                    hits += 1
            return hits / len(words)

        tr_char = char_accuracy(train_words[:N_TEST])
        te_char = char_accuracy(test_words)
        tr_word = exact_word_accuracy(train_words[:N_TEST])
        te_word = exact_word_accuracy(test_words)

    print("\n--- Claim 1: compositional reconstruction (zero-OOV) ---")
    print(f"  train  char-acc {tr_char*100:5.1f}%   exact-word {tr_word*100:5.1f}%")
    print(f"  UNSEEN char-acc {te_char*100:5.1f}%   exact-word {te_word*100:5.1f}%")

    # ---- claim 2: position-awareness via anagrams ----------------------------
    print("\n--- Claim 2: position-awareness (SPLIT vs FULL on anagrams) ---")
    cos = torch.nn.functional.cosine_similarity
    pairs = []
    rng2 = random.Random(123)
    for w in test_words:
        if len(set(w)) < 3:
            continue
        chars = list(w)
        shuf = chars[:]
        while "".join(shuf) == w:
            rng2.shuffle(shuf)
        pairs.append((w, "".join(shuf)))
        if len(pairs) >= 200:
            break

    with torch.no_grad():
        a_words = [p[0] for p in pairs]
        b_words = [p[1] for p in pairs]
        ids_a = model.vocab.encode_batch(a_words, MAX_LEN)
        ids_b = model.vocab.encode_batch(b_words, MAX_LEN)
        za_seq, za_bag, _, _ = model.encode(ids_a)
        zb_seq, zb_bag, _, _ = model.encode(ids_b)
        seq_cos = cos(za_seq, zb_seq, dim=-1)
        bag_cos = cos(za_bag, zb_bag, dim=-1)

    print(f"  anagram pairs tested : {len(pairs)}")
    print(f"  FULL  view cos(word, anagram): mean {bag_cos.mean():.4f}  "
          f"(should be ~1.000 — order-blind)")
    print(f"  SPLIT view cos(word, anagram): mean {seq_cos.mean():.4f}  "
          f"min {seq_cos.min():.4f}  max {seq_cos.max():.4f}  "
          f"(should be < 1 — order-sensitive)")
    sep = (bag_cos.mean() - seq_cos.mean()).item()
    print(f"  separation (FULL - SPLIT)    : {sep:.4f}  "
          f"-> {'PASS' if sep > 0.05 else 'FAIL'}")

    for w, s in pairs[:5]:
        ia = model.vocab.encode_batch([w], MAX_LEN)
        ib = model.vocab.encode_batch([s], MAX_LEN)
        sa, ba, _, _ = model.encode(ia)
        sb, bb, _, _ = model.encode(ib)
        print(f"    {w:>12s} vs {s:<12s}  "
              f"SPLIT cos {cos(sa, sb, -1).item():.3f}  "
              f"FULL cos {cos(ba, bb, -1).item():.3f}")

    # ---- claim 3: trajectory stability ---------------------------------------
    print("\n--- Claim 3: trajectory stability ---")
    with torch.no_grad():
        ids = model.vocab.encode_batch(test_words[:256], MAX_LEN)
        _, _, _, path = model.encode(ids)
        norms = torch.stack([p.norm(dim=-1) for p in path])  # (L, B)
        print(f"  state norm over trajectory: mean {norms.mean():.3f}  "
              f"max {norms.max():.3f}  (clamp wall = 10.0)")


if __name__ == "__main__":
    main()
