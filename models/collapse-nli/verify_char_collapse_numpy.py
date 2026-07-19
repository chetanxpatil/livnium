"""
Forward-only numpy port of CharCollapse dynamics — verifies the structural
claim (position-awareness) WITHOUT needing torch/gradients.

It mirrors char_collapse.py exactly:
    h <- h + delta(h + pos_i) - strength * (1 - cos(h, A_c)) * normalize(h - A_c)
and the order-free FULL view = mean of letter anchors.
"""
import numpy as np

rng = np.random.default_rng(0)
DIM, MAXLEN, STRENGTH = 64, 16, 0.1
CHARS = "abcdefghijklmnopqrstuvwxyz"
stoi = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = PAD

# parameters (random init, exactly like the module)
anchors = rng.standard_normal((27, DIM))
anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
pos = rng.standard_normal((MAXLEN, DIM)) * 0.02
start = rng.standard_normal(DIM) * 0.02
W1 = rng.standard_normal((DIM, DIM)) / np.sqrt(DIM); b1 = np.zeros(DIM)


def norm(v):
    return v / (np.linalg.norm(v) + 1e-8)


def update(x, last_layer_random):
    g = np.tanh(x @ W1 + b1)
    if last_layer_random is None:
        return np.zeros_like(x)              # zero-init: faithful to module init
    W2, b2 = last_layer_random
    return g @ W2 + b2


def encode_seq(word, last_layer):
    h = start.copy()
    for i, ch in enumerate(word):
        target = anchors[stoi[ch]]
        delta = update(h + pos[i], last_layer)
        a = norm(h) @ target
        away = norm(h - target)
        h = h + delta - STRENGTH * (1.0 - a) * away
        n = np.linalg.norm(h)
        if n > 10.0:
            h = h * (10.0 / (n + 1e-8))
    return h


def encode_bag(word):
    return np.mean([anchors[stoi[c]] for c in word], axis=0)


def cos(a, b):
    return float(norm(a) @ norm(b))


def anagram_report(tag, last_layer):
    rng2 = np.random.default_rng(123)
    words = ["chetan", "livnium", "collapse", "entailment", "neutral",
             "anchor", "gravity", "trajectory", "sequence", "position"]
    seq_cos, bag_cos = [], []
    print(f"\n=== {tag} ===")
    for w in words:
        s = list(w)
        while "".join(s) == w:
            rng2.shuffle(s)
        s = "".join(s)
        sc = cos(encode_seq(w, last_layer), encode_seq(s, last_layer))
        bc = cos(encode_bag(w), encode_bag(s))
        seq_cos.append(sc); bag_cos.append(bc)
        print(f"  {w:>11s} vs {s:<11s}  SPLIT cos {sc:+.3f}   FULL cos {bc:+.3f}")
    print(f"  ---- mean SPLIT {np.mean(seq_cos):+.3f}   mean FULL {np.mean(bag_cos):+.3f}")
    print(f"  separation (FULL - SPLIT) = {np.mean(bag_cos) - np.mean(seq_cos):+.3f}")


# 1) faithful module init: last layer zero -> delta=0, order-sensitivity comes
#    purely from the SEQUENCE of collapse attractions.
anagram_report("Faithful init (delta=0): collapse-sequence ordering only", None)

# 2) active update path: small random last layer -> position embeddings now
#    also feed the dynamics (what you get after training moves the MLP off zero).
W2 = rng.standard_normal((DIM, DIM)) / np.sqrt(DIM)
b2 = np.zeros(DIM)
anagram_report("Active update path: position signal engaged via MLP", (W2, b2))

# stability check
norms = [np.linalg.norm(encode_seq(w, None)) for w in
         ["a" * k for k in range(1, MAXLEN + 1)]]
print(f"\nstability: endpoint norms across lengths 1..{MAXLEN}: "
      f"min {min(norms):.2f} max {max(norms):.2f} (clamp wall = 10.0)")
