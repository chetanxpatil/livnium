"""
noun_probe.py — did MEANING condense in the wells, on its own?

The test: nouns should have KIN (semantic neighbors), function words should
have NOISE. Nothing in training knew what a noun is — if the asymmetry shows,
it emerged from prediction pressure alone.

Compares the reply-trained wells (model/chat_reply.pt — wells that had to
PREDICT) against the typer wells (model/chat_typer.pt — wells that only had
to RECONSTRUCT). If prediction sharpened the neighborhoods, meaning lives in
the reply model's geometry.

Usage:
    python3 noun_probe.py
    python3 noun_probe.py --words livnium collapse atom gravity
"""

import argparse

import torch
import torch.nn.functional as F

from paths import model_path

NOUNS = ["livnium", "collapse", "atom", "frequency", "gravity", "torch",
         "cube", "python", "energy", "entropy"]
FUNCTION = ["the", "or", "and", "of", "is"]


def neighbors(word, A, stoi, itos, k=8):
    if word not in stoi:
        return None
    v = A[stoi[word]]
    sims = A @ v
    sims[stoi[word]] = -1e9
    top = sims.topk(k)
    return [(itos.get(int(i), "?"), float(s)) for s, i in zip(top.values, top.indices)]


def load(path, key):
    ck = torch.load(path, map_location="cpu")
    W = ck[key] if key in ck else ck["state_dict"]["word_anchors"]
    A = F.normalize(W, dim=-1)
    itos = ck["itos"] if "itos" in ck else {i: w for w, i in ck["stoi"].items()}
    return A, ck["stoi"], itos


def show(title, words, A, stoi, itos):
    print(f"\n=== {title} ===")
    for w in words:
        n = neighbors(w, A, stoi, itos)
        if n is None:
            print(f"  {w:12s} (not in vocab)")
            continue
        s = "  ".join(f"{x}({c:.2f})" for x, c in n)
        print(f"  {w:12s} -> {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="*", default=None,
                    help="extra nouns to probe")
    args = ap.parse_args()
    nouns = args.words if args.words else NOUNS

    print("REPLY-TRAINED WELLS (had to predict — where meaning should live)")
    A, stoi, itos = load(model_path("chat_reply.pt"), "word_anchors")
    show("nouns (expect KIN)", nouns, A, stoi, itos)
    show("function words (expect NOISE — the control)", FUNCTION, A, stoi, itos)

    print("\n\nTYPER WELLS (only had to reconstruct — the before picture)")
    A2, stoi2, itos2 = load(model_path("chat_typer.pt"), "word_anchors")
    show("same nouns", nouns, A2, stoi2, itos2)


if __name__ == "__main__":
    main()
