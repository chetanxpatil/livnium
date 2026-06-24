"""
eval_nli_meaning_forms_symbols.py — score the saved best checkpoint on the REAL
SNLI test set (snli_1.0_test.jsonl), which the model never saw in train or dev.

Reuses the exact vocab, symbol scaffold, and model class from the trainer, loads
nli_meaning_forms_symbols.pt (the best-dev checkpoint), and reports overall test
accuracy plus a per-class breakdown so you can see where it's right/wrong.

Run from collapse_retrain/:
    python3 eval_nli_meaning_forms_symbols.py
    python3 eval_nli_meaning_forms_symbols.py --nli-path /path/to/snli_1.0_test.jsonl
"""

import argparse
import os
import random

import torch

from train_nli_meaning_forms_symbols import (
    NLI_LABEL_TO_IDX, MeaningForms, build_char_scaffold,
    collate, encode_line, load_vocab, read_nli_jsonl,
)

CKPT = "nli_meaning_forms_symbols.pt"
IDX_TO_LABEL = {v: k for k, v in NLI_LABEL_TO_IDX.items()}
TRAIN_PATH = "/Users/chetanpatil/Desktop/test/data-bank/snli_1.0_train.jsonl"


def find_eval_file(given):
    """Return (path, kind). Prefer a real test file, then dev; both are unseen."""
    if given and os.path.exists(given):
        return given, "given"
    folder = os.path.dirname(TRAIN_PATH)
    for name, kind in [("snli_1.0_test.jsonl", "TEST"),
                       ("snli_1.0_dev.jsonl", "DEV (unseen)")]:
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p, kind
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default="/Users/chetanpatil/Desktop/test/data-bank/snli_1.0_test.jsonl")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    idx2word, w2i, pad_idx, unk_idx = load_vocab()
    scaffold, dim = build_char_scaffold(idx2word, pad_idx)

    model = MeaningForms(scaffold, pad_idx, dim).to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {CKPT}  (saved at epoch {ck.get('epoch','?')}, dev {ck.get('dev',0)*100:.2f}%)")
    print(f"device {device}\n")

    path, kind = find_eval_file(args.nli_path)
    if path is None:
        # no test/dev file on disk -> rebuild the exact held-out dev split (seed 0, frac 0.02)
        print("no test/dev file found; rebuilding the held-out split the model never trained on")
        raw = read_nli_jsonl(TRAIN_PATH)
        data = [(encode_line(s1, w2i, unk_idx), encode_line(s2, w2i, unk_idx), y) for s1, s2, y in raw]
        random.seed(0); random.shuffle(data)
        data = data[:int(len(data) * 0.02)]
        kind = "HELD-OUT DEV (reconstructed, unseen in training)"
    else:
        print(f"eval file: {path}  [{kind}]")
        raw = read_nli_jsonl(path)
        data = [(encode_line(s1, w2i, unk_idx), encode_line(s2, w2i, unk_idx), y) for s1, s2, y in raw]
    print(f"examples ({kind}): {len(data)}")

    ok = tot = 0
    per = {k: [0, 0] for k in range(3)}            # label -> [correct, total]
    conf = [[0, 0, 0] for _ in range(3)]           # true -> pred counts
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
            p, h, y = collate(data[i:i + args.batch_size], pad_idx)
            pred = model(p.to(device), h.to(device)).argmax(-1).cpu()
            for t, pr in zip(y.tolist(), pred.tolist()):
                conf[t][pr] += 1
                per[t][1] += 1
                if t == pr:
                    per[t][0] += 1; ok += 1
                tot += 1

    print(f"\n=== {kind} accuracy: {ok/tot*100:.2f}%  ({ok}/{tot}) ===\n")
    print("per-class recall:")
    for k in range(3):
        c, n = per[k]
        print(f"  {IDX_TO_LABEL[k]:>13s}: {c/max(1,n)*100:5.2f}%  ({c}/{n})")

    print("\nconfusion (rows = true, cols = predicted E / N / C):")
    print(f"  {'':>13s}    E     N     C")
    for k in range(3):
        r = conf[k]
        print(f"  {IDX_TO_LABEL[k]:>13s}  {r[0]:5d} {r[1]:5d} {r[2]:5d}")


if __name__ == "__main__":
    main()
