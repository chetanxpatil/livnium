"""
score_sentence_typer.py — load the trained sentence_typer.pt and score it on
SNLI sentences (trained + held-out), batched so the logits never blow up memory.
No retraining. Reports per-word and exact-sentence accuracy + examples.

Run from collapse_retrain/:  python3 score_sentence_typer.py
"""

import argparse
import random

import torch
import torch.nn.functional as F

from paths import SNLI_TRAIN

from sentence_typer import (
    HELDOUT, MAX_WORDS, SEED, SentenceTyper,
    decode_pred, encode_batch, read_sentences,
)

CKPT = "sentence_typer.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-path", default=SNLI_TRAIN)
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location=device)
    stoi, itos, unk, eos = ck["stoi"], ck["itos"], ck["unk"], ck["eos"]
    n_words, dim = ck["config"]["n_words"], ck["config"]["dim"]

    model = SentenceTyper(n_words, dim).to(device)
    with torch.no_grad():
        model.word_anchors.copy_(ck["word_anchors"].to(device))
        model.start.copy_(ck["start"].to(device))
        model.log_strength.fill_(torch.logit(torch.tensor(ck["strength"])))
        model.log_temp.fill_(torch.log(torch.expm1(torch.tensor(ck["temp"]))))
    model.eval()
    print(f"loaded {CKPT}  (strength {ck['strength']:.3f}  temp {ck['temp']:.3f}  dim {dim})")

    # rebuild the SAME split (same seed/heldout) the typer trained on
    random.seed(SEED)
    sents = read_sentences(args.nli_path)
    random.shuffle(sents)
    sents = [s for s in sents if len(s.split()) <= MAX_WORDS]
    test_sents, train_sents = sents[:HELDOUT], sents[HELDOUT:]

    def scores(sent_list):
        s = sent_list[:args.cap]
        wh = wt = exact = 0
        with torch.no_grad():
            for j in range(0, len(s), args.bs):
                chunk = s[j:j + args.bs]
                ids = encode_batch(chunk, stoi, unk, eos).to(device)
                states, mask = model.encode(ids)
                pred = model.logits(states).argmax(-1)
                wh += ((pred == ids) & mask).sum().item()
                wt += mask.sum().item()
                for k in range(len(chunk)):
                    if decode_pred(pred[k], itos, eos) == decode_pred(ids[k], itos, eos):
                        exact += 1
        return wh / max(1, wt), exact / len(s)

    tr_w, tr_s = scores(train_sents)
    te_w, te_s = scores(test_sents)
    print("\n--- writing sentences from word wells (pure geometry) ---")
    print(f"  trained : per-word {tr_w*100:5.1f}%   exact-sentence {tr_s*100:5.1f}%")
    print(f"  held-out: per-word {te_w*100:5.1f}%   exact-sentence {te_s*100:5.1f}%")

    print("\n--- held-out examples ---")
    show = test_sents[:10]
    ids = encode_batch(show, stoi, unk, eos).to(device)
    with torch.no_grad():
        pred = model.logits(model.encode(ids)[0]).argmax(-1)
    for k in range(len(show)):
        got = decode_pred(pred[k], itos, eos)
        tgt = decode_pred(ids[k], itos, eos)
        print(f"  {'OK ' if got == tgt else 'XX '}{tgt}")
        if got != tgt:
            print(f"     got -> {got}")


if __name__ == "__main__":
    main()
