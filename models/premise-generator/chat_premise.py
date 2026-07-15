"""
chat_premise.py — talk to the trained premise generator.

You type a HYPOTHESIS, the model types back a PREMISE generated under a fixed
label (neutral by default). It's the same free-running collapse decode used in
the per-epoch samples, just interactive.

Usage:
    python3 chat_premise.py                      # neutral label (default)
    python3 chat_premise.py --label entail
    python3 chat_premise.py --ckpt model/premise_from_hyp_align.pt

In the prompt:
    <type any sentence>   -> get a premise back
    :label neutral        -> switch the fixed label (entail / neutral / contra)
    :trace                -> toggle the alignment 'thinking' trace
    :q                    -> quit
"""

import argparse
import os
import torch

from sentence_typer import encode_batch, MAXLEN
from premise_from_hyp import PremiseBrain, N_LABELS

LABELS = {"entail": 0, "neutral": 1, "contra": 2}
NAMES = ["entail", "neutral", "contra"]


def main():
    ap = argparse.ArgumentParser()
    default_ckpt = os.path.join(os.path.dirname(__file__), "model", "premise_from_hyp_align_53.pt")
    ap.add_argument("--ckpt", default=default_ckpt)
    ap.add_argument("--label", default="neutral", choices=list(LABELS))
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device)
    stoi, unk, eos = ck["stoi"], ck["unk"], ck["eos"]
    dim, n_words = ck["config"]["dim"], ck["config"]["n_words"]
    align = ck["config"].get("align", False)
    label_every = ck["config"].get("label_every", False)
    itos = {i: w for w, i in stoi.items()}; itos[unk] = "<unk>"; itos[eos] = "<eos>"

    model = PremiseBrain(n_words, dim, 0, eos, warm=None, align=align,
                         label_every=label_every).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    def decode(ids):
        out = []
        for t in ids.tolist():
            if t == eos or t == 0:
                break
            out.append(itos.get(t, "?"))
        return " ".join(out) if out else "(empty)"

    label = LABELS[args.label]
    show_trace = False
    print(f"loaded {args.ckpt}  (align={align})   device {device}")
    print(f"fixed label = {NAMES[label]}   |   type a hypothesis, or :label / :trace / :q\n")

    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line == ":q":
            break
        if line == ":trace":
            show_trace = not show_trace
            print(f"  [trace {'on' if show_trace else 'off'}]"); continue
        if line.startswith(":label"):
            parts = line.split()
            if len(parts) == 2 and parts[1] in LABELS:
                label = LABELS[parts[1]]; print(f"  [label = {NAMES[label]}]")
            else:
                print(f"  usage: :label {'/'.join(LABELS)}")
            continue

        hyp = encode_batch([line], stoi, unk, eos).to(device)
        y = torch.full((1,), label, dtype=torch.long, device=device)
        gen, att = model.generate(hyp, y, MAXLEN, unk=unk)
        words = decode(gen[0])
        print(f"ai  > {words}   [{NAMES[label]}]")
        if show_trace and att is not None:
            hw = line.split()
            gw = words.split()
            trace = " ".join(f"{w}<-{hw[att[0, j].item()] if att[0, j].item() < len(hw) else '<eos>'}"
                             for j, w in enumerate(gw))
            print(f"      thinking: {trace}")
        print(flush=True)


if __name__ == "__main__":
    main()
