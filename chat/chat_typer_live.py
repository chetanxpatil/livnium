"""
chat_typer_live.py — type sentences back, and MINT unseen words on the fly.

Loads a trained model/chat_typer.pt (the standalone word typer) and lets you
type any sentence. If a word isn't in the trained vocab, instead of collapsing
it to <unk>, we build a well for it FROM ITS SPELLING (char_fingerprint) and
append it to the well table right then — the char->word bridge, fired ONLY for
new words. The trained wells are never touched, nothing is retrained.

    word in vocab      -> use its trained well
    word never seen    -> mint well = char_fingerprint(word), append, use it

So the typer's vocabulary GROWS as you talk to it, with each new word getting a
stable, distinct, deterministic position from its letters.

Usage:
    python3 chat_typer_live.py                       # interactive
    python3 chat_typer_live.py --ckpt model/chat_typer.pt
    echo "your sentence here" | python3 chat_typer_live.py
    python3 chat_typer_live.py --save model/chat_typer_grown.pt   # persist minted words

In the prompt:
    <type any sentence>  -> it types the sentence back, marks *newly minted words
    :words               -> how many wells exist now (trained + minted)
    :q                   -> quit
"""

import argparse
import sys

import torch
import torch.nn.functional as F

from char_fingerprint import letter_anchors, char_fingerprint

PAD = 0


class LiveTyper:
    def __init__(self, ckpt_path, device):
        ck = torch.load(ckpt_path, map_location=device)
        self.device = device
        self.dim = ck["config"]["dim"]
        self.anchors = ck["word_anchors"].to(device)      # (n_words, dim), grows
        self.start = ck["start"].to(device)
        self.strength = float(ck["strength"])
        self.temp = float(ck["temp"])
        self.stoi = dict(ck["stoi"])
        self.itos = dict(ck["itos"])
        self.unk = ck["unk"]
        self.eos = ck["eos"]
        self.trained_n = self.anchors.size(0)
        self.minted = []                                  # words we created live
        self.A = letter_anchors(self.dim, device=device)  # spelling geometry

    def ensure_word(self, w):
        """Return the id for w, minting a spelling-well the first time we see it."""
        if w in self.stoi:
            return self.stoi[w], False
        new_id = self.anchors.size(0)
        fp = char_fingerprint(w, self.A, self.dim).unsqueeze(0)
        self.anchors = torch.cat([self.anchors, fp], dim=0)
        self.stoi[w] = new_id
        self.itos[new_id] = w
        self.minted.append(w)
        return new_id, True

    def encode(self, ids):
        """Same vector-collapse trajectory as training, one sentence."""
        anchors = F.normalize(self.anchors, dim=-1)
        h = self.start.clone()
        states = []
        s = self.strength
        for i in ids:
            target = anchors[i]
            align = (F.normalize(h, dim=0) * target).sum()
            away = F.normalize(h - target, dim=0)
            h = h - s * (1.0 - align) * away
            n = h.norm()
            if n > 10.0:
                h = h * (10.0 / (n + 1e-8))
            states.append(h.clone())
        return torch.stack(states, dim=0)

    def type_back(self, sentence):
        toks = sentence.lower().split()
        new_flags = {}
        ids = []
        for w in toks:
            wid, minted = self.ensure_word(w)
            ids.append(wid)
            new_flags[w] = minted
        ids.append(self.eos)

        states = self.encode(ids)
        anchors = F.normalize(self.anchors, dim=-1)
        logits = (F.normalize(states, dim=-1) @ anchors.t()) / self.temp
        logits[:, PAD] = float("-inf")
        logits[:, self.unk] = float("-inf")
        pred = logits.argmax(-1)

        out = []
        for t in pred.tolist():
            if t == self.eos or t == PAD:
                break
            out.append(self.itos.get(t, "?"))
        return " ".join(out), new_flags

    def save(self, path):
        torch.save({"word_anchors": self.anchors.detach().cpu(),
                    "start": self.start.detach().cpu(),
                    "strength": self.strength, "temp": self.temp,
                    "stoi": self.stoi, "itos": self.itos,
                    "unk": self.unk, "eos": self.eos,
                    "config": {"dim": self.dim, "n_words": self.anchors.size(0)}}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/chat_typer.pt")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    lt = LiveTyper(args.ckpt, device)
    print(f"loaded {args.ckpt}   trained wells {lt.trained_n}   device {device}")
    print("type a sentence; unseen words get *minted from spelling. :words / :q\n")

    def handle(line):
        got, flags = lt.type_back(line)
        shown = " ".join((f"*{w}" if flags.get(w) else w) for w in line.lower().split())
        print(f"in   > {shown}")
        print(f"typed> {got}")
        new = [w for w, m in flags.items() if m]
        if new:
            print(f"      minted {len(new)} new well(s): {', '.join(new)}")
        print(flush=True)

    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line:
                handle(line)
    else:
        while True:
            try:
                line = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not line:
                continue
            if line == ":q":
                break
            if line == ":words":
                print(f"  {lt.anchors.size(0)} wells "
                      f"({lt.trained_n} trained + {len(lt.minted)} minted)\n"); continue
            handle(line)

    if args.save:
        lt.save(args.save)
        print(f"saved grown vocab ({lt.anchors.size(0)} wells) -> {args.save}")


if __name__ == "__main__":
    main()
