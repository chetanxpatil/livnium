"""
prep_chat_pairs.py — turn the ChatGPT export into (your message -> reply) pairs
for the generator (chat_reply.py).

Unlike prep_chat_sentences.py (which harvested loose sentences for the typer),
the generator is SUPERVISED: it must learn to predict the reply from your
message. So here we keep the pairing.

Length handling: assistant replies are ~10x longer than the model can type
(MAX_WORDS=32), so the target is the reply's OPENING — cleaned sentences packed
greedily until the word budget is full. The message side is simply truncated.

Input : flattened_conversations.json  ->  [{"user": ..., "assistant": ...}, ...]
Output: data/chat_pairs.tsv           ->  message <TAB> reply, one pair per line.

Usage:
    python3 prep_chat_pairs.py
    python3 prep_chat_pairs.py --reply-mode first     # first sentence only
"""

import argparse
import json
import os
import re

DEFAULT_IN = ("/Users/chetanpatil/Desktop/test/lab/infected/projects/"
              "chat_crystal/build/unit_test_assets/assets/flattened_conversations.json")

_SPLIT = re.compile(r"[.!?]+|\n+")
_CLEAN = re.compile(r"[^a-z0-9' ]+")


def to_sentences(text):
    for chunk in _SPLIT.split(text or ""):
        s = chunk.lower()
        s = _CLEAN.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            yield s


def clean_message(text, max_words):
    """Whole message as one cleaned line, truncated to the word budget."""
    words = " ".join(to_sentences(text)).split()
    return " ".join(words[:max_words])


def reply_target(text, max_words, mode):
    """The reply's opening: first sentence, or sentences packed to the budget."""
    out = []
    for s in to_sentences(text):
        w = s.split()
        if not out:
            out = w[:max_words]
            if mode == "first":
                break
            continue
        if len(out) + len(w) > max_words:
            break
        out += w
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", default="data/chat_pairs.tsv")
    ap.add_argument("--max-words", type=int, default=32)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--reply-mode", choices=["pack", "first"], default="pack",
                    help="pack: fill the 32-word budget with whole sentences (default); "
                         "first: first sentence only")
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        pairs = json.load(f)

    seen, out = set(), []
    for p in pairs:
        msg = clean_message(p.get("user", ""), args.max_words)
        rep = reply_target(p.get("assistant", ""), args.max_words, args.reply_mode)
        if len(msg.split()) < args.min_words or len(rep.split()) < args.min_words:
            continue
        key = (msg, rep)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{msg}\t{rep}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"pairs read : {len(pairs)}")
    print(f"pairs kept : {len(out)}  (mode={args.reply_mode}, {args.min_words}-{args.max_words} words each side)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
