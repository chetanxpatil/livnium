"""
prep_chat_sentences.py — turn a ChatGPT export into plain sentences the typer
can learn to WRITE BACK.

The typer (chat_typer.py) is unsupervised: it just learns to type sentences.
So we don't need the user/assistant pairing here — we only need clean sentences,
each short enough to fit the typer's MAX_WORDS window.

Input : conversations.json (RAW export) — walked by the canonical path via
        prep_chat_context.canonical_turns. Single source of truth; no flatten.
Output: data/chat_sentences.txt  ->  one lowercase sentence per line, deduped.

Usage:
    python3 prep_chat_sentences.py
    python3 prep_chat_sentences.py --in /path/to/flattened_conversations.json \
                                   --out data/chat_sentences.txt --max-words 32
"""

import argparse
import json
import os
import re

from paths import RAW_EXPORT, data_path

DEFAULT_IN = str(RAW_EXPORT)

# ONE cleaning source: prep_chat_context owns tokenization (punctuation = tokens)
from prep_chat_context import to_sentences  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", default=data_path("chat_sentences.txt"))
    ap.add_argument("--max-words", type=int, default=32)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--include", choices=["both", "user", "assistant"], default="both",
                    help="which side of each pair to harvest sentences from")
    args = ap.parse_args()

    from prep_chat_context import canonical_turns
    with open(args.inp, encoding="utf-8") as f:
        convs = json.load(f)

    roles = {"both": ("user", "assistant"),
             "user": ("user",), "assistant": ("assistant",)}[args.include]

    seen, out = set(), []
    n_pairs = 0
    for conv in convs:
        for role, text in canonical_turns(conv):
            n_pairs += 1
            if role not in roles:
                continue
            for s in to_sentences(text):
                n = len(s.split())
                if n < args.min_words or n > args.max_words:
                    continue
                if s in seen:
                    continue
                seen.add(s)
                out.append(s)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    # quick vocab report
    from collections import Counter
    cnt = Counter(t for s in out for t in s.split())
    ge5 = sum(1 for _, n in cnt.items() if n >= 5)
    print(f"turns on canonical paths : {n_pairs}")
    print(f"sentences kept        : {len(out)}  ({args.include}, {args.min_words}-{args.max_words} words)")
    print(f"unique tokens         : {len(cnt)}")
    print(f"tokens appearing >=5x : {ge5}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
