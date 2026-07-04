"""
vocab_overlap.py — why did only N/M wells get semantic-init?

Loads the noun model and a chat/typer checkpoint and answers the honest
question: are the words that FAILED to match rare/genuinely-OOV (expected),
or common words lost to a tokenization mismatch (a real, fixable bug)?

The tell: if high-frequency function/content words ("the", "you", "help",
"office") are in the noun model but NOT matching, tokenization differs.
If only rare/personal/code tokens miss, the 31% is legitimate.

Usage:
    python3 vocab_overlap.py                      # noun vs chat_reply_general
    python3 vocab_overlap.py --chat model/chat_typer.pt
"""

import argparse

import torch


def load_stoi(path):
    ck = torch.load(path, map_location="cpu")
    if "stoi" in ck:
        return dict(ck["stoi"])
    return dict(ck["config"].get("stoi", {}))          # fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noun", default="model/noun_collapse_pure.pt")
    ap.add_argument("--chat", default="model/chat_reply_general.pt")
    args = ap.parse_args()

    noun = set(load_stoi(args.noun))
    chat_stoi = load_stoi(args.chat)
    chat = set(chat_stoi)
    matched = chat & noun
    missed = chat - noun

    print(f"noun vocab : {len(noun):,}")
    print(f"chat vocab : {len(chat):,}")
    print(f"matched    : {len(matched):,}  ({100*len(matched)/len(chat):.1f}% of chat)")
    print(f"missed     : {len(missed):,}\n")

    # the diagnostic: are COMMON words missing?  chat ids are assigned by
    # frequency (lower id = more frequent) in most of these builders, so the
    # lowest-id missed words are the most-common ones that failed to match.
    by_id = sorted(missed, key=lambda w: chat_stoi[w])
    print("--- 40 LOWEST-ID (≈ most frequent) chat words that did NOT match ---")
    print("   (if these are ordinary words, it's a tokenization mismatch)")
    print("   " + "  ".join(by_id[:40]))

    # spot-check words from the dev examples that SHOULD be common
    probe = ["the", "you", "i", "help", "office", "sorry", "sure", "well",
             "job", "think", "don't", "i'm", "can't", "what", "?", ".", ",",
             "<you>", "<me>"]
    print("\n--- spot check (in noun? in chat?) ---")
    for w in probe:
        print(f"   {w:8s}  noun={'Y' if w in noun else '-'}  "
              f"chat={'Y' if w in chat else '-'}  "
              f"{'MATCH' if w in matched else ('MISS' if w in chat else '')}")


if __name__ == "__main__":
    main()
