"""freeze_corpus.py — stage 1: freeze the EXACT corpus every model trains on.

Reads the same source the noun model used (wiki .xml.bz2 / txt / folder),
applies the SAME clean() from prep_chat_context, and writes one cleaned line
per line to work/corpus.txt. Records SHA-256 + line/token counts in
work/corpus_manifest.json. Every downstream stage reads ONLY corpus.txt.

Usage (mirrors the published run):
    python3 freeze_corpus.py \
        --data ~/.../enwiki-latest-pages-articles-multistream.xml.bz2 \
        --max-lines 5000000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "research", "embeddings", "noun-collapse"))
from noun_embed import iter_lines            # noqa: E402
from prep_chat_context import clean          # noqa: E402

from common import WORK, caffeinate, save_json, sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--sample-parts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (only used with --sample-parts)")
    args = ap.parse_args()
    caffeinate()

    out = os.path.join(WORK, "corpus.txt")
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(out):
        sys.exit(f"{out} already exists — delete it to re-freeze (the hash must not drift)")

    lines = tokens = 0
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in iter_lines(args.data, args.max_lines, args.sample_parts, args.seed):
            c = clean(line)
            if not c:
                continue
            f.write(c + "\n")
            lines += 1
            tokens += c.count(" ") + 1
            if lines % 500000 == 0:
                print(f"  {lines:,} lines, {tokens:,} tokens", flush=True)
    os.replace(tmp, out)

    sha = sha256_file(out)
    save_json(os.path.join(WORK, "corpus_manifest.json"), {
        "sha256": sha, "lines": lines, "tokens": tokens,
        "source": os.path.basename(args.data),
        "max_lines": args.max_lines, "sample_parts": args.sample_parts,
        "sample_seed": args.seed, "cleaner": "prep_chat_context.clean",
    })
    print(f"frozen: {lines:,} lines, {tokens:,} tokens\nsha256: {sha}")


if __name__ == "__main__":
    main()
