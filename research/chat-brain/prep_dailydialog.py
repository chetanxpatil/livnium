"""
prep_dailydialog.py — DailyDialog -> the same context<TAB>reply tsv the
personal prep makes. General-fluency fuel for chat_reply.py.

DailyDialog: ~11k human-written everyday dialogues, two anonymous speakers in
strict turns. Every turn (either speaker) becomes a training reply; its
context is the previous turns tagged so the REPLIER is always <me> — exactly
the shape of the personal data. Tokenized by prep_chat_context.clean, the one
tokenizer everything shares.

The two-stage recipe (general fluency -> personal voice, wells shared):
    python3 prep_dailydialog.py                 # -> data/dd_context.tsv
                                                #    (downloads ~7MB of parquet)

    # stage A: general pretrain. --extra-vocab keeps YOUR words in the vocab
    # so stage B can still learn to say them.
    python3 chat_reply.py --data data/dd_context.tsv \
        --extra-vocab data/chat_context.tsv --ckpt model/chat_reply_general.pt

    # stage B: fine-tune on your chats, warm from stage A. gentler lr,
    # no scaffold restart.
    python3 chat_reply.py --data data/chat_context.tsv \
        --resume model/chat_reply_general.pt --pos-anneal 0 --lr 5e-4
"""

import argparse
import os

from prep_chat_context import build_context, reply_target, YOU, ME  # noqa: F401
from paths import data_path


# the original li2017dailydialog repo is a loading SCRIPT, which new HF
# datasets versions refuse to run. roskoN/dailydialog is a parquet mirror of
# the same corpus — fetched directly, no datasets library needed (pyarrow only).
MIRROR = "https://huggingface.co/api/datasets/roskoN/dailydialog/parquet/full/{split}/0.parquet"


def load_dialogs(cache_dir=data_path("dailydialog")):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("needs pyarrow:  pip3 install pyarrow")
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    for split in ("train", "validation", "test"):
        path = os.path.join(cache_dir, f"{split}.parquet")
        if not os.path.exists(path):
            print(f"downloading {split} ...", flush=True)
            urllib.request.urlretrieve(MIRROR.format(split=split), path)
        for utterances in pq.read_table(path).column("utterances").to_pylist():
            yield utterances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=data_path("dd_context.tsv"))
    ap.add_argument("--ctx-turns", type=int, default=6,
                    help="DailyDialog turns are short — reach further back")
    ap.add_argument("--ctx-words", type=int, default=256)
    ap.add_argument("--reply-words", type=int, default=32)
    ap.add_argument("--min-words", type=int, default=2)
    args = ap.parse_args()

    seen, out = set(), []
    n_dialogs = 0
    for dialog in load_dialogs():
        n_dialogs += 1
        turns = [t.strip() for t in dialog if t and t.strip()]
        for i in range(1, len(turns)):
            # tag history so the replier at i is <me>: alternate backwards
            hist = [("user" if (i - j) % 2 == 1 else "assistant", turns[j])
                    for j in range(max(0, i - args.ctx_turns), i)]
            ctx = build_context(hist, args.ctx_words)
            rep = reply_target(turns[i], args.reply_words)
            if len([w for w in ctx.split() if w not in (YOU, ME)]) < args.min_words:
                continue
            if len(rep.split()) < args.min_words:
                continue
            key = (ctx, rep)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{ctx}\t{rep}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"dialogues     : {n_dialogs}")
    print(f"examples kept : {len(out)}  (ctx {args.ctx_turns} turns / "
          f"{args.ctx_words} words -> reply {args.reply_words} words)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
