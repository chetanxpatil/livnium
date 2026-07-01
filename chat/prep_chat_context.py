"""
prep_chat_context.py — session-aware prep from the RAW ChatGPT export.

Replaces the flatten. What the flatten threw away, this keeps:
  * SESSIONS: examples never cross a conversation boundary.
  * ORDER: each conversation's canonical path is walked from current_node up
    the parent chain (branched/edited turns off the final path are skipped —
    the tree is resolved the way ChatGPT itself shows it).
  * CONTEXT: the model no longer predicts a reply from one amnesiac message.
    Each example is (last K turns -> the next reply), turns tagged <you> / <me>.
    Context is TAIL-truncated: when it exceeds the word budget the OLDEST words
    drop first, so the most recent turn always survives.

Input : conversations.json  (the raw 273MB export)
Output: data/chat_context.tsv  ->  context <TAB> reply, one example per line.
        e.g.  <you> how do i fix it <me> update torch <you> still broken \t try the fallback env var

Usage:
    python3 prep_chat_context.py
    python3 prep_chat_context.py --ctx-turns 3 --ctx-words 48
"""

import argparse
import json
import os
import re

DEFAULT_IN = ("/Users/chetanpatil/Desktop/test/lab/infected/projects/"
              "chat_crystal/build/unit_test_assets/assets/conversations.json")

YOU, ME = "<you>", "<me>"          # speaker wells — trainable tokens, not text
_CLEAN = re.compile(r"[^a-z0-9' ]+")
_SPLIT = re.compile(r"[.!?]+|\n+")


def clean(text):
    s = _CLEAN.sub(" ", (text or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def to_sentences(text):
    for chunk in _SPLIT.split(text or ""):
        s = clean(chunk)
        if s:
            yield s


def node_text(node):
    """Extract plain text from a mapping node, or None."""
    msg = node.get("message")
    if not msg:
        return None
    role = (msg.get("author") or {}).get("role")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content") or {}
    if content.get("content_type") not in ("text", "multimodal_text"):
        return None                      # drops code/tool payloads, keeps image+text asks
    parts = [p for p in (content.get("parts") or []) if isinstance(p, str) and p.strip()]
    if not parts:
        return None
    return role, "\n".join(parts)


def canonical_turns(conv):
    """Walk current_node -> root via parents: the path ChatGPT itself displays.
    Branches/edits off this path are skipped. Returns [(role, text), ...] in order."""
    mapping = conv.get("mapping") or {}
    node_id = conv.get("current_node")
    path = []
    seen = set()
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        node = mapping[node_id]
        rt = node_text(node)
        if rt:
            path.append(rt)
        node_id = node.get("parent")
    return path[::-1]


def reply_target(text, max_words):
    """The reply's opening: whole sentences packed into the word budget."""
    out = []
    for s in to_sentences(text):
        w = s.split()
        if not out:
            out = w[:max_words]
            continue
        if len(out) + len(w) > max_words:
            break
        out += w
    return " ".join(out)


def build_context(turns, ctx_words):
    """Tag each turn with its speaker well, then keep the LAST ctx_words tokens."""
    toks = []
    for role, text in turns:
        t = clean(text)
        if not t:
            continue
        toks.append(YOU if role == "user" else ME)
        toks += t.split()
    return " ".join(toks[-ctx_words:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEFAULT_IN)
    ap.add_argument("--out", default="data/chat_context.tsv")
    ap.add_argument("--ctx-turns", type=int, default=3,
                    help="how many turns of history feed each prediction")
    ap.add_argument("--ctx-words", type=int, default=48,
                    help="context word budget (oldest words drop first)")
    ap.add_argument("--reply-words", type=int, default=32)
    ap.add_argument("--min-words", type=int, default=2)
    args = ap.parse_args()

    print("reading raw export (273MB takes a moment) ...", flush=True)
    with open(args.inp, encoding="utf-8") as f:
        convs = json.load(f)

    seen, out = set(), []
    n_turns = 0
    for conv in convs:
        turns = canonical_turns(conv)
        n_turns += len(turns)
        for i, (role, text) in enumerate(turns):
            if role != "assistant" or i == 0:
                continue
            ctx = build_context(turns[max(0, i - args.ctx_turns):i], args.ctx_words)
            rep = reply_target(text, args.reply_words)
            # min-words: count real words, not the <you>/<me> tags
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

    print(f"conversations : {len(convs)}")
    print(f"turns on canonical paths : {n_turns}")
    print(f"examples kept : {len(out)}  (ctx {args.ctx_turns} turns / {args.ctx_words} words -> reply {args.reply_words} words)")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
