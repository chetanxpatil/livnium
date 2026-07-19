#!/usr/bin/env python3
"""
qwen_probe.py — is qwen3:0.6b a useful NLI teacher?

Runs locally on YOUR machine (where ollama is reachable). It sends a few
labeled premise/hypothesis pairs to qwen3:0.6b, parses its E/N/C answer, and
reports agreement with the gold label. If agreement is well above chance (33%)
and above the SNLI hypothesis-only baseline (~62%), soft-label distillation is
worth pursuing; if not, prefer embedding distillation only.

No dependencies — just Python 3 + a running ollama.

    ollama serve            # (usually already running)
    python3 qwen_probe.py

It writes qwen_probe_out.json so you can share the raw results back.
"""
from __future__ import annotations

import json
import re
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:0.6b"

# (premise, hypothesis, gold)
PAIRS = [
    ("A man is playing a guitar on stage.", "A person is performing music.", "entailment"),
    ("A dog is running in the park.", "The animal is sleeping indoors.", "contradiction"),
    ("A woman is buying vegetables at a market.", "She is preparing dinner tonight.", "neutral"),
    ("Two children are building a sandcastle on the beach.", "Kids are at the beach.", "entailment"),
    ("A chef is chopping onions in a kitchen.", "The chef is on vacation at the sea.", "contradiction"),
    ("A cyclist rides down a busy street.", "The cyclist is training for a race.", "neutral"),
    ("A group of people wait at a bus stop.", "People are outdoors.", "entailment"),
    ("The cat is sleeping on the couch.", "The cat is chasing a mouse outside.", "contradiction"),
]

SYSTEM = (
    "You are an NLI classifier. Given a premise and a hypothesis, decide the "
    "relationship. Answer with EXACTLY ONE word: entailment, neutral, or "
    "contradiction. No explanation."
)

LABELS = ("entailment", "neutral", "contradiction")


def ask(premise: str, hypothesis: str) -> str:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"},
        ],
        "stream": False,
        "think": False,            # disable Qwen3 reasoning trace for a clean label
        "options": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    text = resp.get("message", {}).get("content", "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # strip if any
    return text.strip().lower()


def parse_label(text: str) -> str:
    text = text.lower()
    for lab in LABELS:
        if lab in text:
            return lab
    # short forms
    if re.search(r"\bentail", text): return "entailment"
    if re.search(r"\bcontradict", text): return "contradiction"
    if re.search(r"\bneutral", text): return "neutral"
    return "?"


def main() -> None:
    rows, correct = [], 0
    print(f"Probing {MODEL} on {len(PAIRS)} NLI pairs...\n")
    for premise, hyp, gold in PAIRS:
        try:
            raw = ask(premise, hyp)
        except Exception as e:
            print(f"ERROR talking to ollama: {e}")
            print("Is ollama running?  Try:  ollama serve   and   ollama run qwen3:0.6b")
            return
        pred = parse_label(raw)
        ok = pred == gold
        correct += ok
        mark = "✓" if ok else "✗"
        print(f"  {mark} gold={gold:<13} pred={pred:<13} raw={raw[:40]!r}")
        rows.append({"premise": premise, "hypothesis": hyp,
                     "gold": gold, "pred": pred, "raw": raw})

    acc = correct / len(PAIRS)
    print(f"\nagreement with gold: {correct}/{len(PAIRS)} = {acc:.0%}")
    print("  chance = 33% · SNLI hypothesis-only baseline ≈ 62% · livnium v1 ≈ 69%")
    if acc >= 0.62:
        print("  → qwen3:0.6b looks like a usable teacher; soft-label distillation worth trying.")
    else:
        print("  → weak as a label teacher; prefer EMBEDDING distillation instead.")

    with open("qwen_probe_out.json", "w") as f:
        json.dump({"model": MODEL, "accuracy": acc, "rows": rows}, f, indent=2)
    print("\nwrote qwen_probe_out.json — share it back and we'll decide the approach.")


if __name__ == "__main__":
    main()
