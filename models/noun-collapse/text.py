"""Tokenization used by the noun-collapse training and benchmark pipeline.

This lives with the model so noun training does not depend on the personal
chat-data experiment. Punctuation is kept as standalone tokens and common
space-split contractions are repaired before tokenization.
"""

import re

_KEEP = set(".,!?;:")
_UNI = str.maketrans(
    {"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'", "…": "."}
)
_CONTRACT_APOS = re.compile(r"\s*'\s*(t|s|re|ve|ll|d|m)\b")
_CONTRACT_NT = re.compile(r"(\w)\s+n't\b")


def clean(text):
    """Lowercase text, repair contractions, and peel edge punctuation."""
    value = (text or "").lower().translate(_UNI)
    value = _CONTRACT_NT.sub(r"\1n't", value)
    value = _CONTRACT_APOS.sub(r"'\1", value)
    out = []
    for token in value.split():
        head, tail = [], []
        while token and not token[0].isalnum():
            head.append(token[0])
            token = token[1:]
        while token and not token[-1].isalnum():
            tail.append(token[-1])
            token = token[:-1]
        for mark in head:
            if mark in _KEEP and not (out and out[-1] == mark):
                out.append(mark)
        if token:
            out.append(token)
        for mark in reversed(tail):
            if mark in _KEEP and not (out and out[-1] == mark):
                out.append(mark)
    return " ".join(out)
