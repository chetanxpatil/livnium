"""
layer_language.py — the Livnium Layer Language, with semantics grounded in base-27.

A small, fully-defined symbolic algebra (the "static perfect universe"): it parses
Layer-Language expressions and COMPUTES a deterministic result for each, using the
base-27 codec for the magnitude encoding.

Notation
--------
  shapes : ``o`` / ``○`` = hollow (inner, sign -1)
           ``*`` / ``●`` = filled (outer, sign +1)
  depth  : an integer written after the shape, optionally with ``^``  (e.g. o^2, *9)
  ``|``  : the LAYER operator -> output is the function F(left -> right)
  ``~``  : the RELATIONSHIP operator -> pure relationship, no layer function

Semantics (F, grounded in base-27)
----------------------------------
  signed_depth(shape, n) = sign(shape) * n
  F(L | R):
      delta   = signed_depth(R) - signed_depth(L)      # directed transition
      base27  = int_to_base27(|delta|)                 # magnitude in the codec
      dSW     = SW(depthR) - SW(depthL)                # energy change, SW = 9*(d mod 4)
      dir     = 'in->out' if delta>0 else 'out->in' if delta<0 else 'neutral'
  L ~ R:
      aligned = (sign(L) == sign(R))

Honest scope
------------
This computes STRUCTURE (relationships between the symbols' own depths/signs),
deterministically and self-consistently — a perfect *reference frame*. It does
NOT encode meaning: F(cat->mom) would transform the *codes*, not the fact that a
cat loves a mom. Meaning would enter only as data deforming this base. See
LIMITS.md and REARRANGEMENT.md.
"""
from __future__ import annotations
import re
from typing import Dict
from .base27 import int_to_base27

_SIGN = {"o": -1, "○": -1, "*": +1, "●": +1}
_TOKEN = re.compile(r"\s*([o○*●])\s*\^?\s*(\d+)\s*([|~])\s*([o○*●])\s*\^?\s*(\d+)\s*$")


def _sw(depth: int) -> int:
    """Symbolic-weight echo of the exposure classes (0, 9, 18, 27)."""
    return 9 * (depth % 4)


def parse(expr: str) -> Dict:
    m = _TOKEN.match(expr)
    if not m:
        raise ValueError(f"not a valid Layer-Language expression: {expr!r}")
    s1, d1, op, s2, d2 = m.group(1), int(m.group(2)), m.group(3), m.group(4), int(m.group(5))
    return {"left": (s1, d1), "op": op, "right": (s2, d2)}


def evaluate(expr: str) -> Dict:
    """Compute the defined result of a Layer-Language expression."""
    p = parse(expr)
    (s1, n), op, (s2, k) = p["left"], p["op"], p["right"]
    if op == "|":
        delta = _SIGN[s2] * k - _SIGN[s1] * n
        return {
            "op": "layer",
            "delta": delta,
            "base27": int_to_base27(abs(delta)),
            "dSW": _sw(k) - _sw(n),
            "dir": "in->out" if delta > 0 else "out->in" if delta < 0 else "neutral",
        }
    # relationship
    return {
        "op": "relation",
        "aligned": _SIGN[s1] == _SIGN[s2],
        "pair": (f"{s1}{n}", f"{s2}{k}"),
    }
