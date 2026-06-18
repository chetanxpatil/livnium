"""
base27.py — the Livnium base-27 positional codec.

Alphabet: '0' (value 0, the Om/core symbol) then 'a'..'z' (values 1..26).
This is a standard positional numeral system in base 27, so it inherits
exact carry behaviour and is losslessly convertible to/from integers and
binary.

    >>> int_to_base27(27)
    'a0'
    >>> base27_to_int('a0')
    27
    >>> base27_to_int('z') + 1 == base27_to_int('a0')   # carry: 26 + 1 = 27
    True

Verified by tests/test_base27.py.
"""

from __future__ import annotations

ALPHABET = "0abcdefghijklmnopqrstuvwxyz"  # index == digit value
_CHAR_TO_VAL = {c: v for v, c in enumerate(ALPHABET)}
_VAL_TO_CHAR = {v: c for v, c in enumerate(ALPHABET)}
BASE = 27


def base27_to_int(s: str) -> int:
    """Decode a base-27 string to a non-negative integer."""
    if not s:
        raise ValueError("empty string")
    n = 0
    for ch in s:
        if ch not in _CHAR_TO_VAL:
            raise ValueError(f"invalid base-27 character: {ch!r}")
        n = n * BASE + _CHAR_TO_VAL[ch]
    return n


def int_to_base27(n: int) -> str:
    """Encode a non-negative integer to a base-27 string."""
    if n < 0:
        raise ValueError("only non-negative integers are supported")
    if n == 0:
        return "0"
    digits = []
    while n > 0:
        digits.append(_VAL_TO_CHAR[n % BASE])
        n //= BASE
    return "".join(reversed(digits))


def base27_to_binary(s: str) -> str:
    """Base-27 string -> binary string (no '0b' prefix)."""
    return bin(base27_to_int(s))[2:]


def binary_to_base27(b: str) -> str:
    """Binary string -> base-27 string."""
    return int_to_base27(int(b, 2))


def add(x: str, y: str) -> str:
    """Add two base-27 numerals (carry is exact, by positional arithmetic)."""
    return int_to_base27(base27_to_int(x) + base27_to_int(y))
