import random

from livnium_core import base27_to_binary, base27_to_int, binary_to_base27, int_to_base27
from livnium_core.base27 import add


def test_known_values():
    assert int_to_base27(0) == "0"
    assert int_to_base27(1) == "a"
    assert int_to_base27(26) == "z"
    assert int_to_base27(27) == "a0"
    assert int_to_base27(28) == "aa"
    assert int_to_base27(89) == "ch"


def test_int_roundtrip():
    for n in [0, 1, 26, 27, 89, 729, 123456789, 2**32, 10**18]:
        assert base27_to_int(int_to_base27(n)) == n


def test_binary_roundtrip():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(0, 10**15)
        s = int_to_base27(n)
        assert binary_to_base27(base27_to_binary(s)) == s


def test_carry():
    # 26 + 1 = 27 must roll a digit over: 'z' + 'a' -> 'a0'
    assert add("z", "a") == "a0"
    assert add("zz", "a") == "a00"
    assert add("a0", "a0") == "b0"  # 27 + 27 = 54
    assert add("zzz", "a") == "a000"  # 19682 + 1 = 19683


def test_carry_matches_integer_addition():
    rng = random.Random(1)
    for _ in range(500):
        a, b = rng.randint(0, 10**9), rng.randint(0, 10**9)
        assert base27_to_int(add(int_to_base27(a), int_to_base27(b))) == a + b
