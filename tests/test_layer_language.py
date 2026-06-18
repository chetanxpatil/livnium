from livnium_core.layer_language import evaluate, parse


def test_parses_ascii_and_unicode():
    assert parse("o^2 | *^1")["op"] == "|"
    assert parse("○2 ~ ●1")["op"] == "~"
    assert parse("*3|*3")["left"] == ("*", 3)


def test_rejects_garbage():
    for bad in ["", "hello", "o^2", "o^2 + *^1", "2 | 3"]:
        try:
            parse(bad)
            assert False, bad
        except ValueError:
            pass


def test_deterministic():
    assert evaluate("o^2 | *^9") == evaluate("o^2 | *^9")


def test_distinct_inputs_distinct_outputs():
    assert evaluate("o^2 | *^9") != evaluate("o^3 | *^9")


def test_base27_grounding():
    r = evaluate("o^2 | *^1")  # delta = (+1*1) - (-1*2) = 3
    assert r["delta"] == 3
    assert r["base27"] == "c"  # 3 -> 'c' in 0,a,b,c...
    assert r["dir"] == "in->out"


def test_layer_is_antisymmetric():
    # F(L|R).delta == -F(R|L).delta  -- a clean structural invariant
    a = evaluate("o^2 | *^1")["delta"]
    b = evaluate("*^1 | o^2")["delta"]
    assert a == -b


def test_relationship_alignment():
    assert evaluate("*^3 ~ *^1")["aligned"] is True  # same sign
    assert evaluate("o^2 ~ *^1")["aligned"] is False  # opposite sign


def test_neutral_when_equal():
    assert evaluate("*^3 | *^3")["delta"] == 0
    assert evaluate("*^3 | *^3")["dir"] == "neutral"
