import pytest

from finrag.evaluate import first_number, numbers_match, parse_number


@pytest.mark.parametrize("text,value", [
    ("$1577.00", 1577.0),
    ("$11,588.00", 11588.0),
    ("0.66", 0.66),
    ("-0.02", -0.02),
    ("(305)", -305.0),
    ("10.3%", 10.3),
    ("0", 0.0),
    ("$2.1 billion", 2.1e9),
    ("$2,018mn", 2018e6),
    (" 42.69. ", 42.69),
])
def test_parse_number_accepts_pure_numbers(text, value):
    assert parse_number(text) == pytest.approx(value)


@pytest.mark.parametrize("text", [
    "$400,000,000 increase.",
    "Yes. It decreased.",
    "JnJ sold its inventory 2.7 times in FY2022.",
    "Data Center",
    "",
])
def test_parse_number_rejects_sentences(text):
    assert parse_number(text) is None


def test_first_number_skips_fiscal_year_tokens():
    assert first_number("FY2018 capex was $1,577 million") == pytest.approx(1577e6)
    assert first_number("no numbers here") is None


def test_tolerance():
    assert numbers_match(1577.4, 1577.0, rel_tol=0.001)
    assert not numbers_match(1600.0, 1577.0, rel_tol=0.001)
    assert numbers_match(0.66, 0.657, rel_tol=0.01)


def test_zero_gold_needs_abs_tol():
    assert not numbers_match(0.001, 0.0, rel_tol=0.01)
    assert numbers_match(0.001, 0.0, rel_tol=0.01, abs_tol=0.01)


def test_scale_mismatch_only_when_allowed():
    assert not numbers_match(1.577e9, 1577.0, rel_tol=0.001)
    assert numbers_match(1.577e9, 1577.0, rel_tol=0.001, allow_scale=True)


def test_missing_prediction_never_matches():
    assert not numbers_match(None, 1.0, rel_tol=1.0)
