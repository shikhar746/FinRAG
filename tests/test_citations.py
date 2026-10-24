from finrag.evaluate import citation_scores
from finrag.generate import build_prompt, is_refusal, parse_citations, parse_final_answer


def test_single_and_multiple_citations():
    assert parse_citations("Capex was $1,577m [p. 59].") == {59}
    assert parse_citations("See [p. 12, 14] and [p.30]") == {12, 14, 30}


def test_ranges_and_words():
    assert parse_citations("[pp. 5-7]") == {5, 6, 7}
    assert parse_citations("[page 5 and 9]") == {5, 9}
    assert parse_citations("[Pages 3–4]") == {3, 4}


def test_ignores_non_citations():
    assert parse_citations("In 2018 [note 3] revenue grew") == set()
    assert parse_citations("[p. 1-999]") == set()  # implausible range is dropped


def test_final_answer_takes_last_occurrence():
    text = "Final answer: 10\nWait, recomputing.\nFinal answer: 12.5"
    assert parse_final_answer(text) == "12.5"
    assert parse_final_answer("no marker") is None


def test_refusal_detection():
    assert is_refusal("Final answer: Cannot answer from the provided pages.")
    assert is_refusal("Final answer: I don't know.")
    assert not is_refusal("Final answer: 1577")


def test_citation_scores():
    assert citation_scores({59}, {59}) == {"cited_gold": 1.0, "citation_precision": 1.0}
    assert citation_scores({59, 60}, {59}) == {"cited_gold": 1.0, "citation_precision": 0.5}
    assert citation_scores(set(), {59}) == {"cited_gold": 0.0, "citation_precision": 0.0}


def test_prompt_uses_page_ids():
    p = build_prompt("Q?", "3M_2018_10K", [(59, "cash flow")])
    assert '<page id="59">' in p and "Q?" in p
    assert "<page" not in build_prompt("Q?", "3M_2018_10K", None)
