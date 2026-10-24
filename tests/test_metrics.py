import pytest

from finrag.evaluate import hit_at_k, paired_bootstrap, recall_at_k, reciprocal_rank

RANKED = ["a", "b", "c", "d", "e"]


def test_recall_counts_fraction_of_gold():
    assert recall_at_k(RANKED, {"b", "z"}, 1) == 0.0
    assert recall_at_k(RANKED, {"b", "z"}, 2) == 0.5
    assert recall_at_k(RANKED, {"a", "c"}, 5) == 1.0


def test_recall_needs_gold():
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 5)


def test_hit_is_any_gold():
    assert hit_at_k(RANKED, {"c", "z"}, 2) == 0.0
    assert hit_at_k(RANKED, {"c", "z"}, 3) == 1.0


def test_reciprocal_rank_uses_first_gold():
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"d", "b"}) == 0.5
    assert reciprocal_rank(RANKED, {"z"}) == 0.0


def test_page_keys_are_tuples():
    ranked = [("DOC", 3), ("DOC", 7)]
    assert reciprocal_rank(ranked, {("DOC", 7)}) == 0.5
    assert reciprocal_rank(ranked, {("OTHER", 7)}) == 0.0


def test_bootstrap_identical_systems_has_zero_diff():
    r = paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0])
    assert r["mean_diff"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert not r["excludes_zero"]


def test_bootstrap_clear_win_excludes_zero():
    r = paired_bootstrap([0] * 50, [1] * 45 + [0] * 5, seed=1)
    assert r["mean_diff"] == pytest.approx(0.9)
    assert r["ci_low"] > 0 and r["excludes_zero"]


def test_bootstrap_is_seeded():
    a, b = [0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 1, 1]
    assert paired_bootstrap(a, b, seed=3) == paired_bootstrap(a, b, seed=3)


def test_bootstrap_rejects_unpaired():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])
