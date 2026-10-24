import pytest

from finrag.generate import AnthropicBackend, BudgetExceeded, GenerationCache, generate


class FakeBackend:
    name, model, params = "fake", "fake-1", {"temperature": 0.0}

    def __init__(self):
        self.calls = 0

    def complete(self, system, prompt):
        self.calls += 1
        return {"text": "Final answer: 1 [p. 3]", "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.01}


def test_cache_prevents_second_call_and_survives_reload(tmp_path):
    path = tmp_path / "gen.jsonl"
    backend = FakeBackend()
    first = generate(backend, GenerationCache(path), "prompt", {"financebench_id": "q1"})
    again = generate(backend, GenerationCache(path), "prompt", {"financebench_id": "q1"})
    assert backend.calls == 1 and again["text"] == first["text"]
    generate(backend, GenerationCache(path), "other prompt", {"financebench_id": "q2"})
    assert backend.calls == 2
    assert GenerationCache(path).spent_usd("fake") == pytest.approx(0.02)


def test_budget_cap_refuses_before_any_request():
    b = AnthropicBackend("m", "key", max_usd=0.01, usd_per_mtok_in=3.0, usd_per_mtok_out=15.0, spent_usd=0.0)
    with pytest.raises(BudgetExceeded):  # 512 output tokens at $15/Mtok already exceeds $0.01
        b.complete("system", "x" * 3000)


def test_budget_requires_prices():
    with pytest.raises(ValueError):
        AnthropicBackend("m", "key", max_usd=5.0, usd_per_mtok_in=None, usd_per_mtok_out=None)
