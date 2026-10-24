"""Answer generation: prompts, LLM backends, a JSONL cache, and output parsing.

Output format we ask every model for:
    ...reasoning, with citations like [p. 59] or [p. 59, 61]...
    Final answer: <answer>

Page labels shown to the model are the same 0-based page indices used for
gold evidence, so a citation can be checked directly against the gold pages.
"""
import hashlib
import json
import re
from pathlib import Path

import requests

SYSTEM = "You are a careful financial analyst answering questions about SEC filings."

RAG_TEMPLATE = """Answer the question using only the filing pages below.
Each page starts with a tag like <page id="12">.
Cite every page you use as [p. ID], for example [p. 12] or [p. 12, 14], using the page id from the tag.
If the pages do not contain the answer, reply exactly: "Final answer: Cannot answer from the provided pages."
Finish with one line in the form "Final answer: <answer>". For a numeric answer, give only the number,
in the unit the question asks for.

Filing: {doc_name}

{pages}

Question: {question}"""

CLOSED_BOOK_TEMPLATE = """Answer the question from your own knowledge. No documents are provided.
If you do not know, reply exactly: "Final answer: I don't know."
Finish with one line in the form "Final answer: <answer>". For a numeric answer, give only the number,
in the unit the question asks for.

Filing: {doc_name}

Question: {question}"""


def build_prompt(question: str, doc_name: str, pages: list[tuple[int, str]] | None) -> str:
    """pages=None -> closed-book prompt; otherwise [(page_id, text), ...]."""
    if pages is None:
        return CLOSED_BOOK_TEMPLATE.format(doc_name=doc_name, question=question)
    blocks = "\n\n".join(f'<page id="{p}">\n{text.strip()}\n</page>' for p, text in pages)
    return RAG_TEMPLATE.format(doc_name=doc_name, pages=blocks, question=question)


# ---------------------------------------------------------------- parsing

# "[p." / "[pp." / "[page" / "[pages" followed by anything up to "]".
_CITE = re.compile(r"\[\s*(?:pages?|pp?)\.?\s*([^\]]+)\]", re.IGNORECASE)
_FINAL = re.compile(r"final answer\s*[:\-]\s*(.+)", re.IGNORECASE)
_REFUSAL = re.compile(r"cannot answer from the provided pages|i don't know|i do not know", re.IGNORECASE)


def parse_citations(text: str) -> set[int]:
    """All page ids cited as [p. 5], [pp. 5-7], [p. 5, 9], [page 5 and 6]."""
    pages = set()
    for group in _CITE.findall(text):
        for part in re.split(r"\s*(?:,|and|&)\s*", group.strip()):
            if not part:
                continue
            bounds = re.split(r"\s*[\-–]\s*", part)
            if len(bounds) == 2 and all(b.isdigit() for b in bounds):
                lo, hi = int(bounds[0]), int(bounds[1])
                if lo <= hi and hi - lo <= 20:  # guard against nonsense like [p. 1-999]
                    pages.update(range(lo, hi + 1))
            elif part.isdigit():
                pages.add(int(part))
    return pages


def parse_final_answer(text: str) -> str | None:
    """Text after the LAST 'Final answer:' (models sometimes restate it), or None."""
    matches = _FINAL.findall(text)
    return matches[-1].strip() if matches else None


def is_refusal(text: str) -> bool:
    final = parse_final_answer(text)
    return bool(_REFUSAL.search(final if final is not None else text))


# ---------------------------------------------------------------- cache


class GenerationCache:
    """Append-only JSONL cache keyed by a hash of everything that affects the output.

    Re-running an evaluation never calls the model twice for the same prompt,
    and the cache file is a complete record of what the model said.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        if self.path.exists():
            for line in open(self.path):
                e = json.loads(line)
                self.entries[e["key"]] = e

    @staticmethod
    def key(backend: str, model: str, params: dict, system: str, prompt: str) -> str:
        blob = json.dumps([backend, model, params, system, prompt], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, entry: dict) -> None:
        self.entries[entry["key"]] = entry
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def spent_usd(self, backend: str) -> float:
        return sum(e.get("cost_usd", 0.0) for e in self.entries.values() if e["backend"] == backend)


# ---------------------------------------------------------------- backends


class OllamaBackend:
    """A local model served by Ollama (free). num_ctx matters: Ollama's default
    context window is small and silently drops the start of long prompts."""

    name = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.0,
                 seed: int = 0, num_ctx: int = 16384, max_tokens: int = 512):
        self.model = model
        self.host = host
        self.params = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": max_tokens}

    def complete(self, system: str, prompt: str) -> dict:
        r = requests.post(f"{self.host}/api/chat", timeout=600, json={
            "model": self.model, "stream": False, "options": self.params,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
        r.raise_for_status()
        body = r.json()
        return {"text": body["message"]["content"], "input_tokens": body.get("prompt_eval_count"),
                "output_tokens": body.get("eval_count"), "cost_usd": 0.0}


class BudgetExceeded(RuntimeError):
    pass


class AnthropicBackend:
    """Anthropic Messages API with a hard spending cap.

    Prices are passed in (from the config) rather than hard-coded, because they
    change. Before each call we charge a worst-case estimate (prompt chars / 3
    tokens + max_tokens) against the cap; after the call we record the real cost.
    Spend is summed from the cache, so restarting does not reset the budget.
    """

    name = "anthropic"

    def __init__(self, model: str, api_key: str, max_usd: float, usd_per_mtok_in: float,
                 usd_per_mtok_out: float, spent_usd: float = 0.0, temperature: float = 0.0,
                 max_tokens: int = 512):
        if not (max_usd and usd_per_mtok_in and usd_per_mtok_out):
            raise ValueError("Anthropic backend needs max_usd and both prices set in the config")
        self.model, self.api_key = model, api_key
        self.max_usd, self.spent = max_usd, spent_usd
        self.price_in, self.price_out = usd_per_mtok_in / 1e6, usd_per_mtok_out / 1e6
        self.params = {"temperature": temperature, "max_tokens": max_tokens}

    def complete(self, system: str, prompt: str) -> dict:
        worst = (len(system + prompt) / 3) * self.price_in + self.params["max_tokens"] * self.price_out
        if self.spent + worst > self.max_usd:
            raise BudgetExceeded(f"spent ${self.spent:.4f}; next call could cost ${worst:.4f}; cap ${self.max_usd}")
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=600, headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": self.model, "system": system, **self.params,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        body = r.json()
        usage = body["usage"]
        cost = usage["input_tokens"] * self.price_in + usage["output_tokens"] * self.price_out
        self.spent += cost
        text = "".join(block.get("text", "") for block in body["content"])
        return {"text": text, "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"], "cost_usd": cost}


def generate(backend, cache: GenerationCache, prompt: str, meta: dict) -> dict:
    """Return the cached generation for this prompt, or call the backend and cache it."""
    key = cache.key(backend.name, backend.model, backend.params, SYSTEM, prompt)
    hit = cache.get(key)
    if hit is not None:
        return hit
    out = backend.complete(SYSTEM, prompt)
    entry = {"key": key, "backend": backend.name, "model": backend.model, "params": backend.params,
             **meta, "prompt": prompt, **out}
    cache.put(entry)
    return entry
