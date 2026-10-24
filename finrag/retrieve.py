"""Page retrievers: BM25 (lexical) and dense (MiniLM embeddings).

Both share one interface:

    retriever.search(query, doc_name=None, top_k=100) -> list[(doc_name, page, score)]

`doc_name=None` searches the whole corpus (all filings). Passing a doc_name
restricts the search to that filing, which is the usual FinanceBench setting
(the question names the company and period, so the filing is known).
"""
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer


def tokenize(text: str) -> list[str]:
    """Lowercased word/number tokens. No stemming or stopwords (kept simple on purpose)."""
    return re.findall(r"\w+", text.lower())


def _top(keys: list[tuple[str, int]], scores: np.ndarray, top_k: int) -> list[tuple[str, int, float]]:
    # Stable sort so ties keep page order (deterministic results).
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(*keys[i], float(scores[i])) for i in order]


class BM25Retriever:
    """BM25 over whole pages.

    Document scope uses one index per filing, so IDF is computed within that
    filing: words that appear on every page of a 10-K (the company name,
    "fiscal", ...) get low weight, which is what we want when searching inside it.
    Corpus scope uses one index over all pages.
    """

    def __init__(self, pages: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.pages = pages.sort_values(["doc_name", "page"]).reset_index(drop=True)
        self._indexes: dict[str | None, tuple[BM25Okapi, list]] = {}

    def _index(self, doc_name: str | None):
        if doc_name not in self._indexes:
            sub = self.pages if doc_name is None else self.pages[self.pages.doc_name == doc_name]
            keys = list(zip(sub.doc_name, sub.page))
            bm25 = BM25Okapi([tokenize(t) for t in sub.text], k1=self.k1, b=self.b)
            self._indexes[doc_name] = (bm25, keys)
        return self._indexes[doc_name]

    def search(self, query: str, doc_name: str | None = None, top_k: int = 100):
        bm25, keys = self._index(doc_name)
        return _top(keys, bm25.get_scores(tokenize(query)), top_k)


class MiniLMEncoder:
    """sentence-transformers/all-MiniLM-L6-v2 run with ONNX Runtime.

    This reproduces what sentence-transformers does for this model:
    WordPiece tokenize -> transformer -> mean of token vectors (ignoring
    padding) -> L2 normalise. After normalising, dot product = cosine similarity.
    """

    def __init__(self, model_dir: str | Path, max_tokens: int = 256, batch_size: int = 64):
        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0  # use all cores
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), opts,
                                            providers=["CPUExecutionProvider"])

    def _embed(self, encodings) -> np.ndarray:
        """Run the model on a list of tokenizer encodings (already <= max_tokens)."""
        out = np.zeros((len(encodings), 384), dtype=np.float32)
        # Sort by length so each batch pads to a similar size (much faster on CPU).
        order = sorted(range(len(encodings)), key=lambda i: len(encodings[i].ids))
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            width = max(len(encodings[i].ids) for i in idx)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros((len(idx), width), dtype=np.int64)
            for row, i in enumerate(idx):
                n = len(encodings[i].ids)
                ids[row, :n] = encodings[i].ids
                mask[row, :n] = 1
            hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask,
                                             "token_type_ids": np.zeros_like(ids)})[0]
            summed = (hidden * mask[..., None]).sum(axis=1)
            mean = summed / np.clip(mask.sum(axis=1, keepdims=True), 1, None)
            out[idx] = mean / np.linalg.norm(mean, axis=1, keepdims=True)
        return out

    def encode(self, texts: list[str]) -> np.ndarray:
        """One vector per text; text beyond max_tokens is cut off (sentence-transformers default)."""
        self.tokenizer.enable_truncation(self.max_tokens)
        return self._embed(self.tokenizer.encode_batch(texts))

    def encode_windows(self, texts: list[str], stride: int) -> tuple[np.ndarray, np.ndarray]:
        """Split each text into overlapping max_tokens windows and embed every window.

        Returns (vectors, owner) where owner[j] is the index of the text window j came from.
        `stride` is the number of tokens shared by neighbouring windows.
        """
        self.tokenizer.enable_truncation(self.max_tokens, stride=stride)
        windows, owner = [], []
        for i, enc in enumerate(self.tokenizer.encode_batch(texts)):
            for w in [enc, *enc.overflowing]:
                windows.append(w)
                owner.append(i)
        return self._embed(windows), np.array(owner)


class DenseRetriever:
    """Cosine-similarity search over page embeddings, held in a NumPy matrix.

    mode="truncate":   one vector per page from its first max_tokens tokens.
    mode="max_window": one vector per window; a page's score is its best window.
                       The retrieval unit is still the whole page.
    Embeddings are cached to `cache_dir` so they are computed once.
    """

    def __init__(self, pages: pd.DataFrame, encoder: MiniLMEncoder, mode: str = "truncate",
                 stride: int = 32, cache_dir: str | Path | None = None):
        if mode not in ("truncate", "max_window"):
            raise ValueError(mode)
        self.pages = pages.sort_values(["doc_name", "page"]).reset_index(drop=True)
        self.keys = list(zip(self.pages.doc_name, self.pages.page))
        self.encoder, self.mode = encoder, mode

        cache = Path(cache_dir) / f"dense_{mode}_{encoder.max_tokens}_{stride}.npz" if cache_dir else None
        if cache and cache.exists():
            z = np.load(cache)
            self.vectors, self.owner = z["vectors"], z["owner"]
        else:
            texts = self.pages.text.tolist()
            if mode == "truncate":
                self.vectors, self.owner = encoder.encode(texts), np.arange(len(texts))
            else:
                self.vectors, self.owner = encoder.encode_windows(texts, stride)
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, vectors=self.vectors, owner=self.owner)
        if self.owner.max() != len(self.pages) - 1:
            raise ValueError("embedding cache does not match pages; delete it and re-run")

    def page_scores(self, query: str) -> np.ndarray:
        sims = self.vectors @ self.encoder.encode([query])[0]
        scores = np.full(len(self.pages), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self.owner, sims)  # page score = best window (= the vector, in truncate mode)
        return scores

    def search(self, query: str, doc_name: str | None = None, top_k: int = 100):
        scores = self.page_scores(query)
        if doc_name is not None:
            scores = np.where(self.pages.doc_name.to_numpy() == doc_name, scores, -np.inf)
            top_k = min(top_k, int((self.pages.doc_name == doc_name).sum()))
        return _top(self.keys, scores, top_k)
