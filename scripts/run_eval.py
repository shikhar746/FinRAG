"""Run one configuration and write its results table.

    python scripts/run_eval.py configs/bm25_page.yaml
    python scripts/run_eval.py configs/bm25_rag.yaml      # needs llm.backend set

Retrieval configs  -> results/retrieval/<name>.md and <name>_per_question.csv
Generation configs -> results/generation/<name>.md and <name>_answers.csv
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml

from finrag.evaluate import (citation_scores, first_number, hit_at_k, numbers_match, parse_number,
                             recall_at_k, reciprocal_rank, rounding_tol)
from finrag.generate import (AnthropicBackend, GenerationCache, OllamaBackend, build_prompt, generate,
                             is_refusal, parse_citations, parse_final_answer)
from finrag.ingest import gold_pages, load_questions
from finrag.retrieve import BM25Retriever, DenseRetriever, MiniLMEncoder


def load_config(path: str | Path) -> dict:
    """Load a YAML config; `base:` names a file (same folder) whose keys it overrides."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = _merge(base, cfg)
    return cfg


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def select_questions(cfg: dict, which: str) -> list[dict]:
    qs = load_questions(Path(cfg["paths"]["data"]) / "raw/financebench_open_source.jsonl")
    splits = json.loads(Path(cfg["paths"]["splits"]).read_text())
    split_of = {qid: s for s in ("dev", "test") for qid in splits[s]}
    for q in qs:
        q["split"] = split_of[q["financebench_id"]]
    return qs if which == "all" else [q for q in qs if q["split"] == which]


def build_retriever(cfg: dict, pages: pd.DataFrame, kind: str):
    if kind == "bm25":
        return BM25Retriever(pages, **cfg["bm25"])
    if kind == "dense":
        d = cfg["dense"]
        enc = MiniLMEncoder(d["model_dir"], d["max_tokens"], d["batch_size"])
        return DenseRetriever(pages, enc, cfg["dense_mode"], d["window_stride"], cfg["paths"]["cache"])
    raise ValueError(kind)


def table(df: pd.DataFrame, cols: list[str], floatfmt: str = ".3f") -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    rows = []
    for _, r in df[cols].iterrows():
        rows.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in r) + " |")
    return head + "\n".join(rows) + "\n"


# ---------------------------------------------------------------- retrieval


def run_retrieval(cfg: dict) -> None:
    pages = pd.read_parquet(Path(cfg["paths"]["data"]) / "pages.parquet")
    retriever = build_retriever(cfg, pages, cfg["retriever"])
    ks, top_k = cfg["retrieval"]["ks"], cfg["retrieval"]["top_k"]

    rows = []
    for q in select_questions(cfg, cfg["questions"]):
        gold = {(q["doc_name"], p) for p in gold_pages(q)}
        for scope in cfg["retrieval"]["scopes"]:
            hits = retriever.search(q["question"], q["doc_name"] if scope == "document" else None, top_k)
            ranked = [(d, p) for d, p, _ in hits]
            row = {"financebench_id": q["financebench_id"], "split": q["split"],
                   "question_type": q["question_type"], "scope": scope,
                   "gold_pages": sorted(p for _, p in gold),
                   "top10": [f"{d}:{p}" if scope == "corpus" else p for d, p in ranked[:10]],
                   "rr": reciprocal_rank(ranked, gold),
                   # Did corpus-wide search even find the right filing at rank 1?
                   "top1_right_doc": float(ranked[0][0] == q["doc_name"])}
            for k in ks:
                row[f"recall@{k}"] = recall_at_k(ranked, gold, k)
                row[f"hit@{k}"] = hit_at_k(ranked, gold, k)
            rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(cfg["paths"]["results"]) / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{cfg['name']}_per_question.csv", index=False)

    metric_cols = [f"recall@{k}" for k in ks] + [f"hit@{k}" for k in ks] + ["rr"]
    summary = df.groupby("scope", sort=False)[metric_cols].mean().reset_index()
    summary.insert(1, "n", df.groupby("scope", sort=False).size().values)
    summary = summary.rename(columns={"rr": "MRR@100"})
    md = f"### {cfg['name']}\n\n" + table(summary, list(summary.columns))
    (out / f"{cfg['name']}.md").write_text(md)
    print(md)


# ---------------------------------------------------------------- generation


def build_backend(cfg: dict, cache: GenerationCache):
    llm = cfg["llm"]
    if llm["backend"] is None or llm["model"] is None:
        sys.exit("llm.backend / llm.model are not set in configs/base.yaml.")
    if llm["backend"] == "ollama":
        return OllamaBackend(llm["model"], llm["ollama"]["host"], llm["temperature"], cfg["seed"],
                             llm["ollama"]["num_ctx"], llm["max_tokens"])
    if llm["backend"] == "anthropic":
        a = llm["anthropic"]
        return AnthropicBackend(llm["model"], os.environ["ANTHROPIC_API_KEY"], a["max_usd"],
                                a["usd_per_mtok_in"], a["usd_per_mtok_out"],
                                cache.spent_usd("anthropic"), llm["temperature"], llm["max_tokens"])
    raise ValueError(llm["backend"])


def run_generation(cfg: dict) -> None:
    ans = cfg["answers"]
    pages = pd.read_parquet(Path(cfg["paths"]["data"]) / "pages.parquet")
    text = {(r.doc_name, r.page): r.text for r in pages.itertuples()}
    retriever = BM25Retriever(pages, **cfg["bm25"]) if cfg["context"] == "bm25" else None
    cache = GenerationCache(Path(cfg["paths"]["cache"]) / "generations.jsonl")
    backend = build_backend(cfg, cache)

    rows = []
    for q in select_questions(cfg, ans["questions"]):
        gold = gold_pages(q)
        retrieved = None
        if retriever is not None:
            retrieved = [p for _, p, _ in retriever.search(q["question"], q["doc_name"], ans["rag_k"])]
            context = [(p, text[(q["doc_name"], p)]) for p in sorted(retrieved)]
        else:
            context = None
        prompt = build_prompt(q["question"], q["doc_name"], context)
        try:
            out = generate(backend, cache, prompt, {"financebench_id": q["financebench_id"], "config": cfg["name"]})
        except requests.ConnectionError:
            sys.exit(f"Cannot reach Ollama at {cfg['llm']['ollama']['host']}. Start it with `ollama serve` "
                     f"and pull the model with `ollama pull {cfg['llm']['model']}`.")

        final = parse_final_answer(out["text"])
        gold_num = parse_number(q["answer"])
        row = {"financebench_id": q["financebench_id"], "question_type": q["question_type"],
               "question": q["question"], "gold_answer": q["answer"], "model_answer": out["text"],
               "final_answer": final, "refusal": float(is_refusal(out["text"])),
               "gold_pages": sorted(gold), "retrieved_pages": retrieved,
               "retrieval_hit": float(bool(set(retrieved or []) & gold)) if retriever else None,
               "gold_is_numeric": gold_num is not None,
               # Ollama silently drops the start of prompts longer than num_ctx; flag any that came close.
               "context_overflow": float((out.get("input_tokens") or 0)
                                         >= cfg["llm"]["ollama"]["num_ctx"] - cfg["llm"]["max_tokens"])}
        if gold_num is not None:
            pred = first_number(final) if final else None
            abs_tol = rounding_tol(q["answer"]) if ans["gold_rounding_tol"] else 0.0
            row["numeric_correct"] = float(numbers_match(pred, gold_num, ans["numeric_rel_tol"],
                                                         abs_tol, ans["allow_scale"]))
            # Diagnostic only: would it be right if we forgave a thousand/million/billion unit slip?
            row["numeric_correct_any_scale"] = float(numbers_match(pred, gold_num, ans["numeric_rel_tol"],
                                                                   abs_tol, allow_scale=True))
        if retriever is not None:
            row["cited_pages"] = sorted(parse_citations(out["text"]))
            row.update(citation_scores(set(row["cited_pages"]), gold))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = Path(cfg["paths"]["results"]) / "generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{cfg['name']}_answers.csv", index=False)

    num = df[df.gold_is_numeric]
    summary = {"config": cfg["name"], "n": len(df), "n_numeric": len(num),
               "numeric_accuracy": num["numeric_correct"].mean(),
               "numeric_acc_any_scale": num["numeric_correct_any_scale"].mean(),
               "refusal_rate": df.refusal.mean(), "context_overflow": int(df.context_overflow.sum())}
    if retriever is not None:
        summary.update({"retrieval_hit@k": df.retrieval_hit.mean(), "cited_gold": df.cited_gold.mean(),
                        "citation_precision": df.citation_precision.mean()})
    s = pd.DataFrame([summary])
    md = f"### {cfg['name']}\n\n" + table(s, list(s.columns))
    (out_dir / f"{cfg['name']}.md").write_text(md)
    print(md)


def main() -> None:
    cfg = load_config(sys.argv[1])
    {"retrieval": run_retrieval, "generation": run_generation}[cfg["task"]](cfg)


if __name__ == "__main__":
    main()
