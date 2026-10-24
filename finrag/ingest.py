"""Ingestion: load FinanceBench questions and turn PDFs into one row per page.

A "page" here is a PDF page as PyMuPDF sees it, identified by
(doc_name, page) where `page` is the 0-based index into the PDF. FinanceBench's
`evidence_page_num` uses the same 0-based convention; `check_gold_alignment`
verifies this against the evidence text instead of assuming it.
"""
import json
import re
from pathlib import Path

import pymupdf
import pandas as pd

# A page with fewer characters than this is logged as a likely parse failure
# (scanned image, chart-only page, or genuinely blank page).
MIN_CHARS = 50


def load_questions(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def gold_pages(question: dict) -> set[int]:
    """Gold evidence pages (0-based) for a question. Duplicates collapse."""
    return {int(e["evidence_page_num"]) for e in question["evidence"]}


def parse_pdf(pdf_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (page rows, failure rows) for one PDF."""
    doc_name = pdf_path.stem
    pages, failures = [], []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return [], [{"doc_name": doc_name, "page": -1, "reason": "open_failed", "detail": repr(e)}]

    with doc:
        for i, page in enumerate(doc):
            try:
                text = page.get_text("text")
            except Exception as e:
                text = ""
                failures.append({"doc_name": doc_name, "page": i, "reason": "extract_failed", "detail": repr(e)})
            else:
                if len(text.strip()) < MIN_CHARS:
                    failures.append({"doc_name": doc_name, "page": i, "reason": "little_or_no_text",
                                     "detail": f"{len(text.strip())} chars"})
            pages.append({"doc_name": doc_name, "page": i, "text": text, "n_chars": len(text)})
    return pages, failures


def parse_all(pdf_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pages, failures = [], []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        p, f = parse_pdf(pdf)
        pages += p
        failures += f
    fail_cols = ["doc_name", "page", "reason", "detail"]
    return pd.DataFrame(pages), pd.DataFrame(failures, columns=fail_cols)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def overlap(a: str, b: str) -> float:
    """Share of a's word types that also appear in b (0..1)."""
    ta = _tokens(a)
    return len(ta & _tokens(b)) / len(ta) if ta else 0.0


def check_gold_alignment(questions: list[dict], pages: pd.DataFrame,
                         offsets=(-1, 0, 1)) -> pd.DataFrame:
    """For every gold evidence item, compare the dataset's full-page text with
    our parsed text at page p+offset. The offset with the highest overlap tells
    us which page-numbering convention the dataset uses, and low overlap at the
    best offset flags a parse problem or a different PDF version."""
    text = {(r.doc_name, r.page): r.text for r in pages.itertuples()}
    rows = []
    for q in questions:
        for ev in q["evidence"]:
            p = int(ev["evidence_page_num"])
            row = {"financebench_id": q["financebench_id"], "doc_name": ev["doc_name"], "gold_page": p}
            for off in offsets:
                row[f"overlap_{off:+d}"] = overlap(ev["evidence_text_full_page"],
                                                   text.get((ev["doc_name"], p + off), ""))
            rows.append(row)
    return pd.DataFrame(rows)
