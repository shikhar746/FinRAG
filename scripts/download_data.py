"""Download the FinanceBench open sample (questions + PDFs).

Usage:
    python scripts/download_data.py            # prints total size, asks before downloading
    python scripts/download_data.py --yes      # no prompt

Where files come from:
- Questions and document metadata: the official FinanceBench GitHub repo
  (same content as the Hugging Face dataset PatronusAI/financebench).
- PDFs: we try each document's `doc_link` first (the company's own site, as the
  dataset specifies). If that fails, we fall back to the copy of the same file
  in the FinanceBench GitHub repo, which is the exact file the annotators used.

Every PDF is recorded in data/pdfs/manifest.csv with its source and sha256, so
anyone can check which copy they parsed.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import requests

MIRROR = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_FILE = "financebench_open_source.jsonl"
DOCINFO_FILE = "financebench_document_information.jsonl"
TIMEOUT = 30
HEADERS = {"User-Agent": "FinRAG-Eval research script (academic, non-commercial)"}


def fetch(url: str) -> bytes:
    r = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    return r.content


def head_size(url: str) -> int:
    r = requests.head(url, timeout=TIMEOUT, headers=HEADERS, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers.get("Content-Length", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--yes", action="store_true", help="skip the size confirmation")
    args = ap.parse_args()

    raw_dir = Path(args.out) / "raw"
    pdf_dir = Path(args.out) / "pdfs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # 1. Questions + document metadata (small JSONL files).
    for name in (QUESTIONS_FILE, DOCINFO_FILE):
        (raw_dir / name).write_bytes(fetch(f"{MIRROR}/data/{name}"))
    questions = [json.loads(l) for l in open(raw_dir / QUESTIONS_FILE)]
    needed = {q["doc_name"] for q in questions}
    docs = [json.loads(l) for l in open(raw_dir / DOCINFO_FILE)]
    docs = [d for d in docs if d["doc_name"] in needed]
    print(f"{len(questions)} questions over {len(docs)} documents")

    # 2. Report total size before downloading anything big.
    todo = [d for d in docs if not (pdf_dir / f"{d['doc_name']}.pdf").exists()]
    total = sum(head_size(f"{MIRROR}/pdfs/{d['doc_name']}.pdf") for d in todo)
    print(f"{len(todo)} PDFs to download, total {total / 1e6:.1f} MB")
    if todo and not args.yes and input("Download? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    # 3. Download: doc_link first, mirror as fallback.
    manifest = []
    for d in docs:
        path = pdf_dir / f"{d['doc_name']}.pdf"
        source = "cached"
        if not path.exists():
            try:
                data = fetch(d["doc_link"])
                if not data.startswith(b"%PDF"):
                    raise ValueError("doc_link did not return a PDF")
                source = "doc_link"
            except Exception as e:  # blocked host, 404, HTML page, ...
                print(f"  {d['doc_name']}: doc_link failed ({type(e).__name__}), using mirror")
                data = fetch(f"{MIRROR}/pdfs/{d['doc_name']}.pdf")
                source = "mirror"
            path.write_bytes(data)
        manifest.append({
            "doc_name": d["doc_name"],
            "source": source,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "doc_link": d["doc_link"],
        })

    with open(pdf_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"done: {len(manifest)} PDFs in {pdf_dir}")


if __name__ == "__main__":
    main()
