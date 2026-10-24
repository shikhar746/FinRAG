# Design decisions

One entry per choice: what we chose, why, and what would change our mind.
Status: **decided**, **pending** (needs the project owner), or **phase 2**.

## Data

### D1. PDF source: `doc_link` first, FinanceBench GitHub copy as fallback (decided)
- `scripts/download_data.py` tries each document's `doc_link`, then falls back to
  `github.com/patronus-ai/financebench/pdfs/`. `data/pdfs/manifest.csv` records which
  source each file came from, plus its sha256.
- Why: page numbers are only meaningful for the exact PDF the annotators used, and
  company sites re-host or re-render filings over time. The GitHub copy is that exact
  file. In the development container every `doc_link` host was blocked by the network
  policy, so all 84 PDFs came from the GitHub copy.
- Check: D2 confirms the parsed pages match the dataset's evidence text.

### D2. Page numbering is 0-based (decided, verified)
- `evidence_page_num` is the 0-based PyMuPDF page index. We did not assume this:
  `scripts/ingest.py` compares each gold evidence's full-page text with our parsed
  page at offsets -1/0/+1. Offset 0 wins for every evidence item
  (`results/ingest_report.md`).
- The same 0-based ids are shown to the LLM as page tags, so a citation can be compared
  with gold pages directly.

### D3. Dev/test split: 50/100, stratified by question type, fixed seed (decided)
- `splits.json`, made by `scripts/make_splits.py` (seed 20240101). FinanceBench has
  50 questions of each type (metrics-generated, domain-relevant, novel-generated). We
  stratify so dev and test have the same mix (17/17/16 dev).
- Not grouped by document: some filings have questions in both splits. Our design
  choices (retriever, k, prompt) are not document-specific, so this does not leak
  answers. A per-document choice (e.g. tuning per company) would need grouping.
- Dev is for every design choice. Test is run once per final configuration (phase 2).

### D4. MVP retrieval runs on all 150 questions (decided)
- The MVP brief asks for BM25 vs dense on all 150. This is safe because nothing was
  tuned. BM25 and the dense model use library or model defaults, chosen before seeing
  any results. From phase 2 on, design choices use dev only and test numbers are
  reported separately.

## Parsing

### D5. PyMuPDF plain text, one row per page (decided; table-aware variant is phase 2)
- `page.get_text("text")` keeps reading order but flattens tables into one value per
  line. A page with < 50 characters is logged in `data/parse_failures.csv` as
  `little_or_no_text` (scanned image, chart-only or blank page). None of the gold
  pages are in that log.
- Known artefact: words are sometimes split by PDF kerning ("Cash Flow s"). This hurts
  lexical matching.

## Retrieval

### D6. Retrieval unit: the whole page (decided)
- Gold evidence is page-level, so page retrieval can be scored directly with no mapping
  from chunks to pages. Chunking ablations are phase 2.

### D7. Two search scopes (decided)
- `document`: search only the question's filing. This is the usual FinanceBench
  setting, because the question names the company and period.
- `corpus`: search all 12k pages of all 84 filings. Harder, because the retriever must
  also find the right filing. We report it to show how much knowing the filing is worth.

### D8. BM25 settings (decided)
- `rank_bm25.BM25Okapi`, k1=1.5, b=0.75 (library defaults, untuned).
- Tokens: lowercase `\w+`, no stemming, no stopword list. This is the simplest
  reasonable baseline; changing it would be a dev-set ablation.
- Document scope builds one index per filing, so IDF is computed inside the filing.
  Words on every page of a 10-K (the company name, "fiscal") get little weight, which
  is the behaviour we want when searching inside one filing.

### D9. Dense model: `sentence-transformers/all-MiniLM-L6-v2`, run with ONNX (decided)
- Why this model: small (22M parameters, 384-d), fast on CPU, and the most common
  sentence-transformers baseline, so results are easy to compare with other work.
- Why ONNX: Hugging Face was blocked in the development container. Chroma publishes the
  same model as ONNX on S3 with a fixed sha256, which `scripts/download_model.py`
  verifies. `MiniLMEncoder` reimplements sentence-transformers' steps for this model:
  WordPiece tokens, transformer, mean pooling over non-padding tokens, then L2
  normalisation. No torch is needed.
- Index: a NumPy matrix with exact (brute-force) dot-product search. 12k pages × 384
  floats is about 18 MB, so FAISS adds a dependency and saves nothing at this size.

### D10. Dense on long pages: truncate vs best-window (decided)
- The model reads at most 256 tokens, and a 10-K page is often much longer.
  - `truncate` (the sentence-transformers default) embeds only the start of each page.
  - `max_window` embeds 256-token windows that overlap by 32 tokens, and scores each
    page by its best window. The retrieval unit is still the page.
- Why both: without `max_window`, "dense loses to BM25" could just mean "dense only
  read the top of each page".

## Metrics

### D11. Retrieval metrics (decided)
- Recall@k = share of a question's gold pages in the top k. Hit@k = 1 if any gold page
  is in the top k. MRR@100 = mean of 1/rank of the first gold page (0 if not in the
  top 100). k ∈ {1, 5, 10}.
- Hit@k is reported because one evidence page is often enough to answer. Recall@k is
  reported because multi-page questions (35 of 150) need every page.

### D12. Paired bootstrap for comparisons (decided)
- 10,000 resamples of questions, seed 13. Each resample keeps both systems' scores on
  the same question together. We call a difference real only if the 95% CI of
  mean(B − A) excludes zero.

## Generation (built and unit-tested, not yet run)

### D13. LLM backend (pending)
- Options: a local model via Ollama (free), or the Anthropic API with a hard dollar
  cap (`AnthropicBackend` refuses any call that could exceed `max_usd`, and counts
  spend from the cache so a restart does not reset it). Both live in `finrag/generate.py`.
- Note: in the development container, `registry.ollama.ai` is blocked, so Ollama models
  cannot be pulled there.

### D14. Prompting and output format (decided, untested on a model)
- Temperature 0. Pages are shown as `<page id="N">` with 0-based ids. The model must
  cite `[p. N]` and finish with `Final answer: ...`. We score only that final line,
  so a model that lists many numbers in its reasoning cannot get lucky.
- RAG gets the top `rag_k = 5` BM25 pages (document scope), in page order. 5 was fixed
  in advance, not tuned: at about 1–2k tokens per page it fits a 16k context window.
  Changing k is a dev-set ablation.
- Ollama `num_ctx = 16384`: Ollama's default window is small and silently cuts long
  prompts, which would look like a retrieval failure.

### D15. Numeric answers (tolerance pending)
- A gold answer is "numeric" only if the whole string is a number (`$1577.00`,
  `-0.02`, `10.3%`). Sentence answers ("$400,000,000 increase.") go to hand labels.
- "%" is dropped without dividing by 100. `allow_scale` (default off) optionally accepts
  answers that are exactly 1e3/1e6/1e9 off (unit slips such as "1.577 billion" vs
  "1577" in millions).
- `numeric_rel_tol`: **pending**.

### D16. MVP generation uses the 50 dev questions only (decided)
- Keeps the 100 test questions untouched until phase 2. The 50 dev answers per config
  are also the 50 hand checks the MVP needs.

### D17. LLM judge (phase 2)
- Not used. It will be used only if Cohen's kappa against 60 hand labels is acceptable.
