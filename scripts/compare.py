"""Paired bootstrap comparisons between retrieval configurations.

    python scripts/compare.py bm25_page dense_page_truncate dense_page_maxwindow

Every pair (A, B) is compared on the same questions, per scope, for MRR,
Recall@5 and Recall@10. Output: results/retrieval/comparisons.md. A difference
counts as real only if its 95% CI excludes zero.
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

from finrag.evaluate import paired_bootstrap

METRICS = {"rr": "MRR@100", "recall@5": "Recall@5", "recall@10": "Recall@10"}
N_BOOT, SEED = 10_000, 13


def main() -> None:
    names = sys.argv[1:]
    out = Path("results/retrieval")
    runs = {n: pd.read_csv(out / f"{n}_per_question.csv") for n in names}

    lines = ["### Paired bootstrap: B minus A", "",
             f"{N_BOOT} resamples of questions, seed {SEED}. * = 95% CI excludes zero.", "",
             "| A | B | scope | metric | mean(B-A) | 95% CI | n |", "|---|---|---|---|---|---|---|"]
    for a, b in itertools.combinations(names, 2):
        for scope in runs[a].scope.unique():
            ra = runs[a][runs[a].scope == scope].set_index("financebench_id").sort_index()
            rb = runs[b][runs[b].scope == scope].set_index("financebench_id").sort_index()
            if list(ra.index) != list(rb.index):
                raise SystemExit(f"{a} and {b} were run on different questions")
            for col, label in METRICS.items():
                r = paired_bootstrap(ra[col], rb[col], N_BOOT, SEED)
                star = " *" if r["excludes_zero"] else ""
                lines.append(f"| {a} | {b} | {scope} | {label} | {r['mean_diff']:+.3f}{star} | "
                             f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['n']} |")
    md = "\n".join(lines) + "\n"
    (out / "comparisons.md").write_text(md)
    print(md)

    # Breakdown by FinanceBench question type, document scope (descriptive only: ~50 questions each).
    lines = ["### Document scope, by question type", "",
             "| config | question_type | n | Hit@5 | Recall@10 | MRR@100 |", "|---|---|---|---|---|---|"]
    for n, df in runs.items():
        d = df[df.scope == "document"]
        for qt, g in d.groupby("question_type"):
            lines.append(f"| {n} | {qt} | {len(g)} | {g['hit@5'].mean():.3f} | "
                         f"{g['recall@10'].mean():.3f} | {g['rr'].mean():.3f} |")
    md = "\n".join(lines) + "\n"
    (out / "by_question_type.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
