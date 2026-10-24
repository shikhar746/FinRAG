"""Score the hand-labelled sheet: accuracy per config and closed-book vs RAG.

    python scripts/score_labels.py      # reads results/labels/to_label.csv

- Accuracy uses `human_correct` (1/0) for every row. A row left blank stops the
  script, so no partly labelled run is ever reported.
- On numeric questions it also reports how often the automatic matcher agreed
  with you, which checks the matcher itself.
- The closed-book vs RAG difference gets a paired bootstrap 95% CI (same questions).
Output: results/generation/hand_labels.md
"""
from pathlib import Path

import pandas as pd

from finrag.evaluate import paired_bootstrap

SHEET = Path("results/labels/to_label.csv")
OUT = Path("results/generation/hand_labels.md")


def main() -> None:
    df = pd.read_csv(SHEET)
    if df.human_correct.isna().any():
        missing = int(df.human_correct.isna().sum())
        raise SystemExit(f"{missing} rows have no human_correct label yet")
    df["human_correct"] = df.human_correct.astype(float)

    lines = ["### Hand-labelled answer accuracy (dev questions)", "",
             "| config | n | accuracy | numeric rows | auto-vs-human agreement on numeric |",
             "|---|---|---|---|---|"]
    for name, g in df.groupby("config"):
        num = g[g.numeric_correct.notna() & (g.numeric_correct.astype(str) != "")]
        agree = (num.numeric_correct.astype(float) == num.human_correct).mean() if len(num) else float("nan")
        lines.append(f"| {name} | {len(g)} | {g.human_correct.mean():.3f} | {len(num)} | {agree:.3f} |")

    configs = sorted(df.config.unique())
    if {"closed_book", "bm25_rag"} <= set(configs):
        wide = df.pivot(index="financebench_id", columns="config", values="human_correct").dropna()
        r = paired_bootstrap(wide["closed_book"], wide["bm25_rag"], seed=13)
        star = " *" if r["excludes_zero"] else ""
        lines += ["", "| A | B | mean(B-A) | 95% CI | n |", "|---|---|---|---|---|",
                  f"| closed_book | bm25_rag | {r['mean_diff']:+.3f}{star} | "
                  f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['n']} |"]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
