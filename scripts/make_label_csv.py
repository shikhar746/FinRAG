"""Build the hand-labelling sheet from the generation runs.

    python scripts/make_label_csv.py closed_book bm25_rag

Writes results/labels/to_label.csv: one row per (question, config) with the
question, gold answer, model answer, retrieved/cited/gold pages and the
automatic numeric verdict (if any). Fill in `human_correct` (1/0) and,
optionally, `notes`. Rows are shuffled with a fixed seed and the config name
is kept in its own column, so you can hide it while labelling to avoid bias.
"""
import sys
from pathlib import Path

import pandas as pd

SEED = 13
COLS = ["financebench_id", "config", "question", "gold_answer", "final_answer", "model_answer",
        "retrieved_pages", "cited_pages", "gold_pages", "numeric_correct", "human_correct", "notes"]


def main() -> None:
    frames = []
    for name in sys.argv[1:]:
        df = pd.read_csv(Path("results/generation") / f"{name}_answers.csv")
        df["config"] = name
        frames.append(df)
    df = pd.concat(frames).sample(frac=1.0, random_state=SEED)
    for col in COLS:
        if col not in df:
            df[col] = ""
    out = Path("results/labels")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "to_label.csv"
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite labels")
    df[COLS].to_csv(path, index=False)
    print(f"{len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
