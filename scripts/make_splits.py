"""Make the fixed dev/test split and save it to splits.json (committed).

- 50 dev questions: used for every design choice (k, prompts, models, ...).
- 100 test questions: touched once per final configuration.

The split is stratified by `question_type` (FinanceBench has 50 each of
metrics-generated, domain-relevant and novel-generated), so dev and test have
the same mix of question styles. Re-running with the same seed gives the same
split; the script refuses to overwrite an existing different split.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20240101
N_DEV = 50


def make_split(questions: list[dict], n_dev: int, seed: int) -> dict:
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for q in sorted(questions, key=lambda q: q["financebench_id"]):
        by_type[q["question_type"]].append(q["financebench_id"])

    # Give each type a share of dev proportional to its size (largest remainder).
    total = len(questions)
    quotas = {t: n_dev * len(ids) / total for t, ids in by_type.items()}
    alloc = {t: int(x) for t, x in quotas.items()}
    for t in sorted(quotas, key=lambda t: quotas[t] - alloc[t], reverse=True)[: n_dev - sum(alloc.values())]:
        alloc[t] += 1

    dev = []
    for t in sorted(by_type):
        ids = by_type[t][:]
        rng.shuffle(ids)
        dev += ids[: alloc[t]]
    test = sorted(set(q["financebench_id"] for q in questions) - set(dev))
    return {"seed": seed, "stratified_by": "question_type", "dev": sorted(dev), "test": test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/raw/financebench_open_source.jsonl")
    ap.add_argument("--out", default="splits.json")
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(args.questions)]
    split = make_split(questions, N_DEV, SEED)
    out = Path(args.out)
    if out.exists() and json.loads(out.read_text()) != split:
        raise SystemExit(f"{out} exists and differs; refusing to overwrite a committed split")
    out.write_text(json.dumps(split, indent=2) + "\n")
    print(f"dev={len(split['dev'])} test={len(split['test'])} -> {out}")


if __name__ == "__main__":
    main()
