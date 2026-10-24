#!/usr/bin/env bash
# Run the MVP generation configs with a local Ollama model, then build the labelling sheet.
#
#   ollama serve &            # if Ollama is not already running
#   bash scripts/run_generation.sh
#
# Re-running is safe: answers are cached in data/cache/generations.jsonl.
set -euo pipefail
PY=${PY:-.venv/bin/python}
export PYTHONPATH=.

MODEL=$($PY -c "import yaml; print(yaml.safe_load(open('configs/base.yaml'))['llm']['model'])")
if ! ollama list | awk 'NR>1 {print $1}' | grep -qxF "$MODEL"; then
  ollama pull "$MODEL"
fi
# Record the exact model build: tags like "qwen2.5:7b" can point to new weights later.
mkdir -p results/generation
ollama list | awk -v m="$MODEL" 'NR==1 || $1==m' > results/generation/ollama_model.txt

for c in closed_book bm25_rag; do
  $PY scripts/run_eval.py "configs/$c.yaml"
done
$PY scripts/make_label_csv.py closed_book bm25_rag
$PY scripts/update_readme.py
echo "Next: fill human_correct (1/0) in results/labels/to_label.csv, then run scripts/score_labels.py"
