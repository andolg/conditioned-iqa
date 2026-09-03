#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
RUN_LOG="$ROOT/runs/logs/multiview-clean-uniform-interaction-s0.log"
SOURCE_RUN_ID=23964c286e144423b73fca861bd89845

DATA=(
  /home/sergey/conditioned-iqa/data/tid2013/labels.csv
  /home/sergey/conditioned-iqa/data/csiq/labels.csv
  /home/sergey/conditioned-iqa/data/cid2013/labels.csv
  /home/sergey/conditioned-iqa/data/koniq10k/labels.csv
  /home/sergey/conditioned-iqa/data/clive/labels.csv
  /home/sergey/conditioned-iqa/data/agiqa3k/labels.csv
  /home/sergey/conditioned-iqa/data/gfiqa20k/labels.csv
  /home/sergey/conditioned-iqa/data/pipal/labels.csv
  /home/sergey/conditioned-iqa/data/uhdiqa/labels.csv
)

echo "[$(date -Is)] retrying uniform external evaluation on cuda:4" | tee -a "$RUN_LOG"
CUDA_VISIBLE_DEVICES=4 "$PYTHON" evaluate_text_conditioned.py \
  --source-run-id "$SOURCE_RUN_ID" --data "${DATA[@]}" \
  --device cuda:0 --batch-size 64 --workers 4 \
  --mlflow-tracking-uri sqlite:///mlflow.db \
  --mlflow-experiment conditioned-iqa-external-eval \
  --results-csv runs/results.csv 2>&1 | tee -a "$RUN_LOG"

"$PYTHON" scripts/append_run_to_sheet.py \
  --run-name multiview-clean-uniform-interaction-s0 \
  --design "CLIP-B/16 five-view uniform interaction" \
  --description "Uniform global-plus-four-local view pooling with native CLIP-B/16 text interaction; removes learned quality-aware view weighting while keeping the clean mixture, split, and seed fixed." \
  --backbone "CLIP-B/16" --baseline variant \
  --results runs/results.csv 2>&1 | tee -a "$RUN_LOG"

bash scripts/run_advanced_experiment.sh \
  configs/text_conditioning/65_clean_multi_multiview_mdtvsfa_interaction.yaml \
  cuda:4 multiview-clean-mdtvsfa-interaction-s0 \
  "CLIP-B/16 five-view calibrated interaction" \
  "CLIP-B/16" variant \
  "Five-view CLIP-B/16 native interaction with the row-14 legacy MDTVSFA-style per-dataset calibration and ranking loss; same clean-mixture data, split, and seed." \
  | tee -a "$ROOT/runs/logs/clipb-full-queue.log"

echo "[$(date -Is)] uniform retry and calibration completed" | tee -a "$ROOT/runs/logs/clipb-full-queue.log"
