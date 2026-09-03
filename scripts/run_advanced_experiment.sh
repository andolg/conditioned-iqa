#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?config path required}
DEVICE=${2:?CUDA device required}
RUN_NAME=${3:?MLflow run name required}
DESIGN=${4:?sheet design required}
BACKBONE_LABEL=${5:?sheet backbone label required}
BASELINE=${6:-variant}
DESCRIPTION=${7:?sheet description required}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/runs/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_NAME}.log"

echo "[$(date -Is)] training $RUN_NAME on $DEVICE" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" "$PYTHON" train_text_conditioned.py \
  --config "$CONFIG" --device cuda:0 --mlflow-tracking-uri sqlite:///mlflow.db \
  --results-csv runs/results.csv 2>&1 | tee -a "$LOG"

RUN_ID=$(RUN_NAME="$RUN_NAME" "$PYTHON" - <<'PY'
import os
from mlflow.tracking import MlflowClient
client = MlflowClient("sqlite:///mlflow.db")
experiment = client.get_experiment_by_name("conditioned-iqa-performance")
runs = client.search_runs([experiment.experiment_id], filter_string=f"tags.mlflow.runName = '{os.environ['RUN_NAME']}'", order_by=["attributes.start_time DESC"])
if not runs:
    raise SystemExit("trained run not found")
print(runs[0].info.run_id)
PY
)
echo "[$(date -Is)] source run $RUN_ID; evaluating held-out suite" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" "$PYTHON" evaluate_text_conditioned.py \
  --source-run-id "$RUN_ID" \
  --data /home/sergey/conditioned-iqa/data/tid2013/labels.csv /home/sergey/conditioned-iqa/data/csiq/labels.csv /home/sergey/conditioned-iqa/data/cid2013/labels.csv /home/sergey/conditioned-iqa/data/koniq10k/labels.csv /home/sergey/conditioned-iqa/data/clive/labels.csv /home/sergey/conditioned-iqa/data/agiqa3k/labels.csv /home/sergey/conditioned-iqa/data/gfiqa20k/labels.csv /home/sergey/conditioned-iqa/data/pipal/labels.csv /home/sergey/conditioned-iqa/data/uhdiqa/labels.csv \
  --device cuda:0 --batch-size 64 --workers 4 --mlflow-tracking-uri sqlite:///mlflow.db \
  --mlflow-experiment conditioned-iqa-external-eval --results-csv runs/results.csv \
  2>&1 | tee -a "$LOG"

"$PYTHON" scripts/append_run_to_sheet.py --run-name "$RUN_NAME" --design "$DESIGN" \
  --description "$DESCRIPTION" --backbone "$BACKBONE_LABEL" --baseline "$BASELINE" \
  --results runs/results.csv 2>&1 | tee -a "$LOG"
echo "[$(date -Is)] completed $RUN_NAME" | tee -a "$LOG"
