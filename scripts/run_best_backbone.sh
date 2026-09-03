#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?config path required}
DEVICE=${2:?CUDA device required}
RUN_NAME=${3:?MLflow run name required}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/runs/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_NAME}.log"

echo "[$(date -Is)] training $RUN_NAME on $DEVICE" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" "$PYTHON" train_text_conditioned.py \
  --config "$CONFIG" \
  --device cuda:0 \
  --mlflow-tracking-uri sqlite:///mlflow.db \
  --results-csv runs/results.csv \
  2>&1 | tee -a "$LOG"

RUN_ID=$(RUN_NAME="$RUN_NAME" "$PYTHON" - <<'PY'
import os
from mlflow.tracking import MlflowClient

client = MlflowClient("sqlite:///mlflow.db")
experiment = client.get_experiment_by_name("conditioned-iqa-performance")
if experiment is None:
    raise SystemExit("MLflow experiment not found")
runs = client.search_runs(
    [experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{os.environ['RUN_NAME']}'",
    order_by=["attributes.start_time DESC"],
)
if not runs:
    raise SystemExit("trained run not found")
print(runs[0].info.run_id)
PY
)
echo "[$(date -Is)] source run $RUN_ID; evaluating all held-out datasets" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}" "$PYTHON" evaluate_text_conditioned.py \
  --source-run-id "$RUN_ID" \
  --data \
    /home/sergey/conditioned-iqa/data/tid2013/labels.csv \
    /home/sergey/conditioned-iqa/data/csiq/labels.csv \
    /home/sergey/conditioned-iqa/data/cid2013/labels.csv \
    /home/sergey/conditioned-iqa/data/koniq10k/labels.csv \
    /home/sergey/conditioned-iqa/data/clive/labels.csv \
    /home/sergey/conditioned-iqa/data/agiqa3k/labels.csv \
    /home/sergey/conditioned-iqa/data/gfiqa20k/labels.csv \
    /home/sergey/conditioned-iqa/data/pipal/labels.csv \
    /home/sergey/conditioned-iqa/data/uhdiqa/labels.csv \
  --device cuda:0 \
  --batch-size 32 \
  --workers 4 \
  --mlflow-tracking-uri sqlite:///mlflow.db \
  --mlflow-experiment conditioned-iqa-external-eval \
  --results-csv runs/results.csv \
  2>&1 | tee -a "$LOG"

case "$RUN_NAME" in
  best-config-clip-large-clean-interaction-s0)
    DESIGN="CLIP-L/14@336 clean-mixture native interaction"
    DESCRIPTION="Current best clean-mixture protocol with frozen CLIP-L/14@336 vision and native text interaction; KADID-10k + SPAQ + AIGCIQA2023, stretch preprocessing, equal dataset sampling, seed 0."
    BACKBONE_LABEL="CLIP-L/14@336"
    ;;
  best-config-siglip-large-clean-interaction-s0)
    DESIGN="SigLIP-L/16@256 clean-mixture native interaction"
    DESCRIPTION="Current best clean-mixture protocol with frozen SigLIP-L/16@256 vision and native text interaction; KADID-10k + SPAQ + AIGCIQA2023, stretch preprocessing, equal dataset sampling, seed 0."
    BACKBONE_LABEL="SigLIP-L/16@256"
    ;;
  *)
    DESIGN=""
    ;;
esac
if [[ -n "$DESIGN" ]]; then
  "$PYTHON" scripts/append_run_to_sheet.py \
    --run-name "$RUN_NAME" --design "$DESIGN" --description "$DESCRIPTION" \
    --backbone "$BACKBONE_LABEL" --baseline variant 2>&1 | tee -a "$LOG"
fi

echo "[$(date -Is)] completed $RUN_NAME" | tee -a "$LOG"
