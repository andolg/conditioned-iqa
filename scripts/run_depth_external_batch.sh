#!/usr/bin/env bash
set -euo pipefail

# Persistent, single-GPU continuation for the MLP-depth experiment.
# Training checkpoints already exist; this script only evaluates the three
# checkpoints not covered by the foreground depth-2 evaluation.
cd /home/sergey/conditioned-iqa/28s_mur/conditioned-iqa

exec 9>/tmp/conditioned-iqa-depth-gpu4.lock
flock -n 9 || {
  echo "another depth evaluation already owns GPU 4" >&2
  exit 2
}

PYTHON=./.venv/bin/python
DEVICE=cuda:4
BATCH_SIZE=64
WORKERS=4
DATA=(
  /home/sergey/conditioned-iqa/data/agiqa3k/labels.csv
  /home/sergey/conditioned-iqa/data/cid2013/labels.csv
  /home/sergey/conditioned-iqa/data/clive/labels.csv
  /home/sergey/conditioned-iqa/data/csiq/labels.csv
  /home/sergey/conditioned-iqa/data/gfiqa20k/labels.csv
  /home/sergey/conditioned-iqa/data/koniq10k/labels.csv
  /home/sergey/conditioned-iqa/data/pipal/labels.csv
  /home/sergey/conditioned-iqa/data/tid2013/labels.csv
  /home/sergey/conditioned-iqa/data/uhdiqa/labels.csv
)

run_eval() {
  local source_run="$1"
  local run_name="$2"
  "$PYTHON" evaluate_text_conditioned.py \
    --source-run-id "$source_run" \
    --data "${DATA[@]}" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --mlflow-run-name "$run_name"
}

run_eval fdfd9846486d413792767689aff9ce6c \
  external-m1-interaction-depth3-s0-full-uhd
run_eval 0fe30549d268472fa0c1815196ce4b08 \
  external-m1-baseline-depth2-matched-s0-full-uhd
run_eval cde7a4323a97405a96b72bfbe766a44c \
  external-m1-baseline-depth3-matched-s0-full-uhd

echo "depth external evaluation batch completed"
