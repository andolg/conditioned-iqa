#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/conditioned-iqa/data}"
DEVICE="${DEVICE:-cuda:7}"
REGRESSOR_DATASET="${REGRESSOR_DATASET:-kadid10k}"

datasets=(
  agiqa3k
  aigciqa2023
  cid2013
  clive
  csiq
  gfiqa20k
  kadid10k
  koniq10k
  pipal
  spaq
  tid2013
  uhdiqa
)

csvs=()
for dataset in "${datasets[@]}"; do
  csv="$DATA_ROOT/$dataset/labels.csv"
  if [[ -f "$csv" ]]; then
    csvs+=("$csv")
  else
    echo "warning: skipping missing $csv" >&2
  fi
done

if (( ${#csvs[@]} == 0 )); then
  echo "error: no labels.csv files found below $DATA_ROOT" >&2
  exit 1
fi

arniqa_args=()
if [[ -n "${ARNIQA_WEIGHTS:-}" ]]; then
  arniqa_args+=(--arniqa-weights "$ARNIQA_WEIGHTS")
elif [[ -f "$HOME/.cache/torch/hub/checkpoints/ARNIQA.pth" ]]; then
  arniqa_args+=(--arniqa-weights "$HOME/.cache/torch/hub/checkpoints/ARNIQA.pth")
fi

regressor_args=()
if [[ -n "${ARNIQA_REGRESSOR_WEIGHTS:-}" ]]; then
  regressor_args+=(--regressor-weights "$ARNIQA_REGRESSOR_WEIGHTS")
elif [[ -f "$HOME/.cache/torch/hub/checkpoints/regressor_$REGRESSOR_DATASET.pth" ]]; then
  regressor_args+=(
    --regressor-weights
    "$HOME/.cache/torch/hub/checkpoints/regressor_$REGRESSOR_DATASET.pth"
  )
fi

evaluate_command=(
  python3 ./evaluate_arniqa.py
  --data "${csvs[@]}"
  --regressor-dataset "$REGRESSOR_DATASET"
  --device "$DEVICE"
  --arniqa-batch-size 64
)
"${evaluate_command[@]}" "${arniqa_args[@]}" "${regressor_args[@]}" "$@"
