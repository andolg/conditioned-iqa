#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/conditioned-iqa/data}"
CHECKPOINT="${CHECKPOINT:-./weights/label_conditioned_clip_large_best.pth}"
DEVICE="${DEVICE:-cuda:7}"
CACHE_REPO="${CACHE_REPO:-$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336}"
REVISION="$(cat "$CACHE_REPO/refs/main")"

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

python3 ./evaluate.py \
  --data "${csvs[@]}" \
  --checkpoint "$CHECKPOINT" \
  --weights "$CACHE_REPO/snapshots/$REVISION" \
  --device "$DEVICE" \
  "$@"
