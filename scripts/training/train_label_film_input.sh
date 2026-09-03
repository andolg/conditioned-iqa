#!/usr/bin/env bash
set -euo pipefail

DATA="${DATA:-$HOME/conditioned-iqa/data/kadid10k/labels.csv}"
BACKBONE="${BACKBONE:-clip-base}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LABEL_DIM="${LABEL_DIM:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
CONDITION_DROPOUT="${CONDITION_DROPOUT:-0.1}"
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
SAVE_DIR="${SAVE_DIR:-./weights}"

python train.py \
  --data "$DATA" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --conditioning label \
  --label-fusion film_input \
  --label-dim "$LABEL_DIM" \
  --condition-dropout "$CONDITION_DROPOUT" \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --save-dir "$SAVE_DIR" \
  --name label_film_input_${BACKBONE}
