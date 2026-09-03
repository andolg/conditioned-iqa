#!/usr/bin/env bash
set -euo pipefail

CACHE_REPO="${CACHE_REPO:-$HOME/.cache/huggingface/hub/models--openai--clip-vit-base-patch16}"
REVISION="$(cat "$CACHE_REPO/refs/main")"
DATA="${DATA:-$HOME/conditioned-iqa/data/kadid10k/labels.csv}"
DEVICE="${DEVICE:-cuda:7}"
EPOCHS="${EPOCHS:-5}"
LOW_RANK_DIM="${LOW_RANK_DIM:-4}"

python3 train.py \
  --data "$DATA" \
  --backbone clip-base \
  --device "$DEVICE" \
  --weights "$CACHE_REPO/snapshots/$REVISION" \
  --epochs "$EPOCHS" \
  --conditioning label \
  --label-fusion low_rank_hypernetwork \
  --label-dim 32 \
  --low-rank-dim "$LOW_RANK_DIM" \
  --condition-dropout 0.1 \
  --save-dir ./weights \
  --name label_low_rank_hypernetwork_clip_base \
  "$@"
