#!/usr/bin/env bash
set -euo pipefail

CACHE_REPO="${CACHE_REPO:-$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336}"
REVISION="$(cat "$CACHE_REPO/refs/main")"
DATA="${DATA:-$HOME/conditioned-iqa/data/kadid10k/labels.csv}"
DEVICE="${DEVICE:-cuda:7}"
EPOCHS="${EPOCHS:-5}"

python3 train.py \
  --data "$DATA" \
  --backbone clip-large \
  --device "$DEVICE" \
  --weights "$CACHE_REPO/snapshots/$REVISION" \
  --epochs "$EPOCHS" \
  --conditioning label \
  --label-fusion patch_attention \
  --label-dim 32 \
  --condition-dropout 0.1 \
  --save-dir ./weights \
  --name label_patch_attention_clip_large \
  "$@"
