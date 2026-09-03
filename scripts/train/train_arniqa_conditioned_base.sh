#!/usr/bin/env bash
set -euo pipefail

CACHE_REPO="${CACHE_REPO:-$HOME/.cache/huggingface/hub/models--openai--clip-vit-base-patch16}"
REVISION="$(cat "$CACHE_REPO/refs/main")"
DATA="${DATA:-$HOME/conditioned-iqa/data/kadid10k/labels.csv}"
DEVICE="${DEVICE:-cuda:7}"
SAVE_DIR="${SAVE_DIR:-./weights}"
NAME="${NAME:-arniqa_conditioned_clip_base}"

arniqa_args=()
if [[ -n "${ARNIQA_WEIGHTS:-}" ]]; then
  arniqa_args+=(--arniqa-weights "$ARNIQA_WEIGHTS")
elif [[ -f "$HOME/.cache/torch/hub/checkpoints/ARNIQA.pth" ]]; then
  arniqa_args+=(--arniqa-weights "$HOME/.cache/torch/hub/checkpoints/ARNIQA.pth")
fi

train_command=(
  python3 train.py
  --data "$DATA"
  --backbone clip-base
  --device "$DEVICE"
  --weights "$CACHE_REPO/snapshots/$REVISION"
  --conditioning arniqa
  --condition-dim 32
  --condition-dropout 0.1
  --arniqa-batch-size 64
  --save-dir "$SAVE_DIR"
  --name "$NAME"
)
"${train_command[@]}" "${arniqa_args[@]}" "$@"
