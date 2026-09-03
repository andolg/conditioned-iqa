#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-./weights/arniqa_conditioned_clip_large_best.pth}"

arniqa_args=()
if [[ -n "${ARNIQA_WEIGHTS:-}" ]]; then
  arniqa_args+=(--arniqa-weights "$ARNIQA_WEIGHTS")
elif [[ -f "$HOME/.cache/torch/hub/checkpoints/ARNIQA.pth" ]]; then
  arniqa_args+=(--arniqa-weights "$HOME/.cache/torch/hub/checkpoints/ARNIQA.pth")
fi

evaluate_command=(
  bash ./scripts/test/evaluate_all_large.sh
  --arniqa-batch-size 64
)
CHECKPOINT="$CHECKPOINT" "${evaluate_command[@]}" "${arniqa_args[@]}" "$@"
