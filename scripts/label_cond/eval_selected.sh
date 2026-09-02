#!/usr/bin/env bash
set -euo pipefail

export MLFLOW_ALLOW_FILE_STORE=true
export CUDA_VISIBLE_DEVICES=4

cd "$(dirname "$0")/../.."

run_configs=("$@")
if (( ${#run_configs[@]} == 0 )); then
    run_configs=(
        configs/label_cond/18_frozen_layer3_norm_deep.yaml
        configs/label_cond/19_frozen_layer4_norm_deep.yaml
    )
fi

args=()
for run_config in "${run_configs[@]}"; do
    args+=(--run-config "$run_config")
done

uv run python -m models.label_cond.eval \
    --config configs/label_cond/eval.yaml \
    --table-root tables/selected \
    "${args[@]}"
