#!/usr/bin/env bash
set -e

export MLFLOW_ALLOW_FILE_STORE=true
export CUDA_VISIBLE_DEVICES=1

cd "$(dirname "$0")/../.."

base=configs/label_cond/label_cond.yaml

run() {
    echo "==> $1"
    uv run python -m models.label_cond.train --config "$base" "$1"
}

run_batch() {
    pids=()
    for patch in "$@"; do
        run "$patch" &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    return "$status"
}



run_batch \
    configs/label_cond/01_hard.yaml \
    configs/label_cond/02_frozen_hard.yaml \
    configs/label_cond/03_frozen_soft.yaml \
    configs/label_cond/04_joint_pretrained_hard.yaml



    # configs/label_cond/00_zero_labels.yaml \
run_batch \
    configs/label_cond/05_joint_pretrained_soft.yaml \
    configs/label_cond/06_joint_untrained_hard.yaml \
    configs/label_cond/07_joint_untrained_soft.yaml
