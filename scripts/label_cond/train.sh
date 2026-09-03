#!/usr/bin/env bash
set -e

export MLFLOW_ALLOW_FILE_STORE=true
export CUDA_VISIBLE_DEVICES=6

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

# run_batch \
#     configs/label_cond/01_hard.yaml \
#     configs/label_cond/02_frozen_hard.yaml \
#     configs/label_cond/03_frozen_soft.yaml \
#     configs/label_cond/04_joint_pretrained_hard.yaml


# run_batch \
#     configs/label_cond/00_zero_labels.yaml \
#     configs/label_cond/05_joint_pretrained_soft.yaml \
#     configs/label_cond/06_joint_untrained_hard.yaml \
#     configs/label_cond/07_joint_untrained_soft.yaml

# run_batch \
#     configs/label_cond/08_hard_emb.yaml \
#     configs/label_cond/09_hard_emb_deep.yaml \
#     configs/label_cond/10_joint_scratch_soft_emb.yaml \
#     configs/label_cond/11_joint_scratch_soft_emb_deep.yaml

# run_batch \
#     configs/label_cond/12_zero_emb_deep.yaml \
#     configs/label_cond/13_hard_emb_deep_add.yaml \
#     configs/label_cond/14_hard_emb_deep_film.yaml

# run_batch \
#     configs/label_cond/15_frozen_soft_emb_deep.yaml \
#     configs/label_cond/16_frozen_layer3_emb_deep.yaml \
#     configs/label_cond/17_frozen_layer4_emb_deep.yaml

# run_batch \
#     configs/label_cond/18_frozen_layer3_norm_deep.yaml \
#     configs/label_cond/19_frozen_layer4_norm_deep.yaml

run_batch \
    configs/label_cond/20_zero_matched_layer3_params.yaml
