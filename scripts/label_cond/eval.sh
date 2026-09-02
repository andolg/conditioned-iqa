#!/usr/bin/env bash
set -e

export MLFLOW_ALLOW_FILE_STORE=true
export CUDA_VISIBLE_DEVICES=1

cd "$(dirname "$0")/../.."
uv run python -m models.label_cond.eval --config configs/label_cond/eval.yaml
