#!/usr/bin/env bash
set -e

export MLFLOW_ALLOW_FILE_STORE=true

cd "$(dirname "$0")/../.."
uv run python -m models.dist_classifier.train --config configs/dist_classifier.yaml
