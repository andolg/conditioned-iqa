#!/usr/bin/env bash
set -e

export MLFLOW_ALLOW_FILE_STORE=true

cd "$(dirname "$0")/.."
uv run mlflow server --port "$1"
