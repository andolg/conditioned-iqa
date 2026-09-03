#!/usr/bin/env bash
set -euo pipefail

# Persistent D1 batch. CUDA_VISIBLE_DEVICES maps the selected physical GPU to
# cuda:0, which is what the D1 YAML files request.  The lock prevents a second
# copy of this batch from competing for the same device.
cd /home/sergey/conditioned-iqa/28s_mur/conditioned-iqa
exec 9>/tmp/conditioned-iqa-gfiqa-d1-gpu3.lock
flock -n 9 || {
  echo "another GFIQA D1 batch already owns physical GPU 3" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=3
PYTHON=./.venv/bin/python
EXTERNAL_DATA=(
  /home/sergey/conditioned-iqa/data/tid2013/labels.csv
  /home/sergey/conditioned-iqa/data/csiq/labels.csv
  /home/sergey/conditioned-iqa/data/cid2013/labels.csv
  /home/sergey/conditioned-iqa/data/koniq10k/labels.csv
  /home/sergey/conditioned-iqa/data/clive/labels.csv
  /home/sergey/conditioned-iqa/data/agiqa3k/labels.csv
  /home/sergey/conditioned-iqa/data/pipal/labels.csv
  /home/sergey/conditioned-iqa/data/uhdiqa/labels.csv
)

run_one() {
  local config="$1"
  local stem="$2"
  local seed="$3"
  local source_run_id
  source_run_id=$("$PYTHON" - "$stem" "$seed" <<'PY'
import sys
from mlflow.tracking import MlflowClient

stem, seed = sys.argv[1:]
name = f"{stem}-s{seed}"
client = MlflowClient("sqlite:///mlflow.db")
experiment = client.get_experiment_by_name("conditioned-iqa-gfiqa-screened")
if experiment is None:
    raise SystemExit(0)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{name}'",
    order_by=["attributes.start_time DESC"],
)
finished = [run for run in runs if run.info.status == "FINISHED"]
if finished:
    print(finished[0].info.run_id)
PY
  ) || true

  if [[ -z "$source_run_id" ]]; then
    "$PYTHON" train_text_conditioned.py \
      --config "$config" \
      --seed "$seed" \
      --device cuda:0 \
      --mlflow-run-name "${stem}-s${seed}"
    source_run_id=$("$PYTHON" - "$stem" "$seed" <<'PY'
import sys
from mlflow.tracking import MlflowClient

stem, seed = sys.argv[1:]
name = f"{stem}-s{seed}"
client = MlflowClient("sqlite:///mlflow.db")
experiment = client.get_experiment_by_name("conditioned-iqa-gfiqa-screened")
if experiment is None:
    raise SystemExit(f"experiment conditioned-iqa-gfiqa-screened is missing while looking for {name}")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{name}'",
    order_by=["attributes.start_time DESC"],
)
finished = [run for run in runs if run.info.status == "FINISHED"]
if len(finished) != 1:
    raise SystemExit(f"expected exactly one finished training run named {name}, found {len(finished)}")
print(finished[0].info.run_id)
PY
    )
  else
    echo "reusing finished training run ${stem}-s${seed}: ${source_run_id}"
  fi

  if ! "$PYTHON" - "$stem" "$seed" <<'PY'
import csv
import sys

stem, seed = sys.argv[1:]
name = f"external-{stem}-s{seed}"
expected = {"tid2013", "csiq", "cid2013", "koniq10k", "clive", "agiqa3k", "pipal", "uhdiqa"}
with open("runs/results.csv", newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
found = {row["dataset"] for row in rows if row.get("run_name") == name and row.get("evaluation") == "held_out_test"}
raise SystemExit(0 if expected.issubset(found) else 1)
PY
  then
    "$PYTHON" evaluate_text_conditioned.py \
      --source-run-id "$source_run_id" \
      --data "${EXTERNAL_DATA[@]}" \
      --device cuda:0 \
      --batch-size 64 \
      --workers 2 \
      --mlflow-run-name "external-${stem}-s${seed}"
  else
    echo "reusing completed external evaluation external-${stem}-s${seed}"
  fi

  # This is idempotent and writes only after all three seeds for a design are
  # present; calling it after every seed makes the batch safe to resume.
  "$PYTHON" scripts/append_gfiqa_screened_results_to_sheet.py
}

for seed in 0 1 2; do
  run_one configs/text_conditioning/53_gfiqa_screened_mdtvsfa_interaction_depth2.yaml \
    gfiqa-screened-m1-interaction-depth2 "$seed"
done
for seed in 0 1 2; do
  run_one configs/text_conditioning/54_gfiqa_screened_mdtvsfa_baseline_depth2_matched.yaml \
    gfiqa-screened-m1-baseline-depth2-matched "$seed"
done

echo "screened GFIQA D1 batch completed"
