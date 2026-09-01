"""Fill missing held-out-test and KADID-throughput rows for trained checkpoints.

This is a small, re-runnable utility.  It reads ``runs/results.csv``, computes
which source runs are missing test-dataset correlations or a KADID-10k
throughput row, and runs the existing evaluation/benchmark entry points on the
requested GPUs.  Existing rows are left untouched, so re-running after a
successful backfill is a no-op.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parent
RESULTS = REPO / "runs" / "results.csv"
DATA = Path("/home/sergey/conditioned-iqa/data")
TEST_DATASETS = ("tid2013", "csiq", "cid2013", "koniq10k", "clive", "agiqa3k", "uhdiqa")


def source_groups() -> dict[str, dict]:
    groups: dict[str, dict] = defaultdict(lambda: {
        "backbone": "",
        "method": "",
        "epochs": "",
        "test": set(),
        "kadid_throughput": False,
    })
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        name = str(row.get("run_name", "")).lower()
        if "smoke" in name:
            continue
        if "heldout" in name:
            continue
        source = row.get("source_run_id") or row.get("run_id")
        if not source:
            continue
        group = groups[source]
        group["backbone"] = row.get("backbone") or group["backbone"]
        group["method"] = row.get("method") or group["method"]
        group["epochs"] = row.get("epochs") or group["epochs"]
        if row.get("evaluation") == "held_out_test" and row.get("dataset") in TEST_DATASETS:
            group["test"].add(row["dataset"])
        if (
            row.get("evaluation") == "held_out_test"
            and row.get("dataset") == "kadid10k"
            and row.get("images_per_second")
        ):
            group["kadid_throughput"] = True
    return groups


def build_tasks(groups: dict[str, dict]) -> list[dict]:
    tasks: list[dict] = []
    for source, group in sorted(groups.items()):
        missing_test = [dataset for dataset in TEST_DATASETS if dataset not in group["test"]]
        if missing_test:
            tasks.append({
                "type": "test",
                "source": source,
                "backbone": group["backbone"],
                "epochs": group["epochs"],
                "datasets": missing_test,
            })
        if not group["kadid_throughput"]:
            tasks.append({
                "type": "benchmark",
                "source": source,
                "backbone": group["backbone"],
                "epochs": group["epochs"],
                "datasets": [],
            })
    return tasks


def batch_size(backbone: str) -> int:
    return 32 if backbone.startswith("clip-large") else 64


def command_for(task: dict) -> list[str]:
    source = task["source"]
    epochs = task["epochs"] or "5"
    if task["type"] == "test":
        data_paths = [str(DATA / dataset / "labels.csv") for dataset in task["datasets"]]
        return [
            "uv", "run", "python", "evaluate_text_conditioned.py",
            "--source-run-id", source,
            "--device", "cuda:0",
            "--batch-size", str(batch_size(task["backbone"])),
            "--workers", "6",
            "--epochs", epochs,
            "--data", *data_paths,
        ]
    return [
        "uv", "run", "python", "benchmark_kadid_throughput.py",
        "--source-run-id", source,
        "--data", str(DATA / "kadid10k" / "labels.csv"),
        "--device", "cuda:0",
        "--batch-size", str(batch_size(task["backbone"])),
        "--workers", "6",
        "--epochs", epochs,
    ]


def run_on_gpu(gpu: str, tasks: list[dict], log_dir: Path, failures: list[str]) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for task in tasks:
        source = task["source"]
        name = f"{task['type']}-{source}"
        log_path = log_dir / f"{name}.log"
        cmd = command_for(task)
        print(f"[gpu {gpu}] start {name}: {' '.join(cmd)}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode != 0:
            failures.append(f"gpu {gpu} {name}")
            print(f"[gpu {gpu}] FAILED {name} (see {log_path})", flush=True)
        else:
            print(f"[gpu {gpu}] ok {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,2,6", help="comma-separated physical GPU indices")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    groups = source_groups()
    tasks = build_tasks(groups)
    if args.dry_run:
        for task in tasks:
            print(" ".join(command_for(task)))
        print(f"\n{len(tasks)} tasks")
        return

    gpus = [part.strip() for part in args.gpus.split(",") if part.strip()]
    log_dir = REPO / "runs" / "backfill_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    threads: list[threading.Thread] = []
    for offset, gpu in enumerate(gpus):
        assigned = tasks[offset::len(gpus)]
        thread = threading.Thread(target=run_on_gpu, args=(gpu, assigned, log_dir, failures))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    print("\nDone.", flush=True)
    if failures:
        print("Failures:", *failures, sep="\n  ", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
