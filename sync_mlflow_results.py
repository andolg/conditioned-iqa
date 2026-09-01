"""Backfill result rows from completed MLflow runs into the shared results table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from result_reporting import ResultReporter, add_reporting_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", action="append", default=[],
                        help="MLflow experiment to sync; repeatable, default is all")
    parser.add_argument("--force", action="store_true", help="also append rows already present in the local CSV")
    add_reporting_arguments(parser)
    return parser


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {(row["run_id"], row["dataset"], row["evaluation"]) for row in csv.DictReader(stream)}


def rows_for_run(run, experiment_name: str) -> list[dict]:
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags
    method = tags.get("conditioning/method", params.get("method", "baseline"))
    common = {
        "run_id": run.info.run_id,
        "experiment": experiment_name,
        "run_name": tags.get("mlflow.runName", ""),
        "source_run_id": tags.get("source_run_id", ""),
        "backbone": params.get("backbone", ""),
        "method": method,
        "seed": params.get("seed", ""),
        "images_per_second": metrics.get("system/validation_images_per_second", ""),
        "head_size_mb": metrics.get("system/head_size_mb", ""),
        "model_parameter_size_mb": metrics.get("system/model_parameter_size_mb", ""),
        "config_path": f"mlflow:{run.info.run_id}",
    }
    rows = []
    for key, srcc in metrics.items():
        if key.startswith("validation/") and key.endswith("/srcc"):
            dataset = key.removeprefix("validation/").removesuffix("/srcc")
            rows.append({
                **common,
                "evaluation": "validation",
                "dataset": dataset,
                "srcc": srcc,
                "plcc": metrics.get(f"validation/{dataset}/plcc", ""),
            })
        if key.startswith("test/") and key.endswith("/srcc"):
            dataset = key.removeprefix("test/").removesuffix("/srcc")
            rows.append({
                **common,
                "evaluation": "held_out_test",
                "dataset": dataset,
                "srcc": srcc,
                "plcc": metrics.get(f"test/{dataset}/plcc", ""),
                "images_per_second": metrics.get(f"system/{dataset}/images_per_second", ""),
            })
    return rows


def main() -> None:
    args = build_parser().parse_args()
    from mlflow.tracking import MlflowClient

    reporter = ResultReporter.from_args(args)
    seen = existing_keys(reporter.results_csv)
    client = MlflowClient(args.mlflow_tracking_uri)
    experiments = client.search_experiments()
    if args.experiment:
        wanted = set(args.experiment)
        experiments = [experiment for experiment in experiments if experiment.name in wanted]
    rows = []
    for experiment in experiments:
        for run in client.search_runs([experiment.experiment_id], max_results=50_000):
            if run.info.status != "FINISHED":
                continue
            for row in rows_for_run(run, experiment.name):
                key = (row["run_id"], row["dataset"], row["evaluation"])
                if args.force or key not in seen:
                    rows.append(row)
    reporter.append(rows)
    print(f"appended {len(rows)} result rows to {reporter.results_csv}")


if __name__ == "__main__":
    main()
