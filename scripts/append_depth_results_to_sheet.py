#!/usr/bin/env python3
"""Append the MLP-depth comparison below the existing 28s_mur table.

The script deliberately writes to the first rows after the last non-empty
design and never clears or overwrites unrelated rows.  Training-dataset
columns use validation metrics; every other dataset, including the full UHD
CSV, uses the external evaluation rows.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import gspread
from mlflow.tracking import MlflowClient

# Make direct execution (`python scripts/...`) resolve repository modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_results_summary import DATASETS, DATASET_NAMES, HEADERS


SHEET_ID = "1DZQKInig5PtctN23TXEkMzrdcc1uA9aATaxT3LTXYZY"
WORKSHEET = "28s_mur"
RESULTS = Path("runs/results.csv")
TRAIN_DATASETS = {"kadid10k", "spaq", "aigciqa2023"}

EXPERIMENTS = [
    {
        "run_name": "m1-mdtvsfa-interaction-depth2-s0",
        "source_run_id": "775655b7ab1742599d2cea3de88393c1",
        "design": "CLIP-B/16 calibrated interaction with 2-layer score MLP",
        "description": (
            "Native CLIP-B/16 image/text interaction [v, t, v*t] with two hidden "
            "256-wide GELU MLP layers; legacy MDTVSFA calibration, equal dataset "
            "weighting, KADID-10k + SPAQ + AIGCIQA2023, seed 0. Full UHD-IQA "
            "(6,073 images) is used as the zero-shot external benchmark."
        ),
        "baseline": "variant",
    },
    {
        "run_name": "m1-mdtvsfa-interaction-depth3-s0",
        "source_run_id": "fdfd9846486d413792767689aff9ce6c",
        "design": "CLIP-B/16 calibrated interaction with 3-layer score MLP",
        "description": (
            "Native CLIP-B/16 image/text interaction [v, t, v*t] with three hidden "
            "256-wide GELU MLP layers; legacy MDTVSFA calibration, equal dataset "
            "weighting, KADID-10k + SPAQ + AIGCIQA2023, seed 0. Full UHD-IQA "
            "(6,073 images) is used as the zero-shot external benchmark."
        ),
        "baseline": "variant",
    },
    {
        "run_name": "m1-mdtvsfa-baseline-depth2-matched-s0",
        "source_run_id": "0fe30549d268472fa0c1815196ce4b08",
        "design": "CLIP-B/16 matched-capacity image-only 2-layer MLP",
        "description": (
            "Image-only CLIP-B/16 pooled scorer with two hidden 476-wide GELU MLP "
            "layers; total head footprint matched to the 2-layer conditioned head "
            "within 0.02%, same MDTVSFA objective, data, split, and seed. Full "
            "UHD-IQA (6,073 images) is used as the zero-shot external benchmark."
        ),
        "baseline": "baseline",
    },
    {
        "run_name": "m1-mdtvsfa-baseline-depth3-matched-s0",
        "source_run_id": "cde7a4323a97405a96b72bfbe766a44c",
        "design": "CLIP-B/16 matched-capacity image-only 3-layer MLP",
        "description": (
            "Image-only CLIP-B/16 pooled scorer with three hidden 413-wide GELU "
            "MLP layers; total head footprint matched to the 3-layer conditioned "
            "head within 0.08%, same MDTVSFA objective, data, split, and seed. "
            "Full UHD-IQA (6,073 images) is used as the zero-shot external benchmark."
        ),
        "baseline": "baseline",
    },
]


def numeric(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def average(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def metric(rows: list[dict[str, str]], dataset: str, key: str) -> float | None:
    return average([numeric(row.get(key)) for row in rows if row["dataset"] == dataset])


def build_rows() -> list[list[object]]:
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    client = MlflowClient("sqlite:///mlflow.db")
    output: list[list[object]] = []
    for experiment in EXPERIMENTS:
        validation = [
            row for row in raw
            if row["run_name"] == experiment["run_name"] and row["evaluation"] == "validation"
        ]
        external = [
            row for row in raw
            if row["run_name"] == experiment["run_name"].replace("m1-mdtvsfa-", "external-m1-")
            and row["evaluation"] == "held_out_test"
        ]
        # The external names include "full-uhd" while the training names do not.
        if not external:
            external = [
                row for row in raw
                if row["run_name"].startswith(
                    experiment["run_name"].replace("m1-mdtvsfa-", "external-m1-") + "-full-uhd"
                ) and row["evaluation"] == "held_out_test"
            ]
        if len(validation) != 3 or len(external) != 9:
            raise RuntimeError(
                f"expected 3 validation and 9 full-external rows for {experiment['run_name']}; "
                f"found {len(validation)} and {len(external)}"
            )
        first = validation[0]
        dataset_values: dict[str, float | None] = {}
        for dataset in DATASETS:
            source = validation if dataset in TRAIN_DATASETS else external
            dataset_values[f"{DATASET_NAMES[dataset]} SRCC"] = metric(source, dataset, "srcc")
            dataset_values[f"{DATASET_NAMES[dataset]} PLCC"] = metric(source, dataset, "plcc")

        val_srcc = average([dataset_values[f"{DATASET_NAMES[d]} SRCC"] for d in TRAIN_DATASETS])
        val_plcc = average([dataset_values[f"{DATASET_NAMES[d]} PLCC"] for d in TRAIN_DATASETS])
        test_srcc = average([
            dataset_values[f"{DATASET_NAMES[d]} SRCC"]
            for d in DATASETS if d not in TRAIN_DATASETS
        ])
        test_plcc = average([
            dataset_values[f"{DATASET_NAMES[d]} PLCC"]
            for d in DATASETS if d not in TRAIN_DATASETS
        ])
        run = client.get_run(experiment["source_run_id"])
        gflops = numeric(run.data.metrics["system/flops"]) / 1e9
        model_mb = numeric(first["model_parameter_size_mb"])
        parameters = f"{model_mb * 1024**2 / 4 / 1e6:.2f}M"
        p50 = numeric(first["latency_p50_ms"])
        row: list[object] = [
            experiment["design"], experiment["description"], "CLIP-B/16",
            "KADID-10k, SPAQ, AIGCIQA2023", int(float(first["epochs"])), "0",
            experiment["baseline"], p50, numeric(first["latency_p95_ms"]),
            numeric(first["peak_memory_mb"]), 1000 / p50 if p50 else None,
        ]
        for dataset in DATASETS:
            row.extend((dataset_values[f"{DATASET_NAMES[dataset]} SRCC"], dataset_values[f"{DATASET_NAMES[dataset]} PLCC"]))
        row.extend((
            val_srcc, val_plcc, test_srcc, test_plcc,
            average([val_srcc, test_srcc]), average([val_plcc, test_plcc]),
            parameters, gflops,
        ))
        if len(row) != len(HEADERS):
            raise RuntimeError(f"constructed {len(row)} cells, expected {len(HEADERS)}")
        output.append(row)
    return output


def main() -> None:
    credential = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "/home/sergey/.config/conditioned-iqa/google-service-account.json"
    worksheet = gspread.service_account(filename=credential).open_by_key(SHEET_ID).worksheet(WORKSHEET)
    existing = worksheet.get_all_values(value_render_option="UNFORMATTED_VALUE")
    if not existing or existing[0][: len(HEADERS)] != HEADERS:
        raise RuntimeError("worksheet header does not match the experiment table")
    rows = build_rows()
    existing_designs = {
        str(row[0]).strip(): index + 1
        for index, row in enumerate(existing[1:], start=1)
        if row and str(row[0]).strip()
    }
    last_nonempty = max(
        (index + 1 for index, row in enumerate(existing) if row and str(row[0]).strip()),
        default=1,
    )
    next_row = last_nonempty + 1
    for offset, (experiment, values) in enumerate(zip(EXPERIMENTS, rows)):
        target = existing_designs.get(experiment["design"], next_row + offset)
        end = gspread.utils.rowcol_to_a1(target, len(HEADERS))
        worksheet.update([values], f"A{target}:{end}", value_input_option="RAW")
        print(f"wrote {experiment['design']} at row {target}")


if __name__ == "__main__":
    main()
