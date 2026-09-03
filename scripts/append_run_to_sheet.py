#!/usr/bin/env python3
"""Append one completed run aggregate to the formatted ``28s_mur`` table.

The source CSV is long-form (one dataset per row); this converts one training
run plus its held-out evaluations into the wide table row.  It writes below
the last non-empty design, copies the preceding row's formatting, and never
clears or rewrites unrelated designs.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import sys
from pathlib import Path

import gspread
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_results_summary import DATASETS, DATASET_NAMES, HEADERS

SHEET_ID = "1DZQKInig5PtctN23TXEkMzrdcc1uA9aATaxT3LTXYZY"
WORKSHEET = "28s_mur"
TRAIN_DATASETS = {"kadid10k", "spaq", "aigciqa2023", "gfiqa20k", "pipal"}


def num(value):
    if value in (None, "", "nan", "NaN"):
        return None
    return float(value)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def metric(rows, dataset, field):
    return mean([num(row.get(field)) for row in rows if row.get("dataset") == dataset])


def copy_format(worksheet, source_row: int, target_row: int) -> None:
    if source_row == target_row:
        return
    worksheet.spreadsheet.batch_update({"requests": [{"copyPaste": {
        "source": {"sheetId": worksheet.id, "startRowIndex": source_row - 1,
                    "endRowIndex": source_row, "startColumnIndex": 0,
                    "endColumnIndex": len(HEADERS)},
        "destination": {"sheetId": worksheet.id, "startRowIndex": target_row - 1,
                         "endRowIndex": target_row, "startColumnIndex": 0,
                         "endColumnIndex": len(HEADERS)},
        "pasteType": "PASTE_FORMAT",
    }}]})


def build_row(raw, client, run_name: str, design: str, description: str, backbone: str, baseline: str):
    validation = [row for row in raw if row.get("run_name") == run_name and row.get("evaluation") == "validation"]
    if not validation:
        raise RuntimeError(f"no validation rows found for {run_name}")
    source_ids = {row.get("run_id") for row in validation if row.get("run_id")}
    if len(source_ids) != 1:
        raise RuntimeError(f"expected one source run for {run_name}, found {sorted(source_ids)}")
    source_id = next(iter(source_ids))
    external = [row for row in raw if row.get("evaluation") == "held_out_test" and row.get("source_run_id") == source_id]
    test_datasets = set(DATASETS) - {row.get("dataset") for row in validation}
    missing = sorted(test_datasets - {row.get("dataset") for row in external})
    if missing:
        raise RuntimeError(f"missing external datasets for {run_name}: {missing}")
    first = validation[0]
    values = {}
    train_set = {row.get("dataset") for row in validation}
    for dataset in DATASETS:
        source = validation if dataset in train_set else external
        values[f"{DATASET_NAMES[dataset]} SRCC"] = metric(source, dataset, "srcc")
        values[f"{DATASET_NAMES[dataset]} PLCC"] = metric(source, dataset, "plcc")
    val_srcc = mean([values[f"{DATASET_NAMES[d]} SRCC"] for d in train_set])
    val_plcc = mean([values[f"{DATASET_NAMES[d]} PLCC"] for d in train_set])
    held_out = [d for d in DATASETS if d not in train_set]
    test_srcc = mean([values[f"{DATASET_NAMES[d]} SRCC"] for d in held_out])
    test_plcc = mean([values[f"{DATASET_NAMES[d]} PLCC"] for d in held_out])
    run = client.get_run(source_id)
    flops = num(run.data.metrics.get("system/flops"))
    model_mb = num(first.get("model_parameter_size_mb"))
    p50 = mean([num(row.get("latency_p50_ms")) for row in validation])
    p95 = mean([num(row.get("latency_p95_ms")) for row in validation])
    memory = max([num(row.get("peak_memory_mb")) for row in validation if num(row.get("peak_memory_mb")) is not None], default=None)
    row = [
        design, description, backbone,
        ", ".join(DATASET_NAMES[d] for d in DATASETS if d in train_set),
        int(float(first.get("epochs", 0))), "0", baseline,
        p50, p95, memory, (1000 / p50 if p50 else None),
    ]
    for dataset in DATASETS:
        row.extend((values[f"{DATASET_NAMES[dataset]} SRCC"], values[f"{DATASET_NAMES[dataset]} PLCC"]))
    row.extend((val_srcc, val_plcc, test_srcc, test_plcc,
                mean([val_srcc, test_srcc]), mean([val_plcc, test_plcc]),
                f"{model_mb * 1024**2 / 4 / 1e6:.2f}M" if model_mb is not None else "",
                flops / 1e9 if flops is not None else None))
    if len(row) != len(HEADERS):
        raise RuntimeError(f"constructed {len(row)} cells, expected {len(HEADERS)}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--baseline", default="variant")
    parser.add_argument("--results", type=Path, default=Path("runs/results.csv"))
    args = parser.parse_args()
    with args.results.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    client = MlflowClient("sqlite:///mlflow.db")
    row = build_row(raw, client, args.run_name, args.design, args.description, args.backbone, args.baseline)
    credential = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "/home/sergey/.config/conditioned-iqa/google-service-account.json"
    # The exporter is called by independent persistent jobs.  Hold a local
    # lock across the sheet read, target-row selection, format copy, and
    # update so concurrent jobs cannot select and overwrite the same row.
    lock_path = Path("runs/.sheet_export.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            worksheet = gspread.service_account(filename=credential).open_by_key(SHEET_ID).worksheet(WORKSHEET)
            existing = worksheet.get_all_values(value_render_option="UNFORMATTED_VALUE")
            if not existing or len(existing[0]) < len(HEADERS) or existing[0][1:len(HEADERS)] != HEADERS[1:]:
                raise RuntimeError("worksheet header does not match the experiment table")
            existing_designs = {str(item[0]).strip(): index for index, item in enumerate(existing[1:], start=2) if item and str(item[0]).strip()}
            last_nonempty = max((index for index, item in enumerate(existing, start=1) if item and str(item[0]).strip()), default=1)
            target = existing_designs.get(args.design, last_nonempty + 1)
            if target == last_nonempty + 1:
                copy_format(worksheet, last_nonempty, target)
            end = gspread.utils.rowcol_to_a1(target, len(HEADERS))
            worksheet.update([row], f"A{target}:{end}", value_input_option="RAW")
            print(f"wrote {args.design} at row {target}")
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
