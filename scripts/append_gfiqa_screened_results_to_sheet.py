#!/usr/bin/env python3
"""Append verified screened-GFIQA D1 aggregates to the experiment sheet.

The script is intentionally idempotent: it updates a row with the same design
label if one already exists, otherwise it appends immediately below the last
non-empty design.  It reads the sheet again before each write and copies the
previous data row's formatting into a newly appended row, so blank formatted
rows and colleague additions are not overwritten.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import gspread
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_results_summary import DATASETS, DATASET_NAMES, HEADERS


SHEET_ID = "1DZQKInig5PtctN23TXEkMzrdcc1uA9aATaxT3LTXYZY"
WORKSHEET = "28s_mur"
RESULTS = Path("runs/results.csv")
TRAIN_DATASETS = {"kadid10k", "spaq", "gfiqa20k", "aigciqa2023"}
EXTERNAL_DATASETS = [dataset for dataset in DATASETS if dataset not in TRAIN_DATASETS]

EXPERIMENTS = [
    {
        "design": "CLIP-B/16 screened-GFIQA calibrated interaction depth2",
        "description": (
            "Native CLIP-B/16 image/text interaction [v, t, v*t] with two hidden "
            "256-wide GELU MLP layers and four dataset calibrators; trained on "
            "KADID-10k + SPAQ + KonIQ-screened GFIQA-20k + AIGCIQA2023, equal "
            "dataset weighting, seeds 0-2. 235 GFIQA rows sharing KonIQ source "
            "IDs were excluded before splitting."
        ),
        "stem": "gfiqa-screened-m1-interaction-depth2",
        "baseline": "variant",
        "config": "configs/text_conditioning/53_gfiqa_screened_mdtvsfa_interaction_depth2.yaml",
    },
    {
        "design": "CLIP-B/16 screened-GFIQA matched-capacity image-only depth2",
        "description": (
            "Image-only CLIP-B/16 pooled scorer with two hidden 476-wide GELU MLP "
            "layers, parameter-matched to the conditioned head; same four dataset "
            "calibrators, screened GFIQA training table, split, weighting, and "
            "seeds 0-2 as the conditioned D1 run."
        ),
        "stem": "gfiqa-screened-m1-baseline-depth2-matched",
        "baseline": "baseline",
        "config": "configs/text_conditioning/54_gfiqa_screened_mdtvsfa_baseline_depth2_matched.yaml",
    },
]


def number(value: str | float | int | None) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    return float(value)


def mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def metric(rows: list[dict[str, str]], dataset: str, field: str) -> float | None:
    return mean([number(row.get(field)) for row in rows if row.get("dataset") == dataset])


def load_rows() -> list[dict[str, str]]:
    with RESULTS.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def latest_run_ids(rows: list[dict[str, str]], client: MlflowClient) -> set[str]:
    """Keep only the latest completed training run for each seed.

    A disconnected wrapper can leave a finished MLflow run before its
    external evaluation is written.  A supervisor may then retry the same
    seed, producing two rows with the same run name.  Aggregating both would
    silently double-count that seed, so select by MLflow start time.
    """
    by_seed: dict[str, set[str]] = {}
    for row in rows:
        run_id = row.get("run_id", "")
        seed = row.get("seed", "")
        if run_id and seed != "":
            by_seed.setdefault(seed, set()).add(run_id)
    selected: set[str] = set()
    for run_ids in by_seed.values():
        runs = [client.get_run(run_id) for run_id in run_ids]
        finished = [run for run in runs if run.info.status == "FINISHED"]
        if finished:
            selected.add(max(finished, key=lambda run: run.info.start_time).info.run_id)
    return selected


def build_row(experiment: dict[str, str], raw: list[dict[str, str]], client: MlflowClient) -> list[object] | None:
    validation_all = [
        row for row in raw
        if row.get("run_name", "").startswith(experiment["stem"] + "-s")
        and row.get("evaluation") == "validation"
    ]
    validation_run_ids = latest_run_ids(validation_all, client)
    validation = [row for row in validation_all if row.get("run_id") in validation_run_ids]
    external_all = [
        row for row in raw
        if row.get("run_name", "").startswith("external-" + experiment["stem"] + "-s")
        and row.get("evaluation") == "held_out_test"
    ]
    # External rows carry source_run_id, which points to the selected training
    # run.  If a seed was retried, retain only rows tied to the latest run.
    selected_source_ids = validation_run_ids
    external = [row for row in external_all if row.get("source_run_id") in selected_source_ids]
    seeds = sorted({row.get("seed", "") for row in validation if row.get("seed", "") != ""})
    if len(seeds) < 3:
        print(f"{experiment['design']}: waiting for 3 validation seeds; found {seeds}")
        return None
    expected_external = len(EXTERNAL_DATASETS) * len(seeds)
    if len(external) < expected_external:
        print(
            f"{experiment['design']}: waiting for {expected_external} external rows; "
            f"found {len(external)}"
        )
        return None

    values: dict[str, float | None] = {}
    for dataset in DATASETS:
        source = validation if dataset in TRAIN_DATASETS else external
        values[f"{DATASET_NAMES[dataset]} SRCC"] = metric(source, dataset, "srcc")
        values[f"{DATASET_NAMES[dataset]} PLCC"] = metric(source, dataset, "plcc")
    val_srcc = mean([values[f"{DATASET_NAMES[d]} SRCC"] for d in TRAIN_DATASETS])
    val_plcc = mean([values[f"{DATASET_NAMES[d]} PLCC"] for d in TRAIN_DATASETS])
    test_srcc = mean([values[f"{DATASET_NAMES[d]} SRCC"] for d in EXTERNAL_DATASETS])
    test_plcc = mean([values[f"{DATASET_NAMES[d]} PLCC"] for d in EXTERNAL_DATASETS])

    first = sorted(validation, key=lambda row: (row.get("seed", ""), row.get("dataset", "")))[0]
    run_id = first.get("run_id", "")
    run = client.get_run(run_id)
    flops = number(run.data.metrics.get("system/flops"))
    model_mb = number(first.get("model_parameter_size_mb"))
    # The existing table reports total trainable+frozen model footprint in the
    # same 4-byte conversion used by the other append scripts.
    parameters = f"{model_mb * 1024**2 / 4 / 1e6:.2f}M" if model_mb is not None else ""
    p50 = mean([number(row.get("latency_p50_ms")) for row in validation])
    p95 = mean([number(row.get("latency_p95_ms")) for row in validation])
    memory = mean([number(row.get("peak_memory_mb")) for row in validation])
    row: list[object] = [
        experiment["design"], experiment["description"], "CLIP-B/16",
        "KADID-10k, SPAQ, screened GFIQA-20k, AIGCIQA2023", 5, "0-2",
        experiment["baseline"], p50, p95, memory, 1000 / p50 if p50 else None,
    ]
    for dataset in DATASETS:
        row.extend((values[f"{DATASET_NAMES[dataset]} SRCC"], values[f"{DATASET_NAMES[dataset]} PLCC"]))
    row.extend((
        val_srcc, val_plcc, test_srcc, test_plcc,
        mean([val_srcc, test_srcc]), mean([val_plcc, test_plcc]),
        parameters, flops / 1e9 if flops is not None else None,
    ))
    if len(row) != len(HEADERS):
        raise RuntimeError(f"constructed {len(row)} cells, expected {len(HEADERS)}")
    return row


def find_target(existing: list[list[str]], design: str) -> tuple[int, int | None]:
    existing_designs = {
        str(row[0]).strip(): index + 1
        for index, row in enumerate(existing[1:], start=1)
        if row and str(row[0]).strip()
    }
    target = existing_designs.get(design)
    last_nonempty = max(
        (index + 1 for index, row in enumerate(existing) if row and str(row[0]).strip()),
        default=1,
    )
    return target or last_nonempty + 1, last_nonempty if target is None else None


def copy_format(worksheet, source_row: int, target_row: int, width: int) -> None:
    if source_row == target_row:
        return
    worksheet.spreadsheet.batch_update({
        "requests": [{
            "copyPaste": {
                "source": {
                    "sheetId": worksheet.id,
                    "startRowIndex": source_row - 1,
                    "endRowIndex": source_row,
                    "startColumnIndex": 0,
                    "endColumnIndex": width,
                },
                "destination": {
                    "sheetId": worksheet.id,
                    "startRowIndex": target_row - 1,
                    "endRowIndex": target_row,
                    "startColumnIndex": 0,
                    "endColumnIndex": width,
                },
                "pasteType": "PASTE_FORMAT",
            }
        }]
    })


def main() -> None:
    credential = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or \
        "/home/sergey/.config/conditioned-iqa/google-service-account.json"
    worksheet = gspread.service_account(filename=credential).open_by_key(SHEET_ID).worksheet(WORKSHEET)
    existing = worksheet.get_all_values(value_render_option="UNFORMATTED_VALUE")
    if not existing or len(existing[0]) < len(HEADERS) or existing[0][1:len(HEADERS)] != HEADERS[1:]:
        raise RuntimeError("worksheet header does not match the experiment table after the first column")
    client = MlflowClient("sqlite:///mlflow.db")
    raw = load_rows()
    for experiment in EXPERIMENTS:
        row = build_row(experiment, raw, client)
        if row is None:
            continue
        # Re-read immediately before selecting the target to avoid using a
        # stale last row if a colleague appended while this script ran.
        existing = worksheet.get_all_values(value_render_option="UNFORMATTED_VALUE")
        target, previous = find_target(existing, experiment["design"])
        if previous is not None:
            copy_format(worksheet, previous, target, len(HEADERS))
        end = gspread.utils.rowcol_to_a1(target, len(HEADERS))
        worksheet.update([[*row]], f"A{target}:{end}", value_input_option="RAW")
        print(f"wrote {experiment['design']} at row {target}")


if __name__ == "__main__":
    main()
