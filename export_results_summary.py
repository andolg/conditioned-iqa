"""Export the unified IQA design table to CSV and Google Sheets.

One row per metric/design.  The first columns describe the model, epochs and
seed, latency/memory, followed by per-dataset SRCC/PLCC pairs in alternating
order (KADID SRCC, KADID PLCC, ...).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

SHEET_ID = "1DZQKInig5PtctN23TXEkMzrdcc1uA9aATaxT3LTXYZY"

TRAIN_DATASETS = ("kadid10k", "spaq", "gfiqa20k", "pipal", "aigciqa2023")
TEST_DATASETS = ("tid2013", "csiq", "cid2013", "koniq10k", "clive", "agiqa3k", "uhdiqa")
DATASETS = TRAIN_DATASETS + TEST_DATASETS
DATASET_NAMES = {
    "kadid10k": "KADID-10k",
    "spaq": "SPAQ",
    "gfiqa20k": "GFIQA-20k",
    "pipal": "PIPAL",
    "aigciqa2023": "AIGCIQA2023",
    "tid2013": "TID2013",
    "csiq": "CSIQ",
    "cid2013": "CID2013",
    "koniq10k": "KonIQ-10k",
    "clive": "CLIVE",
    "agiqa3k": "AGIQA-3K",
    "uhdiqa": "UHD-IQA",
}

HEADERS = [
    "Design",
    "Description",
    "Backbone",
    "Train datasets",
    "Epochs",
    "Seed",
    "Baseline",
    "Latency p50 (ms)",
    "Latency p95 (ms)",
    "Peak memory (MB)",
    "FPS",
]
for dataset in DATASETS:
    name = DATASET_NAMES[dataset]
    HEADERS.extend((f"{name} SRCC", f"{name} PLCC"))
HEADERS.extend([
    "Avg validation SRCC",
    "Avg validation PLCC",
    "Avg test SRCC",
    "Avg test PLCC",
    "Avg val+test SRCC",
    "Avg val+test PLCC",
    "Parameters",
    "GFLOPs",
])

METHOD_ORDER = {"baseline": 0, "concat": 1, "interaction": 2, "residual": 3}
BACKBONE_ORDER = {"clip-base": 0, "clip-large": 1, "siglip": 2, "siglip2-base": 3, "siglip2-large": 4}

GFLOP_BY_DESIGN = {
    ("clip-base", "baseline"): 33.6965,
    ("clip-base", "concat"): 33.6965,
    ("clip-base", "interaction"): 33.6966,
    ("clip-base", "residual"): 33.6970,
    ("clip-large", "baseline"): 349.1905,
    ("clip-large", "interaction"): 349.1914,
}

DESIGNS = {
    ("clip-base", "baseline"): {
        "design": "CLIP-B/16 baseline",
        "description": "Baseline: frozen CLIP-B/16 [CLS] -> LN -> MLP(768,256,1)",
        "backbone": "CLIP-B/16",
    },
    ("clip-base", "baseline", "matched_capacity"): {
        "design": "CLIP-B/16 matched-capacity image-only control",
        "description": (
            "Image-only MLP(768,686,1), parameter-matched to the calibrated "
            "text-interaction head; KADID-10k, SPAQ, AIGCIQA2023 training"
        ),
        "backbone": "CLIP-B/16",
    },
    ("clip-base", "concat"): {
        "design": "CLIP-B/16 concat",
        "description": "Concat [image, text] -> MLP",
        "backbone": "CLIP-B/16",
    },
    ("clip-base", "interaction"): {
        "design": "CLIP-B/16 interaction",
        "description": "Concat [image, text, image*text] -> MLP",
        "backbone": "CLIP-B/16",
    },
    ("clip-base", "interaction", "instructor"): {
        "design": "CLIP-B/16 interaction (INSTRUCTOR)",
        "description": "Concat [image, INSTRUCTOR text, image*text] -> MLP",
        "backbone": "CLIP-B/16",
    },
    ("clip-base", "residual"): {
        "design": "CLIP-B/16 residual",
        "description": "Unconditional base score + text correction",
        "backbone": "CLIP-B/16",
    },
    ("clip-large", "baseline"): {
        "design": "CLIP-L/14@336 baseline",
        "description": "Baseline: frozen CLIP-L/14@336 [CLS] -> LN -> MLP(1024,256,1)",
        "backbone": "CLIP-L/14@336",
    },
    ("clip-large", "baseline", "joint"): {
        "design": "CLIP-L/14@336 joint baseline",
        "description": "Baseline: frozen CLIP-L/14@336, multi-dataset training suite",
        "backbone": "CLIP-L/14@336",
    },
    ("clip-large", "interaction"): {
        "design": "CLIP-L/14@336 interaction",
        "description": "Concat [image, text, image*text] -> MLP",
        "backbone": "CLIP-L/14@336",
    },
    ("clip-large", "interaction", "joint"): {
        "design": "CLIP-L/14@336 joint interaction",
        "description": "Concat [image, text, image*text] -> MLP, multi-dataset training suite",
        "backbone": "CLIP-L/14@336",
    },
}


def number(value: str | float | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def rounded(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def format_many(values: list[str]) -> str:
    values = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if not values:
        return ""
    if all(value.isdigit() for value in values):
        numbers = sorted(int(value) for value in values)
        if numbers[-1] - numbers[0] == len(numbers) - 1:
            return f"{numbers[0]}-{numbers[-1]}" if len(numbers) > 1 else str(numbers[0])
    return ", ".join(values)


def metric_value(rows: list[dict[str, str]], dataset: str, key: str) -> float | None:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    values = [number(row.get(key)) for row in dataset_rows]
    return mean(values)


def metric_cell(rows: list[dict[str, str]], dataset: str, key: str) -> str:
    return rounded(metric_value(rows, dataset, key))


NUMERIC_HEADERS = {
    "Epochs",
    "Latency p50 (ms)",
    "Latency p95 (ms)",
    "Peak memory (MB)",
    "FPS",
    "Avg validation SRCC",
    "Avg validation PLCC",
    "Avg test SRCC",
    "Avg test PLCC",
    "Avg val+test SRCC",
    "Avg val+test PLCC",
    "GFLOPs",
}
NUMERIC_HEADERS.update(
    f"{DATASET_NAMES[dataset]} {metric}"
    for dataset in DATASETS
    for metric in ("SRCC", "PLCC")
)


def sheet_value(header: str, value: str) -> str | float | int:
    """Keep labels as text, but store metrics as native Google Sheet numbers."""
    if header not in NUMERIC_HEADERS or value in (None, ""):
        return value
    number_value = number(value)
    if number_value is None:
        return value
    if header == "Epochs" and number_value.is_integer():
        return int(number_value)
    return number_value


def canonical_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        run_name = str(row.get("run_name", "")).lower()
        if "smoke" in run_name:
            continue
        if "heldout" in run_name:
            continue
        if "instructor" in run_name and "best" not in run_name:
            continue
        filtered.append(row)
    return filtered


def design_family(row: dict[str, str], joint_source_ids: set[str] | None = None) -> str:
    metadata = " ".join((str(row.get("run_name", "")), str(row.get("config_path", "")))).lower()
    if "matched-capacity" in metadata or "matched_capacity" in metadata:
        return "matched_capacity"
    if "instructor" in metadata:
        return "instructor"
    source = str(row.get("source_run_id") or row.get("run_id") or "")
    if joint_source_ids is not None:
        return "joint" if source in joint_source_ids else "standard"
    run_name = str(row.get("run_name", "")).lower()
    return "joint" if run_name.startswith("joint-") else "standard"


def build_summary(source: Path) -> list[dict[str, str]]:
    with source.open(newline="", encoding="utf-8") as stream:
        raw = canonical_rows(list(csv.DictReader(stream)))

    joint_source_ids = {
        str(row.get("source_run_id") or row.get("run_id") or "")
        for row in raw
        if str(row.get("run_name", "")).lower().startswith("joint-")
    }

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[(row["backbone"], row["method"], design_family(row, joint_source_ids))].append(row)

    rows: list[dict[str, str]] = []
    keys = sorted(
        grouped,
        key=lambda key: (
            METHOD_ORDER.get(key[1], 99),
            BACKBONE_ORDER.get(key[0], 99),
            0 if key[2] == "standard" else 1,
            key[0],
            key[1],
        ),
    )

    for backbone, method, family in keys:
        group = grouped[(backbone, method, family)]
        design = DESIGNS.get(
            (backbone, method, family),
            DESIGNS.get(
                (backbone, method),
                {
                    "design": f"{backbone} {method}" + ("" if family == "standard" else f" {family}"),
                    "description": f"{method} on {backbone}",
                    "backbone": backbone,
                },
            ),
        )
        validation = [row for row in group if row["evaluation"] == "validation"]
        train_dataset_keys = []
        for dataset in TRAIN_DATASETS:
            if any(row["dataset"] == dataset for row in validation):
                train_dataset_keys.append(dataset)
        train_datasets = [DATASET_NAMES[dataset] for dataset in train_dataset_keys]

        latency_p50 = mean(
            [value for row in group if (value := number(row.get("latency_p50_ms"))) is not None]
        )
        latency_p95 = mean(
            [value for row in group if (value := number(row.get("latency_p95_ms"))) is not None]
        )
        peak_memory = mean(
            [value for row in group if (value := number(row.get("peak_memory_mb"))) is not None]
        )
        peak_memory_values = [
            value for row in group if (value := number(row.get("peak_memory_mb"))) is not None
        ]
        kadid_throughput = mean(
            [
                value
                for row in group
                if row["dataset"] == "kadid10k"
                and row["evaluation"] == "held_out_test"
                and (value := number(row.get("images_per_second"))) is not None
            ]
        )
        image_throughput = kadid_throughput if kadid_throughput else (
            1000 / latency_p50 if latency_p50 else None
        )

        summary: dict[str, str] = {
            "Design": design["design"],
            "Description": design["description"],
            "Backbone": design["backbone"],
            "Train datasets": ", ".join(train_datasets),
            "Epochs": format_many([str(row.get("epochs", "")) for row in group]),
            "Seed": format_many([str(row.get("seed", "")) for row in group]),
            "Baseline": "baseline" if method == "baseline" else "variant",
            "Latency p50 (ms)": rounded(latency_p50, 1),
            "Latency p95 (ms)": rounded(latency_p95, 1),
            "Peak memory (MB)": rounded(max(peak_memory_values), 1) if peak_memory_values else "",
            "FPS": rounded(image_throughput, 2),
        }

        train_srcc: list[float] = []
        train_plcc: list[float] = []
        test_srcc: list[float] = []
        test_plcc: list[float] = []

        for dataset in DATASETS:
            if dataset in train_dataset_keys:
                # A training-dataset column is a validation result only.  Do
                # not fall back to its held-out test result: that would make
                # an untrained dataset (for example SPAQ for a KADID-only
                # run) contribute to the validation aggregate.
                dataset_rows = [row for row in validation if row["dataset"] == dataset]
            else:
                dataset_rows = [
                    row
                    for row in group
                    if row["evaluation"] == "held_out_test" and row["dataset"] == dataset
                ]
            srcc_value = metric_value(dataset_rows, dataset, "srcc")
            plcc_value = metric_value(dataset_rows, dataset, "plcc")
            name = DATASET_NAMES[dataset]
            summary[f"{name} SRCC"] = rounded(srcc_value)
            summary[f"{name} PLCC"] = rounded(plcc_value)
            if dataset in train_dataset_keys:
                if srcc_value is not None:
                    train_srcc.append(srcc_value)
                if plcc_value is not None:
                    train_plcc.append(plcc_value)
            else:
                if srcc_value is not None:
                    test_srcc.append(srcc_value)
                if plcc_value is not None:
                    test_plcc.append(plcc_value)

        avg_val_srcc = mean(train_srcc)
        avg_val_plcc = mean(train_plcc)
        avg_test_srcc = mean(test_srcc)
        avg_test_plcc = mean(test_plcc)
        avg_all_srcc = mean([avg_val_srcc, avg_test_srcc])
        avg_all_plcc = mean([avg_val_plcc, avg_test_plcc])
        # Keep the table convention consistent across designs: report the
        # complete model footprint, including the frozen backbone.
        parameter_count = mean(
            [
                value
                for row in group
                if (value := number(row.get("model_parameter_size_mb"))) is not None
            ]
        )
        if parameter_count is not None:
            parameter_count = parameter_count * 1024**2 / 4
        gflops = GFLOP_BY_DESIGN.get((backbone, method))

        summary.update({
            "Avg validation SRCC": rounded(avg_val_srcc),
            "Avg validation PLCC": rounded(avg_val_plcc),
            "Avg test SRCC": rounded(avg_test_srcc),
            "Avg test PLCC": rounded(avg_test_plcc),
            "Avg val+test SRCC": rounded(avg_all_srcc),
            "Avg val+test PLCC": rounded(avg_all_plcc),
            "Parameters": f"{parameter_count / 1e6:.2f}M" if parameter_count is not None else "",
            "GFLOPs": f"{gflops:.4f}" if gflops is not None else "",
        })

        rows.append(summary)

    return rows


def export_google(rows: list[dict[str, str]], sheet_id: str, worksheet_name: str) -> None:
    import gspread

    credential = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or str(
        Path.home() / ".config/conditioned-iqa/google-service-account.json"
    )
    client = gspread.service_account(filename=credential)
    spreadsheet = client.open_by_key(sheet_id)
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(
            title=worksheet_name, rows=max(100, len(rows) + 10), cols=len(HEADERS)
        )

    existing = worksheet.get_all_values(value_render_option="UNFORMATTED_VALUE")
    if not existing or not any(str(value).strip() for value in existing[0]):
        worksheet.update([HEADERS], "A1", value_input_option="RAW")
        worksheet.freeze(rows=1)
        header_range = f"A1:{gspread.utils.rowcol_to_a1(1, len(HEADERS))}"
        worksheet.format(
            header_range,
            {
                "backgroundColor": {"red": 0.12, "green": 0.24, "blue": 0.45},
                "textFormat": {
                    "bold": True,
                    "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                },
                "horizontalAlignment": "CENTER",
            },
        )
        existing = [HEADERS]
    elif existing[0][: len(HEADERS)] != HEADERS:
        raise ValueError(
            f"Worksheet {worksheet_name!r} has an incompatible header; refusing to overwrite it."
        )

    # Upsert generated designs by their stable design name.  Extra rows (for
    # example, a colleague's independently added run) are deliberately left
    # untouched, and no worksheet-wide clear is performed.
    design_column = HEADERS.index("Design")
    existing_by_design = {
        str(row[design_column]).strip(): row_number
        for row_number, row in enumerate(existing[1:], start=2)
        if len(row) > design_column and str(row[design_column]).strip()
    }
    generated_values = [
        [sheet_value(header, row.get(header, "")) for header in HEADERS]
        for row in rows
    ]
    new_values = []
    for row, row_values in zip(rows, generated_values):
        design_name = str(row.get("Design", "")).strip()
        existing_row = existing_by_design.get(design_name)
        if existing_row is None:
            new_values.append(row_values)
            continue
        end = gspread.utils.rowcol_to_a1(existing_row, len(HEADERS))
        worksheet.update([row_values], f"A{existing_row}:{end}", value_input_option="RAW")
    if new_values:
        worksheet.append_rows(new_values, value_input_option="RAW")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("runs/results.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/results_summary.csv"))
    parser.add_argument("--google-sheet-id", default=os.getenv("IQA_GOOGLE_SHEET_ID", SHEET_ID))
    # The shared workbook's summary tab was renamed; keep the exporter on
    # that existing tab so a default export cannot create a second Summary
    # worksheet.
    parser.add_argument("--worksheet", default="28s_mur")
    args = parser.parse_args()

    rows = build_summary(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    export_google(rows, args.google_sheet_id, args.worksheet)
    print(f"Exported {len(rows)} design rows to {args.output} and worksheet {args.worksheet!r}.")


if __name__ == "__main__":
    main()
