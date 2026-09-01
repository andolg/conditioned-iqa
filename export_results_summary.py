"""Export the concise IQA experiment summary to CSV and Google Sheets."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

SHEET_ID = "1DZQKInig5PtctN23TXEkMzrdcc1uA9aATaxT3LTXYZY"
HEADERS = [
    "category", "dataset_or_condition", "backbone", "model", "seeds", "images",
    "srcc", "plcc", "delta_srcc", "images_per_second", "head_size_mb",
    "model_parameter_size_mb", "notes",
]
EXTERNAL_DATASETS = ("tid2013", "csiq", "cid2013", "koniq10k", "clive", "uhdiqa", "agiqa3k")


def number(row: dict[str, str], key: str) -> float | None:
    return float(row[key]) if row.get(key) else None


def metric_summary(rows: list[dict[str, str]]) -> tuple[float, float]:
    return mean(number(row, "srcc") for row in rows), mean(number(row, "plcc") for row in rows)


def add(rows: list[dict[str, object]], **values: object) -> None:
    rows.append({header: values.get(header, "") for header in HEADERS})


def build_summary(source: Path) -> list[dict[str, object]]:
    with source.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    summary: list[dict[str, object]] = []

    # The primary comparison: same KADID split, frozen CLIP-Base, five epochs, seeds 0--2.
    base_rows = [row for row in raw if row["evaluation"] == "validation" and row["dataset"] == "kadid10k"
                 and row["backbone"] == "clip-base" and "heldout" not in row["run_name"]]
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in base_rows:
        if (
            row["method"] in {"baseline", "concat", "interaction", "residual"}
            and row["run_name"].startswith(f"tc-{row['method']}-s")
        ):
            by_method[row["method"]].append(row)
    baseline_srcc, _ = metric_summary(by_method["baseline"])
    labels = {
        "baseline": "image-only baseline",
        "concat": "text concat",
        "interaction": "text interaction [v, t, v*t]",
        "residual": "residual text correction",
    }
    for method in ("baseline", "concat", "interaction", "residual"):
        method_rows = by_method[method]
        srcc, plcc = metric_summary(method_rows)
        add(summary, category="KADID validation", dataset_or_condition="KADID-10k reference split",
            backbone="CLIP-Base", model=labels[method], seeds="0-2", images=2000,
            srcc=round(srcc, 4), plcc=round(plcc, 4), delta_srcc=round(srcc - baseline_srcc, 4),
            notes=f"5 epochs; SRCC sample SD {stdev(number(row, 'srcc') for row in method_rows):.4f}")

    # Canonical versus wording interventions for the interaction method (means from the completed 3-seed study).
    canonical = 0.7861
    for condition, score in (("held-out paraphrase", 0.7752), ("generic prompt", 0.7265),
                             ("wrong condition", 0.7269), ("shuffled condition", 0.7190)):
        add(summary, category="Semantic intervention", dataset_or_condition=condition,
            backbone="CLIP-Base", model="text interaction [v, t, v*t]", seeds="0-2", images=2000,
            srcc=score, delta_srcc=round(score - canonical, 4),
            notes=f"Canonical condition SRCC {canonical:.4f}")

    add(summary, category="Zero-shot reference", dataset_or_condition="KADID-10k reference split",
        backbone="CLIP-Base", model="CLIP-IQA: good vs bad prompt", seeds="0", images=2000,
        srcc=0.5261, plcc=0.5409, notes="No learned IQA head")

    large_validation = [row for row in raw if row["evaluation"] == "validation" and row["dataset"] == "kadid10k"
                        and row["backbone"] == "clip-large"]
    large_by_method = {row["method"]: row for row in large_validation}
    large_baseline = number(large_by_method["baseline"], "srcc")
    for method, label in (("baseline", "image-only baseline"), ("interaction", "text interaction [v, t, v*t]")):
        row = large_by_method[method]
        add(summary, category="KADID validation", dataset_or_condition="KADID-10k reference split",
            backbone="CLIP-Large ViT-L/14@336", model=label, seeds="0", images=2000,
            srcc=round(number(row, "srcc"), 4), plcc=round(number(row, "plcc"), 4),
            delta_srcc=round(number(row, "srcc") - large_baseline, 4), notes="5 epochs")

    external = [row for row in raw if row["evaluation"] == "held_out_test" and row["backbone"] == "clip-large"]
    for dataset in EXTERNAL_DATASETS:
        matched = {row["method"]: row for row in external if row["dataset"] == dataset}
        baseline = matched["baseline"]
        for method, label in (("baseline", "image-only baseline"), ("interaction", "text interaction [v, t, v*t]")):
            row = matched[method]
            add(summary, category="Held-out transfer", dataset_or_condition=dataset.upper(),
                backbone="CLIP-Large ViT-L/14@336", model=label, seeds="0", images=int(row["images"]),
                srcc=round(number(row, "srcc"), 4), plcc=round(number(row, "plcc"), 4),
                delta_srcc=round(number(row, "srcc") - number(baseline, "srcc"), 4),
                images_per_second=round(number(row, "images_per_second"), 2),
                head_size_mb=round(number(row, "head_size_mb"), 3),
                model_parameter_size_mb=round(number(row, "model_parameter_size_mb"), 1),
                notes="Zero-retraining test; KADID-trained checkpoint")
    return summary


def export_google(rows: list[dict[str, object]], sheet_id: str, worksheet_name: str) -> None:
    import gspread

    credential = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or str(
        Path.home() / ".config/conditioned-iqa/google-service-account.json"
    )
    client = gspread.service_account(filename=credential)
    spreadsheet = client.open_by_key(sheet_id)
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=max(100, len(rows) + 10), cols=len(HEADERS))
    values = [HEADERS] + [[row[header] for header in HEADERS] for row in rows]
    worksheet.clear()
    worksheet.update(values, "A1")
    worksheet.freeze(rows=1)
    worksheet.format("A1:M1", {"backgroundColor": {"red": 0.12, "green": 0.24, "blue": 0.45},
                                 "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                                 "horizontalAlignment": "CENTER"})
    worksheet.columns_auto_resize(0, len(HEADERS) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("runs/results.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/results_summary.csv"))
    parser.add_argument("--google-sheet-id", default=os.getenv("IQA_GOOGLE_SHEET_ID", SHEET_ID))
    parser.add_argument("--worksheet", default="Summary")
    args = parser.parse_args()
    rows = build_summary(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    export_google(rows, args.google_sheet_id, args.worksheet)
    print(f"Exported {len(rows)} summary rows to {args.output} and worksheet {args.worksheet!r}.")


if __name__ == "__main__":
    main()
