"""Summarize prepared IQA label tables before a joint-training run.

Example:
    uv run python audit_iqa_data.py /home/sergey/conditioned-iqa/data/multi_train/labels.csv \
        --output runs/audits/joint-training-data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "path", "original_subjective_score", "scaled_subjective_score", "dataset", "reference", "group",
}


def audit(path: Path) -> pd.DataFrame:
    """Return one reproducible data-quality row per source dataset."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    rows = []
    for name, dataset in frame.groupby("dataset", sort=True):
        original = dataset["original_subjective_score"]
        scaled = dataset["scaled_subjective_score"]
        rows.append({
            "dataset": name,
            "images": len(dataset),
            "references": dataset["reference"].nunique(),
            "missing_images": int((~dataset["path"].map(lambda value: Path(value).is_file())).sum()),
            "duplicate_paths": int(dataset["path"].duplicated().sum()),
            "original_min": original.min(),
            "original_p05": original.quantile(0.05),
            "original_median": original.median(),
            "original_p95": original.quantile(0.95),
            "original_max": original.max(),
            "scaled_min": scaled.min(),
            "scaled_max": scaled.max(),
            "groups": ", ".join(sorted(dataset["group"].dropna().astype(str).unique())),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, help="combined labels.csv from prepare_data.py")
    parser.add_argument("--output", type=Path, default=None, help="optional CSV destination")
    args = parser.parse_args()
    report = audit(args.labels)
    print(report.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
