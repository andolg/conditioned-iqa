"""Build a GFIQA-augmented training table without known KonIQ overlap.

GFIQA image names contain a Flickr source ID followed by a crop/variant
suffix (for example ``10296038236_0.png``).  KonIQ uses the source ID as the
image filename.  This script removes every GFIQA row whose source ID occurs
in the labeled KonIQ table, records the decision for every row, and appends
the retained rows to the existing clean-mixture table.

The source-ID filter is deterministic and conservative.  It is a minimum
screen, not a claim that images with different IDs can never be near-
duplicates; any later visual screen should consume the emitted manifest and
append its reason rather than silently changing the training CSV.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


GFI_SOURCE_ID = re.compile(r"^(?P<id>\d+)_\d+$")
KONIQ_SOURCE_ID = re.compile(r"^(?P<id>\d+)$")
REQUIRED = {
    "path", "original_subjective_score", "scaled_subjective_score", "dataset",
    "reference", "distortion", "level", "group",
}


def _source_id(path: str, pattern: re.Pattern[str], label: str) -> str:
    match = pattern.match(Path(path).stem)
    if not match:
        raise ValueError(f"{label}: cannot extract source ID from {path!r}")
    return match.group("id")


def _read(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} {path} lacks columns: {sorted(missing)}")
    return frame


def build(
    base_paths: list[Path],
    gfiqa_path: Path,
    koniq_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> tuple[int, int, int]:
    base = pd.concat([_read(path, "base table") for path in base_paths], ignore_index=True)
    gfiqa = _read(gfiqa_path, "GFIQA table").copy()
    koniq = _read(koniq_path, "KonIQ table")

    gfiqa["source_id"] = gfiqa["path"].map(
        lambda value: _source_id(str(value), GFI_SOURCE_ID, "GFIQA")
    )
    koniq_ids = {
        _source_id(str(value), KONIQ_SOURCE_ID, "KonIQ")
        for value in koniq["path"]
    }
    gfiqa["keep"] = ~gfiqa["source_id"].isin(koniq_ids)
    gfiqa["screen_reason"] = gfiqa["keep"].map(
        {True: "kept", False: "same_source_id_as_koniq"}
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    gfiqa[["path", "source_id", "keep", "screen_reason"]].to_csv(
        manifest_path, index=False
    )

    retained = gfiqa.loc[gfiqa["keep"], base.columns].copy()
    combined = pd.concat([base, retained], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return len(gfiqa), int((~gfiqa["keep"]).sum()), len(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", nargs="+", type=Path, required=True,
        help="prepared clean-mixture tables to retain unchanged",
    )
    parser.add_argument("--gfiqa", type=Path, required=True, help="prepared GFIQA labels.csv")
    parser.add_argument("--koniq", type=Path, required=True, help="prepared labeled KonIQ labels.csv")
    parser.add_argument("--out", type=Path, required=True, help="combined screened training table")
    parser.add_argument("--manifest", type=Path, required=True, help="per-GFIQA screening decisions")
    args = parser.parse_args()
    gfiqa_rows, removed, combined_rows = build(
        args.base, args.gfiqa, args.koniq, args.out, args.manifest
    )
    print(
        f"GFIQA rows: {gfiqa_rows}; removed by KonIQ source ID: {removed}; "
        f"combined training rows: {combined_rows}"
    )


if __name__ == "__main__":
    main()
