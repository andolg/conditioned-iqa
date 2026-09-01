"""Build the designated text-conditioning joint-training label table.

The retained PIPAL rows have a known broad condition.  Its NTIRE validation
split and mixed classical-distortion class do not, so they are excluded to keep
the image-only and text-conditioned comparisons on exactly the same examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prepare_data import prepare


def filter_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows reserved for validation or without a reliable text condition."""
    before = len(frame)
    pipal = frame["dataset"].eq("pipal")
    reserved_pipal = pipal & frame["distortion"].astype(str).str.zfill(2).eq("10")
    missing_condition = frame["group"].isna()
    filtered = frame[~(reserved_pipal | missing_condition)].copy()
    print(
        f"filtered {before - len(filtered)} rows "
        f"({int(reserved_pipal.sum())} reserved PIPAL; "
        f"{int((missing_condition & ~reserved_pipal).sum())} without a condition)"
    )
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="prepared dataset roots")
    parser.add_argument("--out", type=Path, required=True, help="combined CSV destination")
    args = parser.parse_args()

    frames = []
    for root in args.roots:
        frame = prepare(root)
        print(f"{root.name}: {len(frame)} rows before joint-training filters")
        frames.append(frame)
    combined = filter_training_rows(pd.concat(frames, ignore_index=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"wrote {len(combined)} rows from {combined['dataset'].nunique()} datasets to {args.out}")


if __name__ == "__main__":
    main()
