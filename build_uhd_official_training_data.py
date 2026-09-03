"""Build a leakage-safe clean-mixture + UHD-IQA training protocol.

The UHD release publishes fixed ``training``, ``validation``, and ``test``
partitions.  This utility keeps the official test images out of the training
CSV entirely, writes a train/validation manifest consumed by the runner, and
writes the 900-image official test CSV separately for final evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def reference_partitions(frame: pd.DataFrame, seed: int, validation_fraction: float) -> pd.Series:
    """Make a deterministic, per-dataset reference split for existing sources."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    result = pd.Series("train", index=frame.index, dtype="object")
    for _, block in frame.groupby("dataset", sort=True):
        references = np.array(sorted(block["reference"].astype(str).unique()))
        count = max(1, round(len(references) * validation_fraction))
        validation_references = set(rng.permutation(references)[:count])
        result.loc[block.index[block["reference"].astype(str).isin(validation_references)]] = "validation"
    return result


def build(
    clean_labels: Path,
    uhd_labels: Path,
    uhd_metadata: Path,
    output_root: Path,
    *,
    seed: int,
    validation_fraction: float,
) -> dict[str, Path]:
    clean = pd.read_csv(clean_labels)
    uhd = pd.read_csv(uhd_labels)
    metadata = pd.read_csv(uhd_metadata, usecols=["image_name", "set"])
    if set(clean["dataset"].astype(str)) & {"uhdiqa"}:
        raise ValueError("clean_labels must not already contain UHD-IQA rows")

    official_partition = metadata.set_index("image_name")["set"]
    uhd = uhd.copy()
    uhd["official_partition"] = uhd["path"].map(lambda value: official_partition.get(Path(value).name))
    if uhd["official_partition"].isna().any():
        raise ValueError("some UHD labels could not be matched to official metadata")
    expected = {"training": 4269, "validation": 904, "test": 900}
    observed = uhd["official_partition"].value_counts().to_dict()
    if observed != expected:
        raise ValueError(f"unexpected UHD official partition counts: {observed}; expected {expected}")

    # The existing UHD labels table min-maxes MOS using all 6,073 images,
    # including the official test partition.  Keep the published MOS instead:
    # it is already bounded to [0, 1] and does not derive any target transform
    # from held-out labels.
    uhd["scaled_subjective_score"] = uhd["original_subjective_score"].astype(float)

    clean_manifest = pd.DataFrame({
        "path": clean["path"].astype(str),
        "dataset": clean["dataset"].astype(str),
        "partition": reference_partitions(clean, seed, validation_fraction),
        "split_source": "reference_seeded",
    })
    uhd_train_validation = uhd[uhd["official_partition"].isin(["training", "validation"])].copy()
    uhd_manifest = pd.DataFrame({
        "path": uhd_train_validation["path"].astype(str),
        "dataset": "uhdiqa",
        "partition": uhd_train_validation["official_partition"].replace({"training": "train"}),
        "split_source": "uhd_official",
    })
    manifest = pd.concat([clean_manifest, uhd_manifest], ignore_index=True)
    if manifest["path"].duplicated().any():
        raise ValueError("combined manifest contains duplicate image paths")
    if set(manifest["partition"]) != {"train", "validation"}:
        raise ValueError("training manifest must contain only train and validation partitions")

    train_validation = pd.concat(
        [clean, uhd_train_validation.drop(columns="official_partition")], ignore_index=True
    )
    test = uhd[uhd["official_partition"].eq("test")].drop(columns="official_partition").copy()
    train_validation_paths = set(train_validation["path"].astype(str))
    test_paths = set(test["path"].astype(str))
    if train_validation_paths & test_paths:
        raise ValueError("official UHD test paths leaked into the training/validation table")
    if len(test) != expected["test"]:
        raise AssertionError("official UHD test table must contain exactly 900 rows")

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "labels": output_root / "labels.csv",
        "manifest": output_root / "split_manifest.csv",
        "uhd_test": output_root / "uhd_official_test.csv",
        "uhd_partitions": output_root / "uhd_official_partitions.csv",
    }
    train_validation.to_csv(paths["labels"], index=False)
    manifest.to_csv(paths["manifest"], index=False)
    test.to_csv(paths["uhd_test"], index=False)
    uhd[["path", "dataset", "official_partition"]].rename(
        columns={"official_partition": "partition"}
    ).to_csv(paths["uhd_partitions"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-labels", type=Path,
        default=Path("/home/sergey/conditioned-iqa/data/multi_train_clean/labels.csv"),
    )
    parser.add_argument(
        "--uhd-labels", type=Path,
        default=Path("/home/sergey/conditioned-iqa/data/uhdiqa/labels.csv"),
    )
    parser.add_argument(
        "--uhd-metadata", type=Path,
        default=Path("/home/sergey/conditioned-iqa/data/uhdiqa/uhd-iqa-metadata.csv"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("/home/sergey/conditioned-iqa/data/multi_train_clean_uhd_official"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    paths = build(
        args.clean_labels, args.uhd_labels, args.uhd_metadata, args.out,
        seed=args.seed, validation_fraction=args.validation_fraction,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
