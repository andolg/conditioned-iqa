"""A torch Dataset over the CSV that `prepare_data.py` writes.

    from dataset import IQADataset, split_by, make_sampler

    data = IQADataset("~/iqa-data/kadid10k/labels.csv", image_size=224)
    train, val = split_by(data, "reference")
    loader = DataLoader(train, batch_size=32, sampler=make_sampler(train, "balanced"))

Every dataset looks the same by the time it gets here — `prepare_data.py`
already did the work of reading whichever label format the release shipped.
This file only loads images and hands out indices, which is why the two
things worth thinking about, splitting and sampling, are the only things in
it besides `__getitem__`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

# Normalization each backbone family was trained with.
STATS = {
    "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    "siglip": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


class IQADataset(Dataset):
    """Images and quality scores from a prepared CSV.

    csv:         what `prepare_data.py` wrote
    image_size:  square size the backbone wants (224, 256, 336, ...)
    backbone:    "clip" or "siglip" — picks the normalization statistics
    score_column: "scaled_subjective_score" is min-maxed to [0, 1] with
                  higher = better; "original_subjective_score" is the number
                  the release published.  The conditioned runner also accepts
                  "oriented_subjective_score" and flips DMOS datasets at load
                  time while retaining their original units.
    """

    def __init__(
        self,
        csv: str | Path | pd.DataFrame,
        image_size: int = 224,
        backbone: str = "clip",
        score_column: str = "scaled_subjective_score",
    ):
        self.rows = csv.reset_index(drop=True) if isinstance(csv, pd.DataFrame) \
            else pd.read_csv(Path(csv).expanduser())
        self.image_size = image_size
        self.score_column = score_column
        mean, std = STATS[backbone]
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        row = self.rows.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        image = image.resize((self.image_size,) * 2, Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return {
            "image": (pixels - self.mean) / self.std,
            "target": torch.tensor(float(row[self.score_column]), dtype=torch.float32),
            "reference": str(row["reference"]),
            "dataset": str(row.get("dataset", "")),
            "distortion": str(row.get("distortion", "")),
            "level": int(row["level"]) if pd.notna(row.get("level")) else -1,
        }

    def subset(self, rows: pd.DataFrame) -> "IQADataset":
        backbone = "clip" if float(self.mean[0]) != 0.5 else "siglip"
        return IQADataset(rows, self.image_size, backbone, self.score_column)


def _blocks(rows: pd.DataFrame):
    """The frame split per dataset, or as one block when the column is absent.

    Both strategies below draw the held-out share from each dataset separately.
    On a single-dataset CSV that changes nothing; on a combined one it is the
    difference between a split and a lottery, because a reference means
    different things in different releases — one photograph in KonIQ, 125 rows
    in KADID. Pooling them lets one release dominate the draw, and a dataset
    can miss the held-out side entirely.
    """
    if "dataset" not in rows.columns:
        return [rows]
    return [block for _, block in rows.groupby("dataset", sort=True)]


def split_by(dataset: IQADataset, strategy: str = "reference", fraction: float = 0.2, seed: int = 0):
    """Split into (train, held out), taking `fraction` from every dataset.

    "reference": every version of one pristine image lands on one side. The
        default, and the only honest option for the synthetic sets: 125
        images share a reference — the same photo at 25 distortions and 5
        levels — and splitting them across the boundary measures
        memorization. On frozen features that inflates SRCC by up to 0.44.
    "random": split by image. Fine for photographs, where every image is its
        own scene; wrong for anything with references.
    """
    rng = np.random.default_rng(seed)
    if strategy == "random":
        held_index = []
        for block in _blocks(dataset.rows):
            order = rng.permutation(len(block))
            cut = int(len(order) * (1 - fraction))
            held_index.extend(block.index[order[cut:]])
        mask = dataset.rows.index.isin(held_index)
        train, held = dataset.rows[~mask], dataset.rows[mask]
    elif strategy == "reference":
        held_refs: set[str] = set()
        for block in _blocks(dataset.rows):
            references = np.array(sorted(block["reference"].unique()))
            keep = max(1, round(len(references) * fraction))
            held_refs.update(references[rng.permutation(len(references))][:keep])
        mask = dataset.rows["reference"].isin(held_refs)
        train, held = dataset.rows[~mask], dataset.rows[mask]
    else:
        raise ValueError(f"unknown split strategy {strategy!r}; use 'reference' or 'random'")
    return dataset.subset(train), dataset.subset(held)


def make_sampler(dataset: IQADataset, strategy: str = "random", seed: int = 0) -> Sampler | None:
    """A sampler for the DataLoader, or None to let `shuffle=True` do it.

    "random":   None — plain shuffling.
    "balanced": every distortion type contributes equally to a batch, rather
                than the type with the most images deciding.
    "by_level": every severity level equally weighted, so the easy levels do
                not swamp the hard ones.
    "by_dataset": every dataset equally weighted, for a combined CSV where
                one set is ten times the size of another.
    """
    if strategy == "random":
        return None
    column = {"balanced": "distortion", "by_level": "level", "by_dataset": "dataset"}.get(strategy)
    if column is None:
        raise ValueError(f"unknown sampling strategy {strategy!r}")
    keys = dataset.rows[column].fillna("none").astype(str)
    weights = torch.tensor((1.0 / keys.map(keys.value_counts())).to_numpy(), dtype=torch.double)
    return WeightedRandomSampler(
        weights, num_samples=len(keys), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
