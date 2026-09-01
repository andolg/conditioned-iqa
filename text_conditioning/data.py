"""Dataset wrapper that exposes the existing broad distortion group."""

from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import IQADataset


class ConditionedIQADataset(IQADataset):
    """The baseline dataset plus the prepared CSV's ``group`` column."""

    def __init__(
        self,
        csv,
        image_size: int = 224,
        backbone: str = "clip",
        score_column: str = "scaled_subjective_score",
        preprocessing: str = "stretch",
    ):
        super().__init__(csv, image_size=image_size, backbone=backbone, score_column=score_column)
        if preprocessing not in {"stretch", "resize_center_crop"}:
            raise ValueError(f"unknown preprocessing {preprocessing!r}")
        self.preprocessing = preprocessing

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Apply either the historical stretch or CLIP's standard resize/crop.

        The historical runner stretched every image directly to a square.  The
        ``resize_center_crop`` path instead resizes the shorter edge to the
        encoder size and center-crops, matching the CLIP checkpoint's
        documented preprocessing without changing the shared baseline module.
        """
        if self.preprocessing == "stretch":
            image = image.resize((self.image_size,) * 2, Image.Resampling.BICUBIC)
        else:
            width, height = image.size
            scale = self.image_size / min(width, height)
            resized = (round(width * scale), round(height * scale))
            image = image.resize(resized, Image.Resampling.BICUBIC)
            left = (image.width - self.image_size) // 2
            top = (image.height - self.image_size) // 2
            image = image.crop((left, top, left + self.image_size, top + self.image_size))
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (pixels - self.mean) / self.std

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        with Image.open(row["path"]) as source:
            pixels = self._prepare_image(source.convert("RGB"))
        group = row.get("group", "authentic")
        # KADID's prepared labels use American spelling while the shared
        # prompt taxonomy uses British spelling. Treat them as the same
        # condition instead of silently falling back to ``authentic``.
        if str(group).lower() == "color":
            group = "colour"
        return {
            "image": pixels,
            "target": torch.tensor(float(row[self.score_column]), dtype=torch.float32),
            "reference": str(row["reference"]),
            "dataset": str(row.get("dataset", "")),
            "distortion": str(row.get("distortion", "")),
            "level": int(row["level"]) if pd.notna(row.get("level")) else -1,
            "group": "authentic" if pd.isna(group) else str(group),
        }

    def subset(self, rows: pd.DataFrame) -> ConditionedIQADataset:
        backbone = "clip" if float(self.mean[0]) != 0.5 else "siglip"
        return ConditionedIQADataset(
            rows, self.image_size, backbone, self.score_column, self.preprocessing
        )


class FeatureDataset(Dataset):
    """Metadata/target dataset backed by precomputed frozen vision features."""

    def __init__(self, source: ConditionedIQADataset, features: dict[str, torch.Tensor]):
        self.rows = source.rows.reset_index(drop=True)
        self.features = [features[str(path)] for path in self.rows["path"]]
        self.score_column = source.score_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        group = row.get("group", "authentic")
        if str(group).lower() == "color":
            group = "colour"
        return {
            "features": self.features[index],
            "target": torch.tensor(float(row[self.score_column]), dtype=torch.float32),
            "reference": str(row["reference"]),
            "dataset": str(row.get("dataset", "")),
            "distortion": str(row.get("distortion", "")),
            "level": int(row["level"]) if pd.notna(row.get("level")) else -1,
            "group": "authentic" if pd.isna(group) else str(group),
        }

    def subset(self, rows: pd.DataFrame) -> "FeatureDataset":
        feature_map = {str(path): feature for path, feature in zip(self.rows["path"], self.features)}
        source = object.__new__(ConditionedIQADataset)
        source.rows = rows.reset_index(drop=True)
        source.score_column = self.score_column
        return FeatureDataset(source, feature_map)
