"""Dataset wrapper that exposes the existing broad distortion group."""

from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import IQADataset


# Prepared labels keep the score exactly as released and also provide a
# per-dataset min-max score.  MDTVSFA's scale-alignment stage needs the former,
# but scores must still point in the same direction for every dataset.
FLIPPED_SCORE_DATASETS = {"csiq", "liveiqa", "livemd"}


def target_from_row(row, score_column: str) -> float:
    """Read a training target, including the oriented raw-score column.

    ``prepare_data.py`` deliberately preserves ``original_subjective_score``
    unchanged.  For MDTVSFA we need raw scores while retaining the project's
    higher-is-better convention, so ``oriented_subjective_score`` flips DMOS
    datasets at load time without rewriting the prepared CSVs.
    """
    if score_column == "oriented_subjective_score":
        value = float(row["original_subjective_score"])
        if str(row.get("dataset", "")).lower() in FLIPPED_SCORE_DATASETS:
            value = -value
        return value
    return float(row[score_column])


def dataset_score_ranges(rows: pd.DataFrame, score_column: str) -> dict[str, tuple[float, float]]:
    """Return ``(minimum, maximum)`` target values for each dataset.

    The ranges are computed from the training partition only.  In the raw
    score mode the orientation is corrected in the same way as
    :func:`target_from_row`, while the historical scaled column is used as-is.
    """
    if score_column == "oriented_subjective_score":
        values = rows["original_subjective_score"].astype(float).copy()
    else:
        values = rows[score_column].astype(float).copy()
    if score_column == "oriented_subjective_score":
        flipped = rows["dataset"].astype(str).str.lower().isin(FLIPPED_SCORE_DATASETS)
        values.loc[flipped] = -values.loc[flipped]
    frame = pd.DataFrame({"dataset": rows["dataset"].astype(str), "score": values})
    return {
        name: (float(group["score"].min()), float(group["score"].max()))
        for name, group in frame.groupby("dataset", sort=True)
    }


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
        if preprocessing not in {"stretch", "resize_center_crop", "multiscale"}:
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

    def _prepare_image_views(self, image: Image.Image) -> torch.Tensor:
        """Return one global and four local views without stretching the image.

        CLIP-B still receives its native 224x224 input.  The global view uses
        the normal aspect-preserving resize/center crop; the four local views
        are 224px tiles from a 2x enlarged image.  This keeps small local
        distortions visible while retaining a global context view.
        """
        image = image.convert("RGB")
        size = self.image_size
        short = min(image.width, image.height)
        scale = size / max(short, 1)
        resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
        left = max((resized.width - size) // 2, 0)
        top = max((resized.height - size) // 2, 0)
        global_view = resized.crop((left, top, left + size, top + size))

        tile_scale = (2 * size) / max(short, 1)
        enlarged = image.resize((round(image.width * tile_scale), round(image.height * tile_scale)), Image.Resampling.BICUBIC)
        max_left = max(enlarged.width - size, 0)
        max_top = max(enlarged.height - size, 0)
        positions = ((0, 0), (max_left, 0), (0, max_top), (max_left, max_top))
        views = [global_view]
        for x, y in positions:
            crop = enlarged.crop((x, y, x + size, y + size))
            if crop.size != (size, size):
                crop = crop.resize((size, size), Image.Resampling.BICUBIC)
            views.append(crop)

        tensors = []
        for view in views:
            pixels = torch.from_numpy(np.asarray(view, dtype=np.float32) / 255.0).permute(2, 0, 1)
            tensors.append((pixels - self.mean) / self.std)
        return torch.stack(tensors)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        with Image.open(row["path"]) as source:
            if self.preprocessing == "multiscale":
                pixels = None
                image_views = self._prepare_image_views(source)
            else:
                pixels = self._prepare_image(source.convert("RGB"))
                image_views = None
        group = row.get("group", "authentic")
        # KADID's prepared labels use American spelling while the shared
        # prompt taxonomy uses British spelling. Treat them as the same
        # condition instead of silently falling back to ``authentic``.
        if str(group).lower() == "color":
            group = "colour"
        item = {
            "target": torch.tensor(target_from_row(row, self.score_column), dtype=torch.float32),
            "reference": str(row["reference"]),
            "dataset": str(row.get("dataset", "")),
            "distortion": str(row.get("distortion", "")),
            "level": int(row["level"]) if pd.notna(row.get("level")) else -1,
            "group": "authentic" if pd.isna(group) else str(group),
        }
        if image_views is not None:
            item["image_views"] = image_views
        else:
            item["image"] = pixels
        return item

    def subset(self, rows: pd.DataFrame) -> ConditionedIQADataset:
        backbone = "clip" if float(self.mean[0]) != 0.5 else "siglip"
        return ConditionedIQADataset(
            rows, self.image_size, backbone, self.score_column, self.preprocessing
        )


class FeatureDataset(Dataset):
    """Metadata/target dataset backed by precomputed frozen vision features."""

    def __init__(self, source: ConditionedIQADataset, features: dict[str, torch.Tensor | dict[str, torch.Tensor]]):
        self.rows = source.rows.reset_index(drop=True)
        entries = [features[str(path)] for path in self.rows["path"]]
        self.features = [entry["pooled"] if isinstance(entry, dict) and "pooled" in entry else entry for entry in entries]
        self.view_features = [entry["views"] for entry in entries] if entries and isinstance(entries[0], dict) and "views" in entries[0] else None
        self.patch_features = [entry["patches"] for entry in entries] if entries and isinstance(entries[0], dict) and "patches" in entries[0] else None
        self.score_column = source.score_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        group = row.get("group", "authentic")
        if str(group).lower() == "color":
            group = "colour"
        item = {
            "target": torch.tensor(target_from_row(row, self.score_column), dtype=torch.float32),
            "reference": str(row["reference"]),
            "dataset": str(row.get("dataset", "")),
            "distortion": str(row.get("distortion", "")),
            "level": int(row["level"]) if pd.notna(row.get("level")) else -1,
            "group": "authentic" if pd.isna(group) else str(group),
        }
        if self.view_features is not None:
            item["view_features"] = self.view_features[index]
        else:
            item["features"] = self.features[index]
        if self.patch_features is not None:
            item["patch_features"] = self.patch_features[index]
        return item

    def subset(self, rows: pd.DataFrame) -> "FeatureDataset":
        feature_map = {
            str(path): ({"pooled": feature, "patches": patches} if self.patch_features is not None else feature)
            for path, feature, patches in zip(
                self.rows["path"], self.features,
                self.patch_features if self.patch_features is not None else [None] * len(self.features),
            )
        }
        if self.view_features is not None:
            feature_map = {str(path): {"views": views} for path, views in zip(self.rows["path"], self.view_features)}
        source = object.__new__(ConditionedIQADataset)
        source.rows = rows.reset_index(drop=True)
        source.score_column = self.score_column
        return FeatureDataset(source, feature_map)
