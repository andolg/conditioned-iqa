"""Dataset wrapper that exposes the existing broad distortion group."""

from __future__ import annotations

import pandas as pd

from dataset import IQADataset


class ConditionedIQADataset(IQADataset):
    """The baseline dataset plus the prepared CSV's ``group`` column."""

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        group = self.rows.iloc[index].get("group", "authentic")
        item["group"] = "authentic" if pd.isna(group) else str(group)
        return item

    def subset(self, rows: pd.DataFrame) -> ConditionedIQADataset:
        backbone = "clip" if float(self.mean[0]) != 0.5 else "siglip"
        return ConditionedIQADataset(rows, self.image_size, backbone, self.score_column)
