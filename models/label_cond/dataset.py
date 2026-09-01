from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from dataset import IQADataset
from prepare_data import GROUPS

GROUP_TO_LABEL = {group: index for index, group in enumerate(GROUPS)}
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


class LabelConditionedDataset(IQADataset):
    """Prepared IQA rows with both encoder and classifier image tensors."""

    def __init__(
        self,
        csv: str | Path | pd.DataFrame,
        image_size: int,
        backbone: str,
        score_column: str = "scaled_subjective_score",
        classifier_image_size: int | None = None,
    ):
        rows = (
            csv.reset_index(drop=True)
            if isinstance(csv, pd.DataFrame)
            else pd.read_csv(Path(csv).expanduser())
        )
        rows = rows[rows["group"].isin(GROUP_TO_LABEL)].reset_index(drop=True)
        super().__init__(rows, image_size, backbone, score_column)
        self.classifier_image_size = classifier_image_size
        operations = []
        if classifier_image_size is not None:
            operations += [
                transforms.Resize(classifier_image_size + 32),
                transforms.CenterCrop(classifier_image_size),
            ]
        operations += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN.flatten(), IMAGENET_STD.flatten()),
        ]
        self.classifier_transform = transforms.Compose(operations)
        self.labels = [GROUP_TO_LABEL[group] for group in self.rows["group"]]

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        image = Image.open(self.rows.iloc[index]["path"]).convert("RGB")
        sample["classifier_image"] = self.classifier_transform(image)
        sample["group"] = torch.tensor(self.labels[index], dtype=torch.long)
        return sample

    def subset(self, rows: pd.DataFrame) -> "LabelConditionedDataset":
        backbone = "clip" if float(self.mean[0]) != 0.5 else "siglip"
        return LabelConditionedDataset(
            rows,
            self.image_size,
            backbone,
            self.score_column,
            self.classifier_image_size,
        )
