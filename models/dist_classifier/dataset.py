from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLEAN_CLASS = 25


def split_kadid(csv: str | Path, val_fraction: float, seed: int):
    rows = pd.read_csv(Path(csv).expanduser())
    references = np.array(sorted(rows["reference"].unique()))
    rng = np.random.default_rng(seed)
    val_count = max(1, round(len(references) * val_fraction))
    val_references = set(rng.permutation(references)[:val_count])
    is_val = rows["reference"].isin(val_references)
    return rows[~is_val].reset_index(drop=True), rows[is_val].reset_index(drop=True)


class KADIDDistortionDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, image_size: int | None = None, train: bool = False):
        self.samples = [
            (Path(row.path), int(row.distortion) - 1)
            for row in rows.itertuples()
        ]
        for _, group in rows.groupby("reference"):
            distorted = Path(group.iloc[0]["path"])
            clean = distorted.with_name(distorted.stem.split("_")[0] + distorted.suffix)
            self.samples.append((clean, CLEAN_CLASS))

        operations = []
        if image_size is not None:
            operations += [
                transforms.Resize(image_size + 32),
                transforms.RandomCrop(image_size) if train else transforms.CenterCrop(image_size),
            ]
        if train:
            operations.append(transforms.RandomHorizontalFlip())
        operations += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.transform = transforms.Compose(operations)
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)
