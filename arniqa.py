"""Frozen ARNIQA distortion encoder and per-image embedding utilities.

The conditioned models use only the self-supervised encoder checkpoint.
Standalone ARNIQA evaluation can additionally load one of the official
dataset-specific regressors.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


ARNIQA_WEIGHTS_URL = (
    "https://github.com/miccunifi/ARNIQA/releases/download/weights/ARNIQA.pth"
)
ARNIQA_REGRESSOR_URL = (
    "https://github.com/miccunifi/ARNIQA/releases/download/weights/"
    "regressor_{dataset}.pth"
)
ARNIQA_FEATURE_DIM = 4096
ARNIQA_CROP_SIZE = 224
ARNIQA_REGRESSOR_METADATA = {
    "live": (1.0, 100.0, True),
    "csiq": (0.0, 1.0, True),
    "tid2013": (0.0, 9.0, False),
    "kadid10k": (1.0, 5.0, False),
    "flive": (1.0, 100.0, False),
    "spaq": (1.0, 100.0, False),
    "clive": (1.0, 100.0, False),
    "koniq10k": (1.0, 100.0, False),
}


class ARNIQADistortionEncoder(nn.Module):
    """The ResNet-50 encoder and discarded contrastive projector from ARNIQA."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        try:
            from torchvision.models import resnet50
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise ImportError(
                "ARNIQA conditioning requires torchvision; reinstall the project dependencies"
            ) from exc

        model = resnet50(weights=None)
        self.feat_dim = model.fc.in_features
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images).flatten(1)
        return torch.nn.functional.normalize(features, dim=1)


def load_arniqa_encoder(
    weights: str | None,
    device: torch.device,
) -> ARNIQADistortionEncoder:
    """Load and freeze the official self-supervised ARNIQA encoder weights."""
    encoder = ARNIQADistortionEncoder()
    if weights:
        path = Path(weights).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ARNIQA encoder checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        state = torch.hub.load_state_dict_from_url(
            ARNIQA_WEIGHTS_URL,
            map_location="cpu",
            progress=True,
            file_name="ARNIQA.pth",
        )
    encoder.load_state_dict(state, strict=True)
    # ARNIQA trains with the 128-d contrastive projector but explicitly
    # discards it for downstream IQA; keeping it would waste parameters and
    # memory while never participating in the forward pass.
    encoder.projector = nn.Identity()
    return encoder.eval().requires_grad_(False).to(device)


def load_arniqa_regressor(
    weights: str | None,
    dataset: str,
    device: torch.device,
) -> nn.Module:
    """Load an official frozen ARNIQA dataset-specific Ridge regressor."""
    if dataset not in ARNIQA_REGRESSOR_METADATA:
        choices = ", ".join(sorted(ARNIQA_REGRESSOR_METADATA))
        raise ValueError(f"unsupported ARNIQA regressor {dataset!r}; choose from {choices}")
    if weights:
        path = Path(weights).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ARNIQA regressor checkpoint not found: {path}")
    else:
        path = Path(torch.hub.get_dir()) / "checkpoints" / f"regressor_{dataset}.pth"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                ARNIQA_REGRESSOR_URL.format(dataset=dataset),
                str(path),
                progress=True,
            )
    regressor = torch.jit.load(str(path), map_location="cpu")
    if not isinstance(regressor, nn.Module):
        raise TypeError("official ARNIQA regressor checkpoint did not contain a module")
    # Official regressors are TorchScript modules, which do not implement the
    # module-level requires_grad_ convenience method.
    for parameter in regressor.parameters():
        parameter.requires_grad_(False)
    return regressor.eval().to(device)


def scale_arniqa_score(score: torch.Tensor, dataset: str) -> torch.Tensor:
    """Scale an official regressor output to [0, 1], with higher meaning better."""
    minimum, maximum, is_dmos = ARNIQA_REGRESSOR_METADATA[dataset]
    scaled = (score - minimum) / (maximum - minimum)
    return 1.0 - scaled if is_dmos else scaled


@torch.no_grad()
def embed_arniqa(
    encoder: nn.Module,
    full_scale: torch.Tensor,
    half_scale: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Return one 4096-d ARNIQA condition per image.

    Inputs have shape B x 5 x 3 x 224 x 224. The encoder is run on the five
    center/corner crops at both scales in bounded chunks, the two 2048-d
    scale features are concatenated per crop, and crops are averaged.
    """
    if full_scale.ndim != 5 or half_scale.shape != full_scale.shape:
        raise ValueError(
            "ARNIQA inputs must be matching B x crops x C x H x W tensors"
        )
    if chunk_size < 1:
        raise ValueError("ARNIQA chunk size must be positive")

    batch_size, num_crops = full_scale.shape[:2]
    flattened = torch.cat(
        (full_scale.flatten(0, 1), half_scale.flatten(0, 1)), dim=0
    )
    encoded = []
    for start in range(0, len(flattened), chunk_size):
        value = encoder(flattened[start:start + chunk_size])
        if isinstance(value, (tuple, list)):
            value = value[0]
        encoded.append(value.float())
    features = torch.cat(encoded, dim=0)
    split = batch_size * num_crops
    combined = torch.cat((features[:split], features[split:]), dim=1)
    return combined.view(batch_size, num_crops, -1).mean(dim=1)
