from pathlib import Path

import torch
from torch import nn

from hf_mirror_utils import load_transformers_model_from_mirrors

BACKBONES = {
    "clip-base": ("openai/clip-vit-base-patch16", 224),
    "clip-large": ("openai/clip-vit-large-patch14-336", 336),
    "siglip": ("google/siglip-large-patch16-256", 256),
    "siglip2-base": ("google/siglip2-base-patch16-224", 224),
    "siglip2-large": ("google/siglip2-large-patch16-256", 256),
}


class LabelConditionedMetric(nn.Module):
    """Baseline quality MLP conditioned on a distortion-group vector."""

    def __init__(
        self,
        feature_dim: int,
        num_groups: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        fusion: str = "concat",
    ):
        super().__init__()
        self.fusion = fusion
        if fusion == "concat":
            input_dim = feature_dim + num_groups
            self.condition = nn.Identity()
        elif fusion == "add":
            input_dim = feature_dim
            self.condition = nn.Linear(num_groups, feature_dim)
        elif fusion == "film":
            input_dim = feature_dim
            self.condition = nn.Linear(num_groups, 2 * feature_dim)
        else:
            raise ValueError(f"unknown fusion {fusion!r}")

        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, features: torch.Tensor, distortion_labels: torch.Tensor
    ) -> torch.Tensor:
        if self.fusion == "concat":
            fused = torch.cat((features, distortion_labels), dim=1)
        elif self.fusion == "add":
            fused = features + self.condition(distortion_labels)
        else:
            scale, shift = self.condition(distortion_labels).chunk(2, dim=1)
            fused = features * (1.0 + scale) + shift
        return self.head(fused).squeeze(1)


def load_image_encoder(
    name: str, weights: str | None, device: torch.device
):
    from transformers import CLIPVisionModel, SiglipVisionModel

    model_id, image_size = BACKBONES[name]
    model_class = SiglipVisionModel if name.startswith("siglip") else CLIPVisionModel
    if weights:
        local_path = Path(weights).expanduser()
        model = model_class.from_pretrained(str(local_path), local_files_only=True)
    else:
        model = load_transformers_model_from_mirrors(model_class, model_id)
    model = model.eval().requires_grad_(False).to(device)
    return model, image_size, model.config.hidden_size


@torch.no_grad()
def encode_images(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return encoder(pixel_values=images).pooler_output.float()
