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
        hidden_dim: int | list[int] = 256,
        dropout: float = 0.1,
        fusion: str = "concat",
        cls_emb_size: int | None = None,
        condition_layer_norm: bool = False,
    ):
        super().__init__()
        self.fusion = fusion
        self.feature_norm = nn.LayerNorm(feature_dim)

        if cls_emb_size is not None and cls_emb_size < 1:
            raise ValueError("cls_emb_size must be positive or null")
        condition_dim = cls_emb_size or num_groups
        self.condition_norm = (
            nn.LayerNorm(num_groups) if condition_layer_norm else nn.Identity()
        )
        if cls_emb_size is None:
            self.label_embedding = nn.Identity()
        else:
            # A bias-free projection is an embedding lookup for a one-hot
            # label, and a differentiable weighted embedding for soft labels.
            self.label_embedding = nn.Linear(num_groups, cls_emb_size, bias=False)
            nn.init.normal_(self.label_embedding.weight, mean=0.0, std=0.02)

        if fusion == "concat":
            input_dim = feature_dim + condition_dim
            self.condition = nn.Identity()
        elif fusion == "add":
            input_dim = feature_dim
            self.condition = nn.Linear(condition_dim, feature_dim)
        elif fusion == "film":
            input_dim = feature_dim
            self.condition = nn.Linear(condition_dim, 2 * feature_dim)
        else:
            raise ValueError(f"unknown fusion {fusion!r}")

        hidden_dims = [hidden_dim] if isinstance(hidden_dim, int) else list(hidden_dim)
        if not hidden_dims or any(dim < 1 for dim in hidden_dims):
            raise ValueError("hidden_dim must be a positive integer or non-empty list of them")
        layers = []
        previous_dim = input_dim
        for dim in hidden_dims:
            layers.extend((nn.Linear(previous_dim, dim), nn.GELU(), nn.Dropout(dropout)))
            previous_dim = dim
        layers.append(nn.Linear(previous_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, distortion_labels: torch.Tensor
    ) -> torch.Tensor:
        features = self.feature_norm(features)
        distortion_labels = self.condition_norm(distortion_labels)
        distortion_labels = self.label_embedding(distortion_labels)
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
