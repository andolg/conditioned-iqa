"""Small pooled-feature heads used to isolate text conditioning."""

from __future__ import annotations

import torch
from torch import nn


class DatasetScaleHead(nn.Module):
    """Shared quality scorer with monotonic per-dataset calibration.

    The shared scorer is the value used for an unseen dataset. During
    training/validation, a known dataset receives its own sigmoid calibration
    so incompatible MOS ranges are not forced through one global scale. An
    unknown dataset deliberately falls back to the shared latent score.
    """

    def __init__(self, base: nn.Module, datasets: list[str]):
        super().__init__()
        self.base = base
        self.datasets = tuple(sorted(dict.fromkeys(str(name) for name in datasets)))
        self.index = {name: position for position, name in enumerate(self.datasets)}
        self.calibration_bias = nn.Parameter(torch.zeros(len(self.datasets)))
        # softplus(log-slope) starts at one and remains strictly positive.
        self.calibration_log_slope = nn.Parameter(
            torch.full((len(self.datasets),), 0.54132485)
        )

    @property
    def latent(self) -> nn.Module:
        return self.base

    def forward(self, vision: torch.Tensor, text: torch.Tensor | None = None,
                datasets: list[str] | tuple[str, ...] | None = None) -> torch.Tensor:
        latent = self.base(vision) if text is None else self.base(vision, text)
        if datasets is None:
            return latent
        names = [str(name) for name in datasets]
        known = torch.tensor([self.index.get(name, -1) for name in names], device=latent.device)
        output = latent.clone()
        mask = known.ge(0)
        if mask.any():
            slope = torch.nn.functional.softplus(self.calibration_log_slope[known[mask]])
            bias = self.calibration_bias[known[mask]]
            output[mask] = torch.sigmoid(bias + slope * latent[mask])
        return output


class TextFusionHead(nn.Module):
    """Project pooled vision/text features and predict one quality score."""

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        interaction: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.interaction = interaction
        self.vision = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, fusion_dim))
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, fusion_dim))
        input_dim = fusion_dim * (3 if interaction else 2)
        self.score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        vision = self.vision(vision)
        text = self.text(text)
        features = [vision, text]
        if self.interaction:
            features.append(vision * text)
        return self.score(torch.cat(features, dim=-1)).squeeze(-1)


class ResidualTextHead(nn.Module):
    """An unconditional scorer plus a text-dependent correction.

    A literal zero text vector exactly returns ``base(vision)``.  Condition
    dropout can therefore train and test a genuine unconditioned fallback.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.vision = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, fusion_dim))
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, fusion_dim))
        self.correction = nn.Sequential(
            nn.LayerNorm(fusion_dim * 3),
            nn.Linear(fusion_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        base = self.base(vision).squeeze(-1)
        vision_projected = self.vision(vision)
        text_projected = self.text(text)
        correction = self.correction(torch.cat(
            [vision_projected, text_projected, vision_projected * text_projected], dim=-1
        )).squeeze(-1)
        active = text.abs().sum(dim=-1).ne(0).to(correction.dtype)
        return base + active * correction
