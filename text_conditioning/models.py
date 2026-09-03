"""Small pooled-feature heads used to isolate text conditioning."""

from __future__ import annotations

import torch
from torch import nn


def make_score_mlp(
    input_dim: int,
    hidden_dim: int,
    mlp_layers: int = 1,
    dropout: float = 0.1,
) -> nn.Sequential:
    """Build a score MLP with a configurable number of hidden layers.

    ``mlp_layers=1`` intentionally preserves the historical head exactly:
    LayerNorm -> Linear -> GELU -> Dropout -> Linear.  Additional layers are
    hidden-to-hidden GELU/dropout blocks; the output remains a scalar.
    """
    if mlp_layers < 1:
        raise ValueError(f"mlp_layers must be >= 1, got {mlp_layers}")
    layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
    layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
    for _ in range(mlp_layers - 1):
        layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)))
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)


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
        inputs = vision if isinstance(vision, tuple) else (vision,)
        latent = self.base(*inputs) if text is None else self.base(*inputs, text)
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


class MDTVSFAHead(nn.Module):
    """Three-stage MDTVSFA quality head.

    The stages are kept explicit so their distinct losses can supervise the
    same forward pass:

    ``relative`` is a shared, bounded score; ``perceptual`` is a shared
    nonlinear mapping of that score; and ``aligned`` is a dataset-specific
    affine mapping into the dataset's subjective-score units.  An unknown
    dataset has no learned affine map and therefore falls back to the shared
    perceptual score.
    """

    def __init__(
        self,
        base: nn.Module,
        datasets: list[str],
        score_ranges: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__()
        self.base = base
        self.datasets = tuple(dict.fromkeys(str(name) for name in datasets))
        self.index = {name: position for position, name in enumerate(self.datasets)}

        # The paper's four-parameter logistic mapping is implemented as
        # Linear(1, 1) -> Sigmoid -> Linear(1, 1).  It is shared by all
        # datasets; dataset-specific scale changes belong only to ``alignment``.
        self.nonlinear = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid(), nn.Linear(1, 1))
        nn.init.constant_(self.nonlinear[0].weight, 2 * 3**0.5)
        nn.init.constant_(self.nonlinear[0].bias, -3**0.5)
        nn.init.constant_(self.nonlinear[2].weight, 1.0)
        nn.init.constant_(self.nonlinear[2].bias, 0.0)
        # The released MDTVSFA implementation keeps the output affine part
        # fixed at identity; the learned dataset-specific affine layers below
        # provide the subjective-score scale and offset.  The module still
        # exposes all four logistic parameters, matching the paper's stated
        # four-parameter mapping.
        self.nonlinear[2].weight.requires_grad_(False)
        self.nonlinear[2].bias.requires_grad_(False)

        self.alignment = nn.ModuleDict({name: nn.Linear(1, 1) for name in self.datasets})
        for name in self.datasets:
            layer = self.alignment[name]
            low, high = (score_ranges or {}).get(name, (0.0, 1.0))
            scale = max(float(high) - float(low), 1e-6)
            nn.init.constant_(layer.weight, scale)
            nn.init.constant_(layer.bias, float(low))

    @property
    def latent(self) -> nn.Module:
        """Return the shared relative-quality scorer for introspection."""
        return self.base

    def stages(
        self,
        vision: torch.Tensor,
        text: torch.Tensor | None = None,
        datasets: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_relative = self.base(vision) if text is None else self.base(vision, text)
        relative = torch.sigmoid(raw_relative)
        perceptual = self.nonlinear(relative.unsqueeze(-1)).squeeze(-1)
        aligned = perceptual.clone()
        if datasets is None:
            return relative, perceptual, aligned

        names = [str(name) for name in datasets]
        known = torch.tensor([self.index.get(name, -1) for name in names], device=perceptual.device)
        for name, index in self.index.items():
            mask = known.eq(index)
            if mask.any():
                aligned[mask] = self.alignment[name](perceptual[mask].unsqueeze(-1)).squeeze(-1)
        return relative, perceptual, aligned

    def forward(
        self,
        vision: torch.Tensor,
        text: torch.Tensor | None = None,
        datasets: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        # For an unseen dataset, return the shared perceptual stage.  For a
        # known training dataset, return its aligned subjective score.
        _, perceptual, aligned = self.stages(vision, text, datasets)
        return aligned if datasets is not None else perceptual


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
        mlp_layers: int = 1,
    ):
        super().__init__()
        self.interaction = interaction
        self.vision = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, fusion_dim))
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, fusion_dim))
        input_dim = fusion_dim * (3 if interaction else 2)
        self.score = make_score_mlp(input_dim, hidden_dim, mlp_layers, dropout)

    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        vision = self.vision(vision)
        text = self.text(text)
        features = [vision, text]
        if self.interaction:
            features.append(vision * text)
        return self.score(torch.cat(features, dim=-1)).squeeze(-1)


class MultiViewQualityHead(nn.Module):
    """Attention-pool frozen CLIP views into one image-only quality score."""

    requires_view_features = True

    def __init__(self, vision_dim: int, fusion_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.view = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, fusion_dim))
        self.weight = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.score = make_score_mlp(fusion_dim, hidden_dim, 1, dropout)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 3:
            raise ValueError(f"expected [batch, views, channels], got {tuple(views.shape)}")
        projected = self.view(views)
        weights = torch.softmax(self.weight(projected).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * projected, dim=1)
        return self.score(pooled).squeeze(-1)


class MultiViewTextFusionHead(nn.Module):
    """Text-conditioned quality prediction over a global view and local tiles."""

    requires_view_features = True

    def __init__(self, vision_dim: int, text_dim: int, fusion_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.view = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, fusion_dim))
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, fusion_dim))
        self.weight = nn.Sequential(nn.LayerNorm(fusion_dim * 3), nn.Linear(fusion_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.score = make_score_mlp(fusion_dim * 3, hidden_dim, 1, dropout)

    def forward(self, views: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if views.ndim != 3:
            raise ValueError(f"expected [batch, views, channels], got {tuple(views.shape)}")
        image = self.view(views)
        condition = self.text(text)
        condition_views = condition.unsqueeze(1).expand_as(image)
        features = torch.cat([image, condition_views, image * condition_views], dim=-1)
        weights = torch.softmax(self.weight(features).squeeze(-1), dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * image, dim=1)
        fused = torch.cat([pooled, condition, pooled * condition], dim=-1)
        return self.score(fused).squeeze(-1)


class MultiViewUniformTextFusionHead(nn.Module):
    """Text-conditioned quality prediction with uniform view pooling.

    This is the controlled counterpart to :class:`MultiViewTextFusionHead`:
    it keeps the visual/text projections and final interaction scorer but
    removes the learned quality-aware view weighting.  The ablation therefore
    measures whether learned view selection, rather than simply aggregating
    global and local features, is responsible for a gain.
    """

    requires_view_features = True

    def __init__(self, vision_dim: int, text_dim: int, fusion_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.view = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, fusion_dim))
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, fusion_dim))
        self.score = make_score_mlp(fusion_dim * 3, hidden_dim, 1, dropout)

    def forward(self, views: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if views.ndim != 3:
            raise ValueError(f"expected [batch, views, channels], got {tuple(views.shape)}")
        projected = self.view(views)
        pooled = projected.mean(dim=1)
        condition = self.text(text)
        fused = torch.cat([pooled, condition, pooled * condition], dim=-1)
        return self.score(fused).squeeze(-1)


class PooledVisionAdapter(nn.Module):
    """Small residual adapter for a frozen pooled CLIP representation."""

    def __init__(self, vision_dim: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.down = nn.Linear(vision_dim, bottleneck)
        self.up = nn.Linear(bottleneck, vision_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, vision: torch.Tensor) -> torch.Tensor:
        correction = self.up(torch.nn.functional.gelu(self.down(self.norm(vision))))
        return vision + self.scale * correction


class AdapterTextFusionHead(nn.Module):
    """Current interaction head preceded by a trainable pooled visual adapter."""

    def __init__(self, vision_dim: int, text_dim: int, fusion_dim: int = 256, hidden_dim: int = 256, bottleneck: int = 64, dropout: float = 0.1, mlp_layers: int = 1):
        super().__init__()
        self.adapter = PooledVisionAdapter(vision_dim, bottleneck)
        self.score = TextFusionHead(vision_dim, text_dim, fusion_dim, hidden_dim, interaction=True, dropout=dropout, mlp_layers=mlp_layers)

    @property
    def latent(self) -> nn.Module:
        return self

    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        return self.score(self.adapter(vision), text)


class PatchWeightedHead(nn.Module):
    """A learned normalized weighted mean of per-patch quality scores."""

    requires_patch_tokens = True

    def __init__(self, vision_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        def branch():
            return nn.Sequential(
                nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )

        self.patch_score = branch()
        self.patch_weight = branch()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3:
            raise ValueError(f"expected [batch, patches, channels], got {tuple(patches.shape)}")
        scores = self.patch_score(patches).squeeze(-1)
        weights = torch.softmax(self.patch_weight(patches).squeeze(-1), dim=1)
        return torch.sum(weights * scores, dim=1)


class TextPatchWeightedHead(nn.Module):
    """Text-conditioned patch scoring and soft attention over CLIP tokens."""

    requires_patch_tokens = True

    def __init__(
        self, vision_dim: int, text_dim: int, fusion_dim: int = 256,
        hidden_dim: int = 256, dropout: float = 0.1,
    ):
        super().__init__()
        self.patch = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, fusion_dim))
        self.text = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, fusion_dim))
        self.patch_score = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.patch_weight = nn.Sequential(
            nn.LayerNorm(fusion_dim * 3), nn.Linear(fusion_dim * 3, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, patches: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3:
            raise ValueError(f"expected [batch, patches, channels], got {tuple(patches.shape)}")
        patch = self.patch(patches)
        condition = self.text(text).unsqueeze(1).expand_as(patch)
        scores = self.patch_score(patch).squeeze(-1)
        weight_features = torch.cat([patch, condition, patch * condition], dim=-1)
        weights = torch.softmax(self.patch_weight(weight_features).squeeze(-1), dim=1)
        return torch.sum(weights * scores, dim=1)


class GlobalPatchResidualHead(nn.Module):
    """Global pooled quality plus a gated spatial patch correction."""

    requires_patch_tokens = True
    requires_pooled_features = True

    def __init__(self, vision_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.global_score = nn.Sequential(
            nn.LayerNorm(vision_dim), nn.Linear(vision_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.patch = PatchWeightedHead(vision_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, pooled: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        global_score = self.global_score(pooled).squeeze(-1)
        correction = self.patch(patches)
        return global_score + torch.sigmoid(self.gate(pooled).squeeze(-1)) * correction


class GlobalTextPatchResidualHead(nn.Module):
    """Row-14 global interaction scorer plus a gated conditioned patch correction."""

    requires_patch_tokens = True
    requires_pooled_features = True

    def __init__(self, vision_dim: int, text_dim: int, fusion_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.global_score = TextFusionHead(vision_dim, text_dim, fusion_dim, hidden_dim, interaction=True, dropout=dropout)
        self.patch = TextPatchWeightedHead(vision_dim, text_dim, fusion_dim, hidden_dim, dropout)
        self.gate = nn.Sequential(nn.LayerNorm(vision_dim + text_dim), nn.Linear(vision_dim + text_dim, 1))
        nn.init.constant_(self.gate[-1].bias, -3.0)

    def forward(self, pooled: torch.Tensor, patches: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        global_score = self.global_score(pooled, text)
        correction = self.patch(patches, text)
        gate = torch.sigmoid(self.gate(torch.cat([pooled, text], dim=-1)).squeeze(-1))
        return global_score + gate * correction


class GlobalTextCrossAttentionHead(nn.Module):
    """Global interaction score with text-query cross-attention over patches."""

    requires_patch_tokens = True
    requires_pooled_features = True

    def __init__(self, vision_dim: int, text_dim: int, fusion_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.global_score = TextFusionHead(vision_dim, text_dim, fusion_dim, hidden_dim, interaction=True, dropout=dropout)
        self.patch = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, fusion_dim))
        self.query = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, fusion_dim))
        self.attention = nn.MultiheadAttention(fusion_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.correction = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.gate = nn.Sequential(nn.LayerNorm(vision_dim + text_dim), nn.Linear(vision_dim + text_dim, 1))
        nn.init.constant_(self.gate[-1].bias, -3.0)

    def forward(self, pooled: torch.Tensor, patches: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        global_score = self.global_score(pooled, text)
        query = self.query(text).unsqueeze(1)
        tokens = self.patch(patches)
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        correction = self.correction(attended.squeeze(1)).squeeze(-1)
        gate = torch.sigmoid(self.gate(torch.cat([pooled, text], dim=-1)).squeeze(-1))
        return global_score + gate * correction


class FiLMTextHead(nn.Module):
    """Feature-wise linear modulation of image features by frozen text."""

    def __init__(self, vision_dim: int, text_dim: int, fusion_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.vision = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, fusion_dim))
        self.condition = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, fusion_dim * 2))
        self.score = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image = self.vision(vision)
        scale, shift = self.condition(text).chunk(2, dim=-1)
        modulated = image * (1 + 0.1 * torch.tanh(scale)) + shift
        return self.score(modulated).squeeze(-1)


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
