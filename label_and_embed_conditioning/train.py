"""Train an MLP on frozen CLIP features to predict image quality.

    python prepare_data.py ~/iqa-data/kadid10k          # once, writes labels.csv
    python train.py --data ~/iqa-data/kadid10k/labels.csv
    python train.py --data ~/iqa-data/kadid10k/labels.csv --sampler balanced

    image -> frozen CLIP -> pooled embedding -> MLP -> quality score

Pass ``--conditioning label`` to condition the quality head on a small learned
embedding of the prepared CSV's distortion ``group``. ``--label-fusion``
selects concatenation, additive fusion, input/hidden FiLM, residual gating,
parameter-matched low-rank modulation, or label-guided patch attention. The
frozen image backbone and image resolution stay unchanged.

The backbone never trains; only the MLP does, which is a few hundred
thousand parameters over a representation that costs nothing to keep. That
makes this the row every other design is measured against: if a change does
not beat it, the change is not doing anything.

Reports SRCC and PLCC on the held-out split each epoch, one row per dataset.
After training it prints the macro validation result from the best epoch.
SRCC is the number IQA papers report
— it only cares about ranking, which is what a quality metric is for.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from arniqa import (
    ARNIQA_CROP_SIZE,
    ARNIQA_FEATURE_DIM,
    embed_arniqa,
    load_arniqa_encoder,
)
from dataset import IQADataset, make_sampler, split_by
from hf_mirror_utils import load_transformers_model_from_mirrors
from prepare_data import GROUPS

BACKBONES = {
    "clip-base": ("openai/clip-vit-base-patch16", 224),
    "clip-large": ("openai/clip-vit-large-patch14-336", 336),
    "siglip": ("google/siglip-large-patch16-256", 256),
    "siglip2-base": ("google/siglip2-base-patch16-224", 224),
    "siglip2-large": ("google/siglip2-large-patch16-256", 256),
}

UNKNOWN_GROUP = "<unknown>"
CONDITION_GROUPS = (*GROUPS, UNKNOWN_GROUP)
GROUP_TO_ID = {name: index for index, name in enumerate(CONDITION_GROUPS)}


class QualityMLP(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear -> one number."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class LabelConditionedQualityMLP(nn.Module):
    """Quality head conditioned on a learned coarse distortion-group label.

    Only ``label_dim * (hidden_dim + number of groups)`` parameters are added
    over :class:`QualityMLP` (8,480 with the defaults). During training some
    label embeddings are zeroed, making the zero-condition evaluation a
    pathway the model has actually learned rather than an out-of-distribution
    input.
    """

    def __init__(
        self,
        input_dim: int,
        num_groups: int,
        hidden_dim: int = 256,
        label_dim: int = 32,
        dropout: float = 0.1,
        condition_dropout: float = 0.1,
    ):
        super().__init__()
        if label_dim < 1:
            raise ValueError("label_dim must be positive")
        if not 0.0 <= condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be between 0 and 1")
        self.feature_norm = nn.LayerNorm(input_dim)
        self.group_embedding = nn.Embedding(num_groups, label_dim)
        # PyTorch's default embedding initialization has unit variance, much
        # larger than a normalized CLIP coordinate. Start labels gently and
        # let training determine how strongly the condition should act.
        nn.init.normal_(self.group_embedding.weight, mean=0.0, std=0.02)
        self.condition_dropout = condition_dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        group_ids: torch.Tensor,
        *,
        zero_condition: bool = False,
    ) -> torch.Tensor:
        condition = self.group_embedding(group_ids)
        if zero_condition:
            condition = torch.zeros_like(condition)
        elif self.training and self.condition_dropout > 0:
            keep = torch.rand(condition.shape[0], 1, device=condition.device)
            condition = condition * (keep >= self.condition_dropout)
        fused = torch.cat((self.feature_norm(features), condition), dim=-1)
        return self.net(fused).squeeze(-1)


class _LabelConditionedHead(nn.Module):
    """Shared label embedding and condition-dropout behavior."""

    def __init__(self, num_groups: int, label_dim: int, condition_dropout: float):
        super().__init__()
        if label_dim < 1:
            raise ValueError("label_dim must be positive")
        if not 0.0 <= condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be between 0 and 1")
        self.group_embedding = nn.Embedding(num_groups, label_dim)
        nn.init.normal_(self.group_embedding.weight, mean=0.0, std=0.02)
        self.condition_dropout = condition_dropout

    def _condition(
        self,
        group_ids: torch.Tensor,
        zero_condition: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.group_embedding(group_ids)
        present = torch.ones(
            condition.shape[0], 1, device=condition.device, dtype=condition.dtype,
        )
        if zero_condition:
            present.zero_()
        elif self.training and self.condition_dropout > 0:
            present = (
                torch.rand(condition.shape[0], 1, device=condition.device)
                >= self.condition_dropout
            ).to(condition.dtype)
        return condition * present, present


def _matched_hidden_dim(
    input_dim: int,
    requested_hidden_dim: int,
    extra_fixed_parameters: int,
    extra_parameters_per_hidden_unit: int = 0,
) -> int:
    """Spend a baseline MLP's parameter budget on a conditioned head.

    requested_hidden_dim describes the baseline being matched, rather than
    blindly widening the shared trunk. Tiny dimensions used by unit tests may
    not have enough budget for the fixed conditioning layers; those retain one
    hidden unit so the module remains usable.
    """
    baseline_budget = (
        2 * input_dim + 1 + requested_hidden_dim * (input_dim + 2)
    )
    conditioned_fixed = 2 * input_dim + 1 + extra_fixed_parameters
    parameters_per_hidden = input_dim + 2 + extra_parameters_per_hidden_unit
    return max(
        1,
        (baseline_budget - conditioned_fixed) // parameters_per_hidden,
    )


class AdditiveLabelConditionedQualityMLP(_LabelConditionedHead):
    """Project the label to feature space and add it before a shared head."""

    def __init__(
        self, input_dim: int, num_groups: int, hidden_dim: int = 256,
        label_dim: int = 32, dropout: float = 0.1, condition_dropout: float = 0.1,
    ):
        super().__init__(num_groups, label_dim, condition_dropout)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.condition_projection = nn.Linear(label_dim, input_dim, bias=False)
        nn.init.zeros_(self.condition_projection.weight)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, group_ids, *, zero_condition: bool = False):
        condition, _ = self._condition(group_ids, zero_condition)
        fused = self.feature_norm(features) + self.condition_projection(condition)
        return self.net(fused).squeeze(-1)


class InputFiLMLabelConditionedQualityMLP(_LabelConditionedHead):
    """Apply identity-initialized FiLM to normalized frozen image features."""

    def __init__(
        self, input_dim: int, num_groups: int, hidden_dim: int = 256,
        label_dim: int = 32, dropout: float = 0.1, condition_dropout: float = 0.1,
    ):
        super().__init__(num_groups, label_dim, condition_dropout)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.film = nn.Linear(label_dim, 2 * input_dim, bias=False)
        nn.init.zeros_(self.film.weight)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, group_ids, *, zero_condition: bool = False):
        condition, _ = self._condition(group_ids, zero_condition)
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        normalized = self.feature_norm(features)
        fused = normalized * (1.0 + gamma) + beta
        return self.net(fused).squeeze(-1)


class HiddenFiLMLabelConditionedQualityMLP(_LabelConditionedHead):
    """Apply identity-initialized FiLM after the first hidden ReLU layer."""

    def __init__(
        self, input_dim: int, num_groups: int, hidden_dim: int = 256,
        label_dim: int = 32, dropout: float = 0.1, condition_dropout: float = 0.1,
    ):
        super().__init__(num_groups, label_dim, condition_dropout)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.film = nn.Linear(label_dim, 2 * hidden_dim, bias=False)
        nn.init.zeros_(self.film.weight)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, features, group_ids, *, zero_condition: bool = False):
        condition, _ = self._condition(group_ids, zero_condition)
        hidden = torch.relu(self.hidden(self.feature_norm(features)))
        gamma, beta = self.film(condition).chunk(2, dim=-1)
        modulated = hidden * (1.0 + gamma) + beta
        return self.output(self.dropout(modulated)).squeeze(-1)


class ResidualGatedLabelConditionedQualityMLP(_LabelConditionedHead):
    """Add a sigmoid-gated label-dependent correction to a baseline head."""

    def __init__(
        self, input_dim: int, num_groups: int, hidden_dim: int = 256,
        label_dim: int = 32, dropout: float = 0.1, condition_dropout: float = 0.1,
        gate_bias: float = -4.0,
    ):
        super().__init__(num_groups, label_dim, condition_dropout)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.base_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Linear(label_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, features, group_ids, *, zero_condition: bool = False):
        condition, present = self._condition(group_ids, zero_condition)
        normalized = self.feature_norm(features)
        baseline = self.base_head(normalized)
        delta = self.delta_head(torch.cat((normalized, condition), dim=-1))
        gate = torch.sigmoid(self.gate(condition)) * present
        return (baseline + gate * delta).squeeze(-1)


class PatchAttentionLabelConditionedQualityMLP(_LabelConditionedHead):
    """Use the label to select frozen CLIP patch evidence before scoring.

    Token zero is CLIP's pooled/class token and is the exact zero-condition
    fallback. The label attends only over spatial tokens and adds their
    deviation from uniform patch pooling as a gated residual. The shared MLP
    is narrowed automatically so the whole head does not exceed the requested
    baseline head's parameter budget.
    """

    requires_patch_tokens = True

    def __init__(
        self, input_dim: int, num_groups: int, hidden_dim: int = 256,
        label_dim: int = 32, dropout: float = 0.1,
        condition_dropout: float = 0.1,
    ):
        super().__init__(num_groups, label_dim, condition_dropout)
        self.feature_norm = nn.LayerNorm(input_dim)
        self.key_projection = nn.Linear(input_dim, label_dim, bias=False)
        self.attention_gain_logit = nn.Parameter(torch.tensor(-2.0))
        effective_hidden_dim = _matched_hidden_dim(
            input_dim,
            hidden_dim,
            extra_fixed_parameters=(
                num_groups * label_dim + input_dim * label_dim + 1
            ),
        )
        self.effective_hidden_dim = effective_hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, effective_hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(effective_hidden_dim, 1),
        )

    def forward(self, features, group_ids, *, zero_condition: bool = False):
        if features.ndim == 2:
            # Feature-only tests and cached pooled embeddings take the
            # unconditional path. Real runs receive B x tokens x D.
            tokens = features.unsqueeze(1)
        elif features.ndim == 3:
            tokens = features
        else:
            raise ValueError("patch attention expects B x D or B x tokens x D features")

        condition, present = self._condition(group_ids, zero_condition)
        normalized = self.feature_norm(tokens)
        pooled = normalized[:, 0]
        patches = normalized[:, 1:] if normalized.shape[1] > 1 else normalized

        keys = torch.nn.functional.normalize(
            self.key_projection(patches), dim=-1, eps=1e-6,
        )
        query = torch.nn.functional.normalize(condition, dim=-1, eps=1e-6)
        logits = torch.einsum("bnd,bd->bn", keys, query)
        attention = torch.softmax(logits, dim=1)
        attended = torch.einsum("bn,bnd->bd", attention, patches)
        uniform = patches.mean(dim=1)
        gain = torch.sigmoid(self.attention_gain_logit)
        fused = pooled + present * gain * (attended - uniform)
        return self.net(fused).squeeze(-1)


class LowRankHypernetworkLabelConditionedQualityMLP(_LabelConditionedHead):
    """Apply a label-generated low-rank update to the first head layer.

    The update is LoRA-like: image features are projected to a small rank,
    scaled by coefficients generated from the label, and projected into the
    hidden layer. Its output projection starts at zero, so training begins at
    the unconditional pathway. The base hidden width is reduced to match the
    parameter budget of the requested baseline MLP.
    """

    def __init__(
        self, input_dim: int, num_groups: int, hidden_dim: int = 256,
        label_dim: int = 32, dropout: float = 0.1,
        condition_dropout: float = 0.1, rank: int = 4,
    ):
        super().__init__(num_groups, label_dim, condition_dropout)
        if rank < 1:
            raise ValueError("low-rank hypernetwork rank must be positive")
        self.rank = rank
        self.feature_norm = nn.LayerNorm(input_dim)
        effective_hidden_dim = _matched_hidden_dim(
            input_dim,
            hidden_dim,
            extra_fixed_parameters=(
                num_groups * label_dim + input_dim * rank + label_dim * rank
            ),
            extra_parameters_per_hidden_unit=rank,
        )
        self.effective_hidden_dim = effective_hidden_dim
        self.hidden = nn.Linear(input_dim, effective_hidden_dim)
        self.feature_down = nn.Linear(input_dim, rank, bias=False)
        self.rank_coefficients = nn.Linear(label_dim, rank, bias=False)
        self.hidden_up = nn.Linear(rank, effective_hidden_dim, bias=False)
        nn.init.zeros_(self.hidden_up.weight)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(effective_hidden_dim, 1)

    def forward(self, features, group_ids, *, zero_condition: bool = False):
        condition, present = self._condition(group_ids, zero_condition)
        normalized = self.feature_norm(features)
        base_hidden = self.hidden(normalized)
        ranked_features = self.feature_down(normalized)
        coefficients = self.rank_coefficients(condition)
        update = self.hidden_up(ranked_features * coefficients)
        hidden = torch.nn.functional.gelu(base_hidden + present * update)
        return self.output(self.dropout(hidden)).squeeze(-1)


LABEL_FUSION_HEADS = {
    "concat": LabelConditionedQualityMLP,
    "additive": AdditiveLabelConditionedQualityMLP,
    "film_input": InputFiLMLabelConditionedQualityMLP,
    "film_hidden": HiddenFiLMLabelConditionedQualityMLP,
    "low_rank_hypernetwork": LowRankHypernetworkLabelConditionedQualityMLP,
    "patch_attention": PatchAttentionLabelConditionedQualityMLP,
    "residual_gate": ResidualGatedLabelConditionedQualityMLP,
}


def make_label_conditioned_head(
    label_fusion: str,
    input_dim: int,
    num_groups: int,
    *,
    hidden_dim: int,
    label_dim: int,
    dropout: float = 0.1,
    condition_dropout: float = 0.1,
    low_rank_dim: int = 4,
) -> nn.Module:
    """Construct a label head, including fusion-specific arguments."""
    head_class = LABEL_FUSION_HEADS[label_fusion]
    kwargs = {
        "hidden_dim": hidden_dim,
        "label_dim": label_dim,
        "dropout": dropout,
        "condition_dropout": condition_dropout,
    }
    if label_fusion == "low_rank_hypernetwork":
        kwargs["rank"] = low_rank_dim
    return head_class(input_dim, num_groups, **kwargs)


class ARNIQAConditionedQualityMLP(nn.Module):
    """Quality head conditioned on a frozen per-image ARNIQA embedding."""

    def __init__(
        self,
        input_dim: int,
        arniqa_dim: int = ARNIQA_FEATURE_DIM,
        hidden_dim: int = 256,
        condition_dim: int = 32,
        dropout: float = 0.1,
        condition_dropout: float = 0.1,
    ):
        super().__init__()
        if condition_dim < 1:
            raise ValueError("condition_dim must be positive")
        if not 0.0 <= condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be between 0 and 1")
        self.feature_norm = nn.LayerNorm(input_dim)
        self.condition_norm = nn.LayerNorm(arniqa_dim)
        self.condition_projection = nn.Sequential(
            nn.Linear(arniqa_dim, condition_dim),
            nn.GELU(),
        )
        self.condition_dropout = condition_dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        arniqa_features: torch.Tensor,
        *,
        zero_condition: bool = False,
    ) -> torch.Tensor:
        condition = self.condition_projection(self.condition_norm(arniqa_features))
        if zero_condition:
            condition = torch.zeros_like(condition)
        elif self.training and self.condition_dropout > 0:
            keep = torch.rand(condition.shape[0], 1, device=condition.device)
            condition = condition * (keep >= self.condition_dropout)
        fused = torch.cat((self.feature_norm(features), condition), dim=-1)
        return self.net(fused).squeeze(-1)


def encode_groups(groups: list[str] | tuple[str, ...], device: torch.device | None = None) -> torch.Tensor:
    """Map prepared-data group names to stable label-embedding indices."""
    ids = []
    for group in groups:
        name = str(group).strip().lower()
        # Accept the spelling used in prose while keeping the CSV vocabulary
        # and saved checkpoint mapping canonical ("color").
        if name == "colour":
            name = "color"
        ids.append(GROUP_TO_ID.get(name, GROUP_TO_ID[UNKNOWN_GROUP]))
    return torch.tensor(ids, dtype=torch.long, device=device)


def load_backbone(name: str, weights: str | None, device: torch.device):
    """The frozen encoder. `weights` is a local directory, if you have one."""
    from transformers import CLIPVisionModel, SiglipVisionModel

    hf_id, image_size = BACKBONES[name]
    model_class = SiglipVisionModel if name.startswith("siglip") else CLIPVisionModel
    if weights:
        model = model_class.from_pretrained(weights, local_files_only=True)
    else:
        model = load_transformers_model_from_mirrors(model_class, hf_id)
    model = model.eval().requires_grad_(False).to(device)
    return model, image_size, model.config.hidden_size


@torch.no_grad()
def embed(
    backbone,
    images: torch.Tensor,
    *,
    patch_tokens: bool = False,
) -> torch.Tensor:
    """Extract the frozen pooled feature or its full token sequence."""
    outputs = backbone(pixel_values=images)
    if not patch_tokens:
        return outputs.pooler_output.float()

    tokens = outputs.last_hidden_state
    vision_model = getattr(backbone, "vision_model", None)
    post_layernorm = getattr(vision_model, "post_layernorm", None)
    if post_layernorm is not None:
        # Expose spatial tokens under the encoder's frozen final
        # normalization, which CLIP otherwise applies only to its class token.
        tokens = post_layernorm(tokens)
    model_type = str(getattr(getattr(backbone, "config", None), "model_type", ""))
    if "clip" in model_type and tokens.shape[1] > 1:
        tokens = tokens[:, 1:]
    # Always prepend the model's actual pooled output. This gives CLIP and
    # SigLIP the same exact zero-condition fallback while the remaining tokens
    # carry spatial evidence.
    return torch.cat((outputs.pooler_output.unsqueeze(1), tokens), dim=1).float()


def _synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _reset_peak_memory(device: torch.device) -> None:
    """Start a fresh CUDA peak-allocation measurement for evaluation."""
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float | None:
    """Peak allocated accelerator memory, or None when unavailable."""
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def _quality_metrics(predictions, targets, references, datasets) -> dict:
    """Compute the repository's dataset-level and within-reference metrics."""
    p, t = np.concatenate(predictions), np.concatenate(targets)
    frame = pd.DataFrame({"p": p, "t": t, "ref": references, "dataset": datasets})

    per_dataset = {}
    for name, group in frame.groupby("dataset"):
        if len(group) < 2 or group["t"].nunique() < 2:
            continue
        per_dataset[name] = {
            "srcc": float(stats.spearmanr(group["p"], group["t"]).correlation),
            "plcc": float(stats.pearsonr(group["p"], group["t"]).statistic),
            "n": int(len(group)),
        }

    per_reference = []
    for _, group in frame.groupby("ref"):
        if len(group) >= 8 and group["t"].nunique() > 1:
            per_reference.append(stats.spearmanr(group["p"], group["t"]).correlation)

    srccs = [scores["srcc"] for scores in per_dataset.values()]
    return {
        "per_dataset": per_dataset,
        "macro_srcc": float(np.mean(srccs)) if srccs else None,
        "macro_plcc": float(np.mean([s["plcc"] for s in per_dataset.values()])) if srccs else None,
        "worst_srcc": float(min(srccs)) if srccs else None,
        "worst_dataset": min(per_dataset, key=lambda k: per_dataset[k]["srcc"]) if srccs else None,
        "srcc_per_reference": float(np.mean(per_reference)) if per_reference else None,
        "n_references": len(per_reference),
    }


def _evaluation_progress_name(loader) -> str:
    """Short tqdm label for a single prepared dataset or combined split."""
    rows = getattr(loader.dataset, "rows", None)
    if rows is not None and "dataset" in rows:
        names = rows["dataset"].dropna().astype(str).unique()
        if len(names) == 1 and names[0]:
            return f"evaluate {names[0]}"
        if len(names) > 1:
            return f"evaluate {len(names)} datasets"
    return "evaluate"


def evaluate(
    backbone,
    head,
    loader,
    device,
    conditioned: bool = False,
    seed: int = 0,
    *,
    conditioning: str | None = None,
    arniqa_encoder=None,
    arniqa_batch_size: int = 64,
) -> dict:
    """SRCC and PLCC per dataset, their macro, and SRCC within each reference.

    Per dataset rather than pooled, because pooling measures something else.
    Two releases put their subjects on different scales and score different
    pictures, so a correlation over the union partly measures the offset
    between them: on a KADID + KonIQ + SPAQ run the pooled SRCC came out at
    0.766, below every one of the three sets it is made of. The macro is the
    mean of the per-dataset numbers and `worst` is the lowest of them — a mean
    alone hides a collapse on one set.

    The second number exists because of PIPAL. Its scores are Elo ratings
    from pairwise comparisons, and every image starts at 1400 — so a score
    says how a restoration ranks against other restorations *of the same
    picture*, not how good the picture is. Measured on the data: 99.9% of
    the variance sits inside a reference, and the 200 reference means span
    22 points against a 622-point spread within one. Correlating across
    references mixes two different questions; averaging the per-reference
    correlations asks only the one the ratings can answer.
    """
    if conditioning is None:
        conditioning = "label" if conditioned else "none"
    if conditioning not in {"none", "label", "arniqa"}:
        raise ValueError(f"unsupported conditioning mode: {conditioning!r}")
    if conditioning == "arniqa" and arniqa_encoder is None:
        raise ValueError("ARNIQA evaluation requires a frozen ARNIQA encoder")

    head.eval()
    if conditioning == "label":
        modes = ("correct", "shuffled", "wrong", "zeroed")
    elif conditioning == "arniqa":
        modes = ("correct", "shuffled", "zeroed")
    else:
        modes = ("correct",)
    predictions = {mode: [] for mode in modes}
    targets, references, datasets = [], [], []
    stored_features, stored_conditions = [], []
    latency_ms_per_image = []
    inference_seconds = 0.0
    inference_images = 0

    shuffled_groups = None
    if conditioning == "label":
        # Validation loaders in this script are sequential. Building one
        # permutation over the full split avoids the weak "shuffle" produced
        # by independently permuting small or single-group batches.
        rows = loader.dataset.rows
        group_names = rows["group"].fillna("").astype(str).tolist()
        all_group_ids = encode_groups(group_names)
        generator = torch.Generator().manual_seed(seed)
        shuffled_groups = all_group_ids[torch.randperm(len(all_group_ids), generator=generator)]

    offset = 0
    _reset_peak_memory(device)
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=_evaluation_progress_name(loader),
            unit="batch",
            dynamic_ncols=True,
        ):
            batch_size = len(batch["target"])
            _synchronize(device)
            started = time.perf_counter()
            features = embed(
                backbone,
                batch["image"].to(device),
                patch_tokens=getattr(head, "requires_patch_tokens", False),
            )
            if conditioning == "label":
                group_ids = encode_groups(batch["group"], device)
                correct_prediction = head(features, group_ids)
                _synchronize(device)
                elapsed = time.perf_counter() - started
                predictions["correct"].append(correct_prediction.cpu().numpy())

                # Diagnostic interventions reuse the timed image features but
                # are deliberately outside the normal-path latency sample.
                permuted_ids = shuffled_groups[offset:offset + batch_size].to(device)
                # Rotate among real groups so an oracle label is replaced by
                # another meaningful but wrong group, never by <unknown>.
                wrong_ids = torch.where(
                    group_ids < len(GROUPS),
                    (group_ids + 1) % len(GROUPS),
                    torch.zeros_like(group_ids),
                )
                predictions["shuffled"].append(head(features, permuted_ids).cpu().numpy())
                predictions["wrong"].append(head(features, wrong_ids).cpu().numpy())
                predictions["zeroed"].append(
                    head(features, group_ids, zero_condition=True).cpu().numpy()
                )
                offset += batch_size
            elif conditioning == "arniqa":
                condition = embed_arniqa(
                    arniqa_encoder,
                    batch["arniqa_image"].to(device),
                    batch["arniqa_image_ds"].to(device),
                    chunk_size=arniqa_batch_size,
                )
                correct_prediction = head(features, condition)
                _synchronize(device)
                elapsed = time.perf_counter() - started
                predictions["correct"].append(correct_prediction.cpu().numpy())
                # Keep frozen features on CPU so whole-dataset shuffling can
                # reuse both encoders without a second image pass.
                stored_features.append(features.cpu())
                stored_conditions.append(condition.cpu())
            else:
                correct_prediction = head(features)
                _synchronize(device)
                elapsed = time.perf_counter() - started
                predictions["correct"].append(correct_prediction.cpu().numpy())
            inference_seconds += elapsed
            inference_images += batch_size
            latency_ms_per_image.extend([1000.0 * elapsed / batch_size] * batch_size)
            targets.append(batch["target"].numpy())
            references.extend(batch["reference"])
            datasets.extend(batch["dataset"])

    if conditioning == "arniqa":
        all_features = torch.cat(stored_features)
        all_conditions = torch.cat(stored_conditions)
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(all_conditions), generator=generator)
        for start in range(0, len(all_features), arniqa_batch_size):
            stop = start + arniqa_batch_size
            features = all_features[start:stop].to(device)
            condition = all_conditions[permutation[start:stop]].to(device)
            predictions["shuffled"].append(
                head(features, condition).detach().cpu().numpy()
            )
            predictions["zeroed"].append(
                head(features, condition, zero_condition=True).detach().cpu().numpy()
            )
    head.train()
    results = {
        mode: _quality_metrics(values, targets, references, datasets)
        for mode, values in predictions.items()
    }
    scores = results.pop("correct")
    scores.update({
        "latency_p50_ms": float(np.percentile(latency_ms_per_image, 50)),
        "latency_p95_ms": float(np.percentile(latency_ms_per_image, 95)),
        "peak_memory_mb": _peak_memory_mb(device),
        "images_per_second": float(inference_images / inference_seconds),
    })
    if results:
        scores["condition_ablations"] = results
    return scores


def _checkpoint_state(head, args, feature_dim: int, epoch: int, scores: dict) -> dict:
    """Build a self-describing head checkpoint for training or evaluation."""
    return {
        "head": head.state_dict(),
        "backbone": args.backbone,
        "feature_dim": feature_dim,
        "hidden_dim": args.hidden_dim,
        "conditioning": args.conditioning,
        "label_fusion": args.label_fusion if args.conditioning == "label" else None,
        "label_dim": args.label_dim if args.conditioning == "label" else None,
        "low_rank_dim": (
            args.low_rank_dim
            if args.conditioning == "label"
            and args.label_fusion == "low_rank_hypernetwork"
            else None
        ),
        "condition_dim": (
            args.condition_dim if args.conditioning == "arniqa" else None
        ),
        "arniqa_feature_dim": (
            ARNIQA_FEATURE_DIM if args.conditioning == "arniqa" else None
        ),
        "arniqa_crop_size": (
            ARNIQA_CROP_SIZE if args.conditioning == "arniqa" else None
        ),
        "condition_dropout": (
            args.condition_dropout
            if args.conditioning in {"label", "arniqa"} else None
        ),
        "permuted_training_labels": (
            args.permute_training_labels if args.conditioning == "label" else False
        ),
        "permuted_training_conditions": (
            args.permute_training_conditions
            if args.conditioning == "arniqa" else False
        ),
        "groups": list(CONDITION_GROUPS) if args.conditioning == "label" else None,
        "epoch": epoch,
        "validation": {
            "macro_srcc": scores["macro_srcc"],
            "macro_plcc": scores["macro_plcc"],
            "per_dataset": scores["per_dataset"],
        },
    }


def _checkpoint_paths(save_dir: str, name: str) -> tuple[Path, Path]:
    """Return the best/last checkpoint paths for an experiment name."""
    directory = Path(save_dir).expanduser()
    return directory / f"{name}_best.pth", directory / f"{name}_last.pth"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the CSV prepare_data.py wrote")
    ap.add_argument("--backbone", default="clip-base", choices=sorted(BACKBONES))
    ap.add_argument("--weights", default=None, help="local checkpoint directory, if not the hub")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--conditioning", default="none", choices=["none", "label", "arniqa"],
                    help="condition the quality head on a group label or frozen ARNIQA embedding")
    ap.add_argument("--label-dim", type=int, default=32,
                    help="learned group-embedding size for --conditioning label")
    ap.add_argument(
        "--label-fusion", default="concat", choices=sorted(LABEL_FUSION_HEADS),
        help="where/how to inject the learned label embedding",
    )
    ap.add_argument(
        "--low-rank-dim", type=int, default=4,
        help="rank of the parameter-matched low-rank label hypernetwork",
    )
    ap.add_argument("--condition-dim", type=int, default=32,
                    help="ARNIQA bottleneck size for --conditioning arniqa")
    ap.add_argument("--condition-dropout", type=float, default=0.1,
                    help="fraction of training conditions zeroed to learn an unconditional fallback")
    ap.add_argument("--permute-training-labels", action="store_true",
                    help="control run: permute group labels across training images")
    ap.add_argument("--permute-training-conditions", action="store_true",
                    help="control run: pair training images with globally permuted ARNIQA conditions")
    ap.add_argument("--arniqa-weights", default=None,
                    help="local official ARNIQA.pth encoder; downloads it when omitted")
    ap.add_argument("--arniqa-batch-size", type=int, default=64,
                    help="maximum full/half ARNIQA crops encoded at once")
    ap.add_argument("--split", default="reference", choices=["reference", "random"])
    ap.add_argument("--score-column", default="scaled_subjective_score",
                    help="which column of the CSV to regress")
    ap.add_argument("--sampler", default="random",
                    choices=["random", "balanced", "by_level", "by_dataset"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N training images, drawn at random")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default="./weights",
                    help="directory for best and last checkpoints")
    ap.add_argument("--name", default="model",
                    help="checkpoint stem; writes NAME_best.pth and NAME_last.pth")
    args = ap.parse_args()

    if args.permute_training_labels and args.conditioning != "label":
        ap.error("--permute-training-labels requires --conditioning label")
    if args.permute_training_conditions and args.conditioning != "arniqa":
        ap.error("--permute-training-conditions requires --conditioning arniqa")
    if args.arniqa_batch_size < 1:
        ap.error("--arniqa-batch-size must be positive")
    if args.low_rank_dim < 1:
        ap.error("--low-rank-dim must be positive")
    if args.epochs < 1:
        ap.error("--epochs must be at least 1")
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        ap.error("--name must be a filename stem, not a path")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device != "auto"
        else "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    backbone, image_size, feature_dim = load_backbone(args.backbone, args.weights, device)
    arniqa_encoder = (
        load_arniqa_encoder(args.arniqa_weights, device)
        if args.conditioning == "arniqa" else None
    )
    family = "siglip" if args.backbone.startswith("siglip") else "clip"

    print("image_size: ", image_size)
    dataset = IQADataset(
        args.data,
        image_size=image_size,
        backbone=family,
        score_column=args.score_column,
        arniqa=args.conditioning == "arniqa",
        arniqa_crop_size=ARNIQA_CROP_SIZE,
    )
    train_set, val_set = split_by(dataset, args.split, fraction=0.2, seed=args.seed)
    if args.limit and args.limit < len(train_set.rows):
        # Sampled, not the first N rows: the CSV is ordered by dataset and then
        # by reference, so a head() would train on one dataset and a handful of
        # its pictures without saying so.
        train_set = train_set.subset(
            train_set.rows.sample(args.limit, random_state=args.seed)
        )

    if args.permute_training_labels:
        if "group" not in train_set.rows:
            ap.error("--conditioning label requires a 'group' column; rerun prepare_data.py")
        train_set.rows = train_set.rows.copy()
        order = np.random.default_rng(args.seed).permutation(len(train_set.rows))
        train_set.rows["group"] = train_set.rows["group"].to_numpy()[order]
    if args.permute_training_conditions:
        train_set.permute_arniqa_conditions(args.seed)

    sampler = make_sampler(train_set, args.sampler, seed=args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=args.workers,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    if args.conditioning == "label":
        if "group" not in train_set.rows:
            raise ValueError(
                "--conditioning label requires a 'group' column; rerun prepare_data.py"
            )
        normalized_groups = (
            train_set.rows["group"].fillna("").astype(str).str.strip().str.lower()
            .replace({"colour": "color"})
        )
        unknown = ~normalized_groups.isin(GROUPS)
        if unknown.any():
            print(f"warning: {int(unknown.sum())} training rows have no canonical group; "
                  f"using {UNKNOWN_GROUP}")
        head = make_label_conditioned_head(
            args.label_fusion,
            feature_dim,
            len(CONDITION_GROUPS),
            hidden_dim=args.hidden_dim,
            label_dim=args.label_dim,
            condition_dropout=args.condition_dropout,
            low_rank_dim=args.low_rank_dim,
        ).to(device)
    elif args.conditioning == "arniqa":
        head = ARNIQAConditionedQualityMLP(
            feature_dim,
            ARNIQA_FEATURE_DIM,
            args.hidden_dim,
            args.condition_dim,
            condition_dropout=args.condition_dropout,
        ).to(device)
    else:
        head = QualityMLP(feature_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    print(f"{args.backbone} at {image_size}px on {device}, {feature_dim}-d features")
    print(f"train {len(train_set)}  held out {len(val_set)}  "
          f"(split by {args.split}, sampling {args.sampler})")
    if args.conditioning == "label":
        print(f"conditioning: learned {args.label_dim}-d group label via "
              f"{args.label_fusion} (dropout {args.condition_dropout:g})")
        if hasattr(head, "effective_hidden_dim"):
            print(
                f"capacity match: requested baseline width {args.hidden_dim}, "
                f"effective shared width {head.effective_hidden_dim}"
            )
        if args.permute_training_labels:
            print("control: training group labels are permuted across images")
    elif args.conditioning == "arniqa":
        print(
            f"conditioning: frozen {ARNIQA_FEATURE_DIM}-d ARNIQA embedding -> "
            f"{args.condition_dim}-d bottleneck (dropout {args.condition_dropout:g})"
        )
        if args.permute_training_conditions:
            print("control: training ARNIQA conditions are permuted across images")
    print(f"{sum(p.numel() for p in head.parameters()):,} trainable parameters "
          "— the backbone is frozen")

    best_path, last_path = _checkpoint_paths(args.save_dir, args.name)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_epoch = -1
    best_scores = None

    for epoch in range(args.epochs):
        losses = []
        for batch in train_loader:
            print(f"epoch {epoch} batch {len(losses)} of {len(train_loader)}; device: {device}", flush=True)
            features = embed(
                backbone,
                batch["image"].to(device),
                patch_tokens=getattr(head, "requires_patch_tokens", False),
            )
            if args.conditioning == "label":
                group_ids = encode_groups(batch["group"], device)
                prediction = head(features, group_ids)
            elif args.conditioning == "arniqa":
                condition = embed_arniqa(
                    arniqa_encoder,
                    batch["arniqa_image"].to(device),
                    batch["arniqa_image_ds"].to(device),
                    chunk_size=args.arniqa_batch_size,
                )
                prediction = head(features, condition)
            else:
                prediction = head(features)
            loss = loss_fn(prediction, batch["target"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        scores = evaluate(
            backbone, head, val_loader, device,
            conditioned=args.conditioning != "none",
            seed=args.seed,
            conditioning=args.conditioning,
            arniqa_encoder=arniqa_encoder,
            arniqa_batch_size=args.arniqa_batch_size,
        )
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}", flush=True)
        for name, row in sorted(scores["per_dataset"].items()):
            print(f"    {name:<14s} n {row['n']:>6d}   "
                  f"SRCC {row['srcc']:.4f}   PLCC {row['plcc']:.4f}")
        if scores["srcc_per_reference"] is not None:
            print(f"    {'within-ref':<14s} {'':>8s}   SRCC {scores['srcc_per_reference']:.4f}"
                  f"   ({scores['n_references']} references)")
        if "condition_ablations" in scores:
            print("    condition ablations (same images and frozen features):")
            for mode, mode_scores in scores["condition_ablations"].items():
                print(f"      {mode:<10s} macro SRCC {mode_scores['macro_srcc']:.4f}   "
                      f"PLCC {mode_scores['macro_plcc']:.4f}")
        peak_memory = (
            f"{scores['peak_memory_mb']:.1f} MB"
            if scores["peak_memory_mb"] is not None else "N/A"
        )
        print(
            f"    performance: latency p50 {scores['latency_p50_ms']:.3f} ms/img   "
            f"p95 {scores['latency_p95_ms']:.3f} ms/img   "
            f"peak memory {peak_memory}   "
            f"throughput {scores['images_per_second']:.2f} img/s"
        )

        current_srcc = scores["macro_srcc"]
        best_srcc = None if best_scores is None else best_scores["macro_srcc"]
        if (
            best_scores is None
            or np.isfinite(current_srcc)
            and (best_srcc is None or not np.isfinite(best_srcc) or current_srcc > best_srcc)
        ):
            best_epoch = epoch
            best_scores = scores
            torch.save(
                _checkpoint_state(head, args, feature_dim, epoch, scores), best_path,
            )

    torch.save(
        _checkpoint_state(head, args, feature_dim, args.epochs - 1, scores), last_path,
    )
    print(
        f"average validation (best epoch {best_epoch}): "
        f"SRCC {best_scores['macro_srcc']:.4f}   PLCC {best_scores['macro_plcc']:.4f}"
    )
    print(f"saved best -> {best_path}")
    print(f"saved last -> {last_path}")


if __name__ == "__main__":
    main()
