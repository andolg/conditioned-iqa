"""Run pooled text-conditioned IQA experiments without changing ``train.py``.

The vision backbone and text tower are frozen. Methods are ``baseline`` for the
unconditioned pooled MLP, ``concat`` for image/text concatenation, and
``interaction`` for concatenation plus elementwise image-text interaction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader

from dataset import make_sampler, split_by
from result_reporting import (
    ResultReporter,
    add_reporting_arguments,
    measure_flops,
    measure_latency_memory,
    size_megabytes,
)
from text_conditioning.data import (
    ConditionedIQADataset,
    FeatureDataset,
    dataset_score_ranges,
)
from text_conditioning.models import (
    DatasetScaleHead,
    MDTVSFAHead,
    PatchWeightedHead,
    GlobalPatchResidualHead,
    GlobalTextPatchResidualHead,
    GlobalTextCrossAttentionHead,
    MultiViewQualityHead,
    MultiViewTextFusionHead,
    MultiViewUniformTextFusionHead,
    AdapterTextFusionHead,
    FiLMTextHead,
    ResidualTextHead,
    TextPatchWeightedHead,
    TextFusionHead,
)
from text_conditioning.prompts import (
    GENERIC_PROMPT,
    GROUP_PROMPTS,
    GROUPS,
    HELD_OUT_PARAPHRASES,
    TRAINING_GROUP_PROMPTS,
    wrong_group,
)
from text_conditioning.text_encoder import encode_prompts, load_frozen_text_encoder
from train import BACKBONES, MLflowTracker, QualityMLP, embed, embed_patches, load_backbone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="clip-base", choices=sorted(BACKBONES))
    parser.add_argument("--weights", default=None, help="local frozen vision checkpoint")
    parser.add_argument("--text-weights", default=None, help="local frozen text checkpoint; defaults to --weights")
    parser.add_argument("--text-encoder-id", default=None,
                        help="optional external text encoder ID; downloads only via the mirror helper")
    parser.add_argument("--method", choices=["baseline", "concat", "interaction", "residual", "film", "patch_weighted", "patch_interaction", "global_patch_residual", "global_text_patch_residual", "global_text_cross_attention", "multiview_baseline", "multiview_interaction", "multiview_uniform_interaction", "adapter_interaction"], default="concat")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument(
        "--mlp-layers", type=int, default=1,
        help="number of hidden layers in the pooled score MLP (1 preserves the baseline)",
    )
    parser.add_argument("--condition-dropout", type=float, default=0.0,
                        help="probability of replacing a training condition with zero text")
    parser.add_argument("--train-paraphrases", action="store_true",
                        help="sample a training-only prompt paraphrase for each conditioned example")
    parser.add_argument("--paraphrase-consistency-weight", type=float, default=0.0,
                        help="weight of the second-paraphrase prediction consistency loss")
    parser.add_argument("--split", choices=["reference", "random"], default="reference")
    parser.add_argument(
        "--split-manifest", default=None,
        help=(
            "CSV with path and partition columns (train/validation). When supplied, "
            "it replaces the generated split and is checked for overlap or missing rows."
        ),
    )
    parser.add_argument("--score-column", default="scaled_subjective_score")
    parser.add_argument(
        "--preprocessing", choices=("stretch", "resize_center_crop", "multiscale"), default="stretch",
        help="image preprocessing; resize_center_crop matches CLIP's native preprocessing",
    )
    parser.add_argument("--sampler", choices=["random", "balanced", "by_level", "by_dataset"], default="random")
    parser.add_argument(
        "--dataset-objective", choices=("global", "mdtvsfa", "mdtvsfa_faithful"), default="global",
        help="global regression, the legacy calibration approximation, or faithful three-stage MDTVSFA",
    )
    parser.add_argument(
        "--dataset-loss-weighting", choices=("mean", "softmax"), default="mean",
        help="per-dataset loss aggregation for the MDTVSFA-style objective",
    )
    parser.add_argument("--ranking-weight", type=float, default=0.0,
                        help="weight for within-dataset pairwise ranking loss")
    parser.add_argument("--ranking-scope", choices=("dataset", "reference"), default="dataset",
                        help="pair candidates sharing a dataset or a reference")
    parser.add_argument("--pipal-ranking-only", action="store_true",
                        help="exclude PIPAL from regression and use within-reference ranking")
    parser.add_argument("--calibration-datasets", nargs="+", default=None,
                        help="dataset names for MDTVSFA calibration heads (defaults to training CSV)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-features", action="store_true",
                        help="precompute frozen vision features once and reuse them across epochs")
    parser.add_argument("--feature-cache-dir", default="runs/feature_cache")
    parser.add_argument("--exclude-groups", nargs="+", default=None,
                        help="remove these broad condition groups before splitting (E8)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", default="conditioned-iqa-text")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--config-dir", default="runs/configs")
    add_reporting_arguments(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    initial, _ = bootstrap.parse_known_args(argv)
    parser = build_parser()
    if initial.config:
        with Path(initial.config).expanduser().open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        if not isinstance(config, dict):
            parser.error("config must contain a YAML mapping")
        valid = {action.dest for action in parser._actions}
        parser.set_defaults(**{key: value for key, value in config.items() if key in valid})
    args = parser.parse_args(argv)
    if args.data is None:
        parser.error("--data is required, either directly or through --config")
    return args


def split_from_manifest(
    dataset: ConditionedIQADataset, manifest_path: str | Path,
) -> tuple[ConditionedIQADataset, ConditionedIQADataset]:
    """Use a predeclared train/validation split without admitting test rows.

    A manifest is needed when a source publishes official partitions (such as
    UHD-IQA).  It also makes paired baseline/conditioned runs use exactly the
    same examples rather than independently drawing a reference split.
    """
    manifest_path = Path(manifest_path).expanduser()
    manifest = pd.read_csv(manifest_path)
    required = {"path", "partition"}
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"split manifest {manifest_path} lacks columns: {', '.join(sorted(missing_columns))}"
        )
    manifest = manifest[["path", "partition"]].copy()
    manifest["path"] = manifest["path"].astype(str)
    manifest["partition"] = manifest["partition"].astype(str).str.lower()
    if manifest["path"].duplicated().any():
        raise ValueError(f"split manifest {manifest_path} contains duplicate image paths")
    allowed = {"train", "validation"}
    invalid = sorted(set(manifest["partition"]).difference(allowed))
    if invalid:
        raise ValueError(
            f"split manifest {manifest_path} has forbidden partitions {invalid}; "
            "only train and validation rows may enter the training runner"
        )

    rows = dataset.rows.copy()
    rows["path"] = rows["path"].astype(str)
    if rows["path"].duplicated().any():
        raise ValueError("training CSV contains duplicate image paths; cannot apply a safe split manifest")
    manifest_partitions = manifest.set_index("path")["partition"]
    partitions = rows["path"].map(manifest_partitions)
    if partitions.isna().any():
        examples = rows.loc[partitions.isna(), "path"].head(3).tolist()
        raise ValueError(f"split manifest is missing {partitions.isna().sum()} training rows, e.g. {examples}")
    extra_paths = set(manifest_partitions.index).difference(rows["path"])
    if extra_paths:
        raise ValueError(
            f"split manifest contains {len(extra_paths)} paths outside the training CSV; "
            "build a manifest only for the train/validation source table"
        )

    # Synthetic variants of a reference must not cross the boundary.  For
    # authentic datasets references are normally unique, so this is harmless.
    leak_check = rows.assign(partition=partitions).groupby(["dataset", "reference"])["partition"].nunique()
    if (leak_check > 1).any():
        leaked = leak_check[leak_check > 1].index[0]
        raise ValueError(f"reference split leak for dataset/reference {leaked}")
    train_rows = rows.loc[partitions.eq("train")].drop(columns="partition", errors="ignore")
    validation_rows = rows.loc[partitions.eq("validation")].drop(columns="partition", errors="ignore")
    if train_rows.empty or validation_rows.empty:
        raise ValueError("split manifest must contain non-empty train and validation partitions")
    return dataset.subset(train_rows), dataset.subset(validation_rows)


class PromptBank:
    def __init__(self, tokenizer, text_encoder, device: torch.device):
        prompts = [GROUP_PROMPTS[group] for group in GROUPS]
        self.groups = GROUPS
        self.index = {group: position for position, group in enumerate(self.groups)}
        self.training_indices = {}
        self.held_out_indices = {}
        for group in GROUPS:
            self.training_indices[group] = list(range(len(prompts), len(prompts) + len(TRAINING_GROUP_PROMPTS[group])))
            prompts.extend(TRAINING_GROUP_PROMPTS[group])
        for group in GROUPS:
            self.held_out_indices[group] = list(range(len(prompts), len(prompts) + len(HELD_OUT_PARAPHRASES[group])))
            prompts.extend(HELD_OUT_PARAPHRASES[group])
        self.generic_index = len(prompts)
        prompts.append(GENERIC_PROMPT)
        self.vectors = encode_prompts(tokenizer, text_encoder, prompts, device)

    def for_groups(
        self, groups: list[str], device: torch.device, mode: str = "correct", paraphrase_index: int = 0
    ) -> torch.Tensor:
        if mode == "zero":
            return torch.zeros((len(groups), self.vectors.shape[1]), device=device)
        if mode == "generic":
            indices = [self.generic_index] * len(groups)
        elif mode == "heldout":
            indices = [
                self.held_out_indices.get(group, self.held_out_indices["authentic"])[paraphrase_index]
                for group in groups
            ]
        elif mode == "train_paraphrase":
            indices = [
                random.choice(self.training_indices.get(group, self.training_indices["authentic"]))
                for group in groups
            ]
        elif mode == "wrong":
            indices = [self.index[wrong_group(group)] for group in groups]
        else:
            indices = [self.index.get(group, self.index["authentic"]) for group in groups]
        return self.vectors[indices].to(device)


def evaluate(
    backbone, head, loader, device, prompts: PromptBank | None, mode: str = "correct", paraphrase_index: int = 0
) -> dict:
    head.eval()
    started = time.perf_counter()
    predictions, targets, references, datasets = [], [], [], []
    shuffled_groups = None
    offset = 0
    if mode == "shuffled":
        shuffled_groups = loader.dataset.rows["group"].fillna("authentic").astype(str).tolist()
        np.random.default_rng(0).shuffle(shuffled_groups)
    with torch.no_grad():
        for batch in loader:
            base = head.latent if isinstance(head, (DatasetScaleHead, MDTVSFAHead)) else head
            view_head = getattr(base, "requires_view_features", False)
            patch_head = getattr(base, "requires_patch_tokens", False)
            vision = (batch_vision(backbone, batch, device), batch_patches(backbone, batch, device)) if getattr(base, "requires_pooled_features", False) else (batch_patches(backbone, batch, device) if patch_head else batch_vision(backbone, batch, device))
            if prompts is None:
                prediction = (
                    head(vision, datasets=batch["dataset"])
                    if isinstance(head, (DatasetScaleHead, MDTVSFAHead)) else head(vision)
                )
            else:
                groups = batch["group"] if shuffled_groups is None else shuffled_groups[offset:offset + len(batch["group"])]
                text = prompts.for_groups(
                    groups, device, "correct" if mode == "shuffled" else mode, paraphrase_index
                )
                prediction = (
                    head(vision, text, datasets=batch["dataset"])
                    if isinstance(head, (DatasetScaleHead, MDTVSFAHead)) else head(vision, text)
                )
                offset += len(batch["group"])
            predictions.append(prediction.cpu().numpy())
            targets.append(batch["target"].numpy())
            references.extend(batch["reference"])
            datasets.extend(batch["dataset"])
    head.train()
    frame = pd.DataFrame({
        "p": np.concatenate(predictions), "t": np.concatenate(targets),
        "ref": references, "dataset": datasets,
    })
    per_dataset = {}
    for name, rows in frame.groupby("dataset"):
        if len(rows) < 2 or rows["t"].nunique() < 2:
            continue
        per_dataset[name] = {
            "srcc": float(stats.spearmanr(rows["p"], rows["t"]).correlation),
            "plcc": float(stats.pearsonr(rows["p"], rows["t"]).statistic),
            "n": len(rows),
        }
    srccs = [row["srcc"] for row in per_dataset.values()]
    return {
        "per_dataset": per_dataset,
        "macro_srcc": float(np.mean(srccs)) if srccs else None,
        "macro_plcc": float(np.mean([row["plcc"] for row in per_dataset.values()])) if srccs else None,
        "worst_srcc": float(min(srccs)) if srccs else None,
        "worst_dataset": min(per_dataset, key=lambda key: per_dataset[key]["srcc"]) if per_dataset else None,
        "srcc_per_reference": None,
        "images": len(frame),
        "elapsed_seconds": time.perf_counter() - started,
    }


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    datasets: list[str] | tuple[str, ...],
    references: list[str] | tuple[str, ...],
    scope: str,
) -> torch.Tensor:
    """Pairwise logistic loss over compatible examples in one batch."""
    keys = datasets if scope == "dataset" else references
    losses = []
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            if keys[left] != keys[right] or target[left] == target[right]:
                continue
            direction = torch.sign(target[left] - target[right])
            losses.append(torch.nn.functional.softplus(-direction * (prediction[left] - prediction[right])))
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def monotonicity_induced_loss(relative: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MDTVSFA's list-wise monotonicity surrogate for one dataset batch."""
    count = relative.numel()
    if count < 2:
        return relative.sum() * 0.0
    prediction_delta = relative[:, None] - relative[None, :]
    target_direction = torch.sign(target[None, :] - target[:, None])
    errors = torch.relu(prediction_delta * target_direction)
    return errors.sum() / (count * (count - 1))


def linearity_induced_loss(perceptual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """One minus centered cosine similarity, the differentiable PLCC loss."""
    if perceptual.numel() < 2:
        return perceptual.sum() * 0.0
    centered_prediction = perceptual - perceptual.mean()
    centered_target = target - target.mean()
    prediction_norm = centered_prediction.norm()
    target_norm = centered_target.norm()
    if prediction_norm.item() == 0.0 or target_norm.item() == 0.0:
        return perceptual.sum() * 0.0
    correlation = torch.sum(centered_prediction * centered_target) / (prediction_norm * target_norm)
    return (1.0 - correlation.clamp(-1.0, 1.0)) / 2.0


def error_induced_loss(
    aligned: torch.Tensor, target: torch.Tensor, score_range: tuple[float, float] | None,
) -> torch.Tensor:
    """Normalized L1 loss after dataset-specific perceptual-scale alignment."""
    low, high = score_range or (0.0, 1.0)
    scale = max(float(high) - float(low), 1e-6)
    return torch.mean(torch.abs(aligned - target)) / scale


def mdtvsfa_dataset_losses(
    relative: torch.Tensor,
    perceptual: torch.Tensor,
    aligned: torch.Tensor,
    target: torch.Tensor,
    datasets: list[str] | tuple[str, ...],
    score_ranges: dict[str, tuple[float, float]],
    *,
    pipal_ranking_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute all three MDTVSFA losses separately for each dataset."""
    losses: dict[str, torch.Tensor] = {}
    for name in sorted(set(str(value) for value in datasets)):
        mask = torch.tensor([str(value) == name for value in datasets], device=target.device)
        relative_d = relative[mask]
        perceptual_d = perceptual[mask]
        aligned_d = aligned[mask]
        target_d = target[mask]
        relative_loss = monotonicity_induced_loss(relative_d, target_d)
        linearity_loss = linearity_induced_loss(perceptual_d, target_d)
        error_loss = (
            target_d.sum() * 0.0
            if pipal_ranking_only and name == "pipal"
            else error_induced_loss(aligned_d, target_d, score_ranges.get(name))
        )
        losses[name] = relative_loss + linearity_loss + error_loss
    return losses


def aggregate_mdtvsfa_losses(losses: list[torch.Tensor], weighting: str) -> torch.Tensor:
    """Aggregate per-dataset three-stage losses like the MDTVSFA paper."""
    if not losses:
        raise RuntimeError("empty MDTVSFA loss list")
    values = torch.stack(losses)
    if weighting == "softmax":
        # This intentionally retains the gradient through the weights, matching
        # the released implementation's exp(loss) weighted objective.
        weights = torch.softmax(values, dim=0)
        return torch.sum(weights * values)
    return values.mean()


def aggregate_dataset_losses(
    losses: list[torch.Tensor], weighting: str,
) -> torch.Tensor:
    """Aggregate one scalar regression loss per dataset in a batch."""
    if not losses:
        raise RuntimeError("empty per-dataset loss list")
    values = torch.stack(losses)
    if weighting == "softmax":
        weights = torch.softmax(values.detach(), dim=0)
        return torch.sum(weights * values)
    return values.mean()


def load_feature_map(dataset: ConditionedIQADataset, backbone, device: torch.device,
                     batch_size: int, workers: int, cache_dir: str, cache_key: str,
                     include_patches: bool = False, include_views: bool = False) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Precompute/reuse frozen features, keyed by the source image path."""
    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    cache_path = cache_root / f"{digest}.pt"
    paths = [str(path) for path in dataset.rows["path"]]
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("paths") == paths and payload.get("cache_key", cache_key) == cache_key:
            print(f"reusing frozen feature cache {cache_path}", flush=True)
            if include_views:
                if "view_features" not in payload:
                    raise ValueError(f"feature cache {cache_path} lacks multi-view features")
                return {path: {"views": views} for path, views in zip(paths, payload["view_features"])}
            if include_patches:
                if "patch_features" not in payload:
                    raise ValueError(f"feature cache {cache_path} lacks patch tokens")
                return {path: {"pooled": pooled, "patches": patches} for path, pooled, patches in zip(paths, payload["features"], payload["patch_features"])}
            return {path: feature for path, feature in zip(paths, payload["features"])}
    # Leave-one-group-out and mixture subsets can reuse a previously computed
    # superset cache without a second backbone pass.
    requested = set(paths)
    source_key = cache_key.rsplit("|", 1)[0]
    for candidate in cache_root.glob("*.pt"):
        if candidate == cache_path:
            continue
        payload = torch.load(candidate, map_location="cpu", weights_only=True)
        cached_paths = payload.get("paths", [])
        candidate_key = str(payload.get("cache_key", ""))
        same_source = candidate_key == cache_key or candidate_key.rsplit("|", 1)[0] == source_key
        if requested.issubset(cached_paths) and same_source and (not include_patches or "patch_features" in payload) and (not include_views or "view_features" in payload):
            print(f"reusing superset frozen feature cache {candidate}", flush=True)
            if include_views:
                view_lookup = {path: feature for path, feature in zip(cached_paths, payload["view_features"])}
                return {path: {"views": view_lookup[path]} for path in paths}
            lookup = {path: feature for path, feature in zip(cached_paths, payload["features"])}
            if include_patches:
                patch_lookup = {path: feature for path, feature in zip(cached_paths, payload["patch_features"])}
                return {path: {"pooled": lookup[path], "patches": patch_lookup[path]} for path in paths}
            return {path: lookup[path] for path in paths}
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    features, patch_features, view_features = [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if include_views:
                images = batch["image_views"].to(device)
                actual_batch, view_count = images.shape[:2]
                output = backbone(pixel_values=images.reshape(actual_batch * view_count, *images.shape[2:]))
                view_features.append(output.pooler_output.reshape(actual_batch, view_count, -1).float().cpu())
                continue
            images = batch["image"].to(device)
            output = backbone(pixel_values=images)
            features.append(output.pooler_output.float().cpu())
            if include_patches:
                patch_features.append(output.last_hidden_state[:, 1:].float().cpu())
    stacked = None if include_views else torch.cat(features, dim=0)
    payload = {"paths": paths, "features": stacked, "cache_key": cache_key}
    if include_views:
        payload["view_features"] = torch.cat(view_features, dim=0).half()
    if include_views:
        torch.save(payload, cache_path)
        print(f"cached {len(paths)} multi-view features in {time.perf_counter() - started:.1f}s -> {cache_path}", flush=True)
        return {path: {"views": views} for path, views in zip(paths, payload["view_features"])}
    if include_patches:
        # Patch tokens dominate storage (28k CLIP-B images are ~17 GiB in
        # float32).  They come from a frozen encoder, so store losslessly
        # enough for this head experiment in float16 and cast at consumption.
        payload["patch_features"] = torch.cat(patch_features, dim=0).half()
    torch.save(payload, cache_path)
    print(f"cached {len(paths)} frozen features in {time.perf_counter() - started:.1f}s -> {cache_path}", flush=True)
    if include_patches:
        return {path: {"pooled": pooled, "patches": patches} for path, pooled, patches in zip(paths, stacked, payload["patch_features"])}
    return {path: feature for path, feature in zip(paths, stacked)}


def batch_vision(backbone, batch: dict, device: torch.device) -> torch.Tensor:
    if "view_features" in batch:
        return batch["view_features"].to(device).float()
    if "features" in batch:
        return batch["features"].to(device)
    if "image_views" in batch:
        views = batch["image_views"].to(device)
        actual_batch, view_count = views.shape[:2]
        embeddings = embed(backbone, views.reshape(actual_batch * view_count, *views.shape[2:]))
        return embeddings.reshape(actual_batch, view_count, -1)
    return embed(backbone, batch["image"].to(device))


def batch_patches(backbone, batch: dict, device: torch.device) -> torch.Tensor:
    if "patch_features" in batch:
        return batch["patch_features"].to(device).float()
    return embed_patches(backbone, batch["image"].to(device))


def build_mdtvsfa_loaders(train_set, args) -> dict[str, DataLoader]:
    """Build one shuffled loader per dataset for faithful MDTVSFA training.

    The released MDTVSFA loader presents a batch from every dataset at each
    optimization step.  Keeping dataset batches separate gives the ranking and
    correlation losses enough within-dataset examples and avoids accidental
    cross-dataset pairs in a mixed batch.
    """
    names = sorted(str(name) for name in train_set.rows["dataset"].unique())
    loaders = {}
    for offset, name in enumerate(names):
        subset = train_set.subset(train_set.rows[train_set.rows["dataset"] == name])
        sampler_name = args.sampler if args.sampler != "by_dataset" else "random"
        sampler = make_sampler(subset, sampler_name, seed=args.seed + offset)
        loaders[name] = DataLoader(
            subset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=len(subset) >= args.batch_size,
            num_workers=args.workers,
        )
    return loaders


def iter_mdtvsfa_batches(loaders: dict[str, DataLoader]):
    """Cycle shorter dataset loaders while stepping through the longest one."""
    if not loaders:
        return
    steps = max(len(loader) for loader in loaders.values())
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    for _ in range(steps):
        grouped = {}
        for name, loader in loaders.items():
            try:
                grouped[name] = next(iterators[name])
            except StopIteration:
                iterators[name] = iter(loader)
                grouped[name] = next(iterators[name])
        yield grouped


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    backbone, image_size, vision_dim = load_backbone(args.backbone, args.weights, device)
    family = "siglip" if args.backbone.startswith("siglip") else "clip"
    dataset = ConditionedIQADataset(
        args.data, image_size=image_size, backbone=family, score_column=args.score_column,
        preprocessing=args.preprocessing,
    )
    if args.exclude_groups:
        raw_groups = dataset.rows["group"].fillna("authentic").astype(str).replace({"color": "colour"})
        excluded = {str(group) for group in args.exclude_groups}
        dataset = dataset.subset(dataset.rows[~raw_groups.isin(excluded)])
        print(f"excluded groups {sorted(excluded)}; remaining rows {len(dataset)}", flush=True)
    if args.split_manifest:
        train_set, val_set = split_from_manifest(dataset, args.split_manifest)
        print(
            f"using split manifest {args.split_manifest}: "
            f"train {len(train_set)}, validation {len(val_set)}",
            flush=True,
        )
    else:
        train_set, val_set = split_by(dataset, args.split, fraction=0.2, seed=args.seed)
    if args.limit and args.limit < len(train_set.rows):
        train_set = train_set.subset(train_set.rows.sample(args.limit, random_state=args.seed))
    if args.cache_features:
        include_patches = args.method in {"patch_weighted", "patch_interaction", "global_patch_residual", "global_text_patch_residual", "global_text_cross_attention"}
        include_views = args.method in {"multiview_baseline", "multiview_interaction", "multiview_uniform_interaction"}
        cache_dataset = dataset
        if args.limit and args.limit < len(train_set.rows):
            # A smoke/limited run should not spend time encoding examples that
            # cannot participate in training or validation.
            cache_rows = pd.concat([train_set.rows, val_set.rows], ignore_index=True)
            cache_dataset = dataset.subset(cache_rows)
        feature_map = load_feature_map(
            cache_dataset, backbone, device, args.batch_size, args.workers, args.feature_cache_dir,
            f"{args.data}|{args.backbone}|{args.weights}|{args.preprocessing}|{'views' if include_views else ('patches' if include_patches else 'pooled')}",
            include_patches=include_patches, include_views=include_views,
        )
        train_set = FeatureDataset(train_set, feature_map)
        val_set = FeatureDataset(val_set, feature_map)
    faithful_loaders = None
    if args.dataset_objective == "mdtvsfa_faithful":
        # The faithful objective consumes one batch per dataset at each step;
        # a single mixed loader cannot provide the per-dataset stage losses.
        train_loader = None
        faithful_loaders = build_mdtvsfa_loaders(train_set, args)
    else:
        sampler = make_sampler(train_set, args.sampler, seed=args.seed)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, sampler=sampler,
            shuffle=sampler is None, num_workers=args.workers,
        )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    prompts = None
    text_weights = None
    if args.method in {"baseline", "multiview_baseline"}:
        if args.method == "multiview_baseline":
            head = MultiViewQualityHead(vision_dim, args.fusion_dim, args.hidden_dim).to(device)
        else:
            head = QualityMLP(vision_dim, args.hidden_dim, mlp_layers=args.mlp_layers).to(device)
    elif args.method == "patch_weighted":
        head = PatchWeightedHead(vision_dim, args.hidden_dim).to(device)
    elif args.method == "global_patch_residual":
        head = GlobalPatchResidualHead(vision_dim, args.hidden_dim).to(device)
    else:
        model_id = args.text_encoder_id or BACKBONES[args.backbone][0]
        text_weights = args.text_weights or (None if args.text_encoder_id else args.weights)
        tokenizer, text_encoder, text_dim = load_frozen_text_encoder(
            model_id, text_weights, device, native=args.text_encoder_id is None
        )
        prompts = PromptBank(tokenizer, text_encoder, device)
        del text_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.method == "multiview_interaction":
            head = MultiViewTextFusionHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
        elif args.method == "multiview_uniform_interaction":
            head = MultiViewUniformTextFusionHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
        elif args.method == "adapter_interaction":
            head = AdapterTextFusionHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim, mlp_layers=args.mlp_layers).to(device)
        elif args.method == "film":
            head = FiLMTextHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
        elif args.method == "global_text_cross_attention":
            head = GlobalTextCrossAttentionHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
        elif args.method == "global_text_patch_residual":
            head = GlobalTextPatchResidualHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
        elif args.method == "patch_interaction":
            head = TextPatchWeightedHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
        elif args.method == "residual":
            head = ResidualTextHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
        else:
            head = TextFusionHead(
                vision_dim, text_dim, args.fusion_dim, args.hidden_dim,
                args.method == "interaction", mlp_layers=args.mlp_layers,
            ).to(device)
    score_ranges = {}
    if args.dataset_objective == "mdtvsfa":
        training_datasets = args.calibration_datasets or sorted(train_set.rows["dataset"].astype(str).unique())
        head = DatasetScaleHead(head, training_datasets).to(device)
    elif args.dataset_objective == "mdtvsfa_faithful":
        training_datasets = args.calibration_datasets or sorted(train_set.rows["dataset"].astype(str).unique())
        score_ranges = dataset_score_ranges(train_set.rows, args.score_column)
        head = MDTVSFAHead(head, training_datasets, score_ranges).to(device)
    else:
        training_datasets = []
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.1)
    tracker = MLflowTracker(args)
    reporter = ResultReporter.from_args(args)
    run_id = tracker.mlflow.active_run().info.run_id if tracker.mlflow is not None else ""
    if tracker.mlflow is not None:
        if args.split_manifest:
            tracker.mlflow.log_artifact(str(Path(args.split_manifest).expanduser()), artifact_path="splits")
        tracker.mlflow.set_tags({"conditioning/method": args.method, "conditioning/text_encoder": model_id if prompts else "none"})
        tracker.mlflow.log_params({
            "fusion_dim": args.fusion_dim,
            "mlp_layers": args.mlp_layers,
            "condition_dropout": args.condition_dropout,
            "train_paraphrases": args.train_paraphrases,
            "paraphrase_consistency_weight": args.paraphrase_consistency_weight,
            "text_weights": text_weights or "mirror",
            "dataset_objective": args.dataset_objective,
            "dataset_loss_weighting": args.dataset_loss_weighting,
            "ranking_weight": args.ranking_weight,
            "ranking_scope": args.ranking_scope,
            "pipal_ranking_only": args.pipal_ranking_only,
            "calibration_datasets": ",".join(training_datasets),
            "score_column": args.score_column,
        })
    print(f"{args.method}: {args.backbone} on {device}; train {len(train_set)}, validation {len(val_set)}")
    try:
        tracker.log_dataset(len(train_set), len(val_set), vision_dim, sum(parameter.numel() for parameter in head.parameters()), device)
        step = 0
        best_epoch = -1
        best_macro_srcc = float("-inf")
        best_head_state = None
        for epoch in range(args.epochs):
            losses = []
            batches = iter_mdtvsfa_batches(faithful_loaders) if faithful_loaders is not None else train_loader
            for batch in batches:
                if args.dataset_objective == "mdtvsfa_faithful":
                    dataset_losses = []
                    consistency_losses = []
                    for _, dataset_batch in batch.items():
                        vision = batch_vision(backbone, dataset_batch, device)
                        if prompts is None:
                            text = None
                        else:
                            prompt_mode = "train_paraphrase" if args.train_paraphrases else "correct"
                            text = prompts.for_groups(dataset_batch["group"], device, prompt_mode)
                            if args.condition_dropout:
                                text[torch.rand(len(text), device=device) < args.condition_dropout] = 0
                        targets = dataset_batch["target"].to(device)
                        batch_datasets = [str(value) for value in dataset_batch["dataset"]]
                        relative, perceptual, aligned = head.stages(
                            vision, text, datasets=batch_datasets
                        )
                        per_dataset = mdtvsfa_dataset_losses(
                            relative, perceptual, aligned, targets, batch_datasets, score_ranges,
                            pipal_ranking_only=args.pipal_ranking_only,
                        )
                        dataset_losses.extend(per_dataset.values())
                        if prompts is not None and args.paraphrase_consistency_weight:
                            alternate = prompts.for_groups(dataset_batch["group"], device, "train_paraphrase")
                            _, _, alternate_aligned = head.stages(
                                vision, alternate, datasets=batch_datasets
                            )
                            consistency_losses.append(torch.mean(torch.abs(aligned - alternate_aligned)))
                    loss = aggregate_mdtvsfa_losses(dataset_losses, args.dataset_loss_weighting)
                    if consistency_losses:
                        loss = loss + args.paraphrase_consistency_weight * torch.stack(consistency_losses).mean()
                else:
                    vision = (batch_vision(backbone, batch, device), batch_patches(backbone, batch, device)) if args.method in {"global_patch_residual", "global_text_patch_residual", "global_text_cross_attention"} else (batch_patches(backbone, batch, device) if args.method in {"patch_weighted", "patch_interaction"} else batch_vision(backbone, batch, device))
                    if prompts is None:
                        prediction = (
                            head(vision, datasets=batch["dataset"])
                            if isinstance(head, DatasetScaleHead) else head(vision)
                        )
                    else:
                        prompt_mode = "train_paraphrase" if args.train_paraphrases else "correct"
                        text = prompts.for_groups(batch["group"], device, prompt_mode)
                        if args.condition_dropout:
                            text[torch.rand(len(text), device=device) < args.condition_dropout] = 0
                        prediction = (
                            head(vision, text, datasets=batch["dataset"])
                            if isinstance(head, DatasetScaleHead) else head(vision, text)
                        )
                    targets = batch["target"].to(device)
                if args.dataset_objective == "mdtvsfa":
                    regression_losses = []
                    batch_datasets = [str(value) for value in batch["dataset"]]
                    for name in sorted(set(batch_datasets)):
                        mask = torch.tensor([value == name for value in batch_datasets], device=device)
                        if args.pipal_ranking_only and name == "pipal":
                            continue
                        regression_losses.append(loss_fn(prediction[mask], targets[mask]))
                    loss = (
                        aggregate_dataset_losses(regression_losses, args.dataset_loss_weighting)
                        if regression_losses else prediction.sum() * 0.0
                    )
                    if args.ranking_weight:
                        ranking_scope = "reference" if args.pipal_ranking_only else args.ranking_scope
                        ranking = pairwise_ranking_loss(
                            prediction, targets, batch_datasets, list(batch["reference"]), ranking_scope
                        )
                        loss = loss + args.ranking_weight * ranking
                elif args.dataset_objective != "mdtvsfa_faithful":
                    loss = loss_fn(prediction, targets)
                if args.dataset_objective != "mdtvsfa_faithful" and prompts is not None and args.paraphrase_consistency_weight:
                    alternate = prompts.for_groups(batch["group"], device, "train_paraphrase")
                    alternate_prediction = (
                        head(vision, alternate, datasets=batch["dataset"])
                        if isinstance(head, DatasetScaleHead) else head(vision, alternate)
                    )
                    loss = loss + args.paraphrase_consistency_weight * torch.mean(
                        torch.abs(prediction - alternate_prediction)
                    )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                value = float(loss.detach())
                losses.append(value)
                tracker.log_train_step(value, step)
                step += 1
            scores = evaluate(backbone, head, val_loader, device, prompts)
            tracker.log_epoch(epoch, float(np.mean(losses)), scores)
            if scores["macro_srcc"] is not None and scores["macro_srcc"] > best_macro_srcc:
                best_epoch = epoch
                best_macro_srcc = scores["macro_srcc"]
                best_head_state = copy.deepcopy(head.state_dict())
            print(f"epoch {epoch}: loss {np.mean(losses):.4f} macro SRCC {scores['macro_srcc']:.4f}", flush=True)
        if best_head_state is None:
            raise RuntimeError("no finite validation SRCC was recorded; cannot select a checkpoint")
        head.load_state_dict(best_head_state)
        final_validation_scores = evaluate(backbone, head, val_loader, device, prompts)
        if tracker.mlflow is not None:
            tracker.mlflow.log_metrics({
                "selection/best_epoch": best_epoch,
                "selection/best_validation_macro_srcc": best_macro_srcc,
                "selection/selected_validation_macro_srcc": final_validation_scores["macro_srcc"],
                "selection/selected_validation_macro_plcc": final_validation_scores["macro_plcc"],
            })
        print(f"selected epoch {best_epoch}: macro SRCC {best_macro_srcc:.4f}", flush=True)
        if prompts is not None:
            for mode in ("zero", "generic", "heldout", "wrong", "shuffled"):
                intervention_scores = evaluate(backbone, head, val_loader, device, prompts, mode)
                if tracker.mlflow is not None and intervention_scores["macro_srcc"] is not None:
                    tracker.mlflow.log_metric(f"intervention/{mode}/macro_srcc", intervention_scores["macro_srcc"])
                print(f"{mode}: macro SRCC {intervention_scores['macro_srcc']:.4f}", flush=True)
            paraphrase_scores = [
                evaluate(backbone, head, val_loader, device, prompts, "heldout", paraphrase_index=index)
                for index in range(len(HELD_OUT_PARAPHRASES["authentic"]))
            ]
            worst_paraphrase = min(
                score["macro_srcc"] for score in paraphrase_scores if score["macro_srcc"] is not None
            )
            if tracker.mlflow is not None:
                tracker.mlflow.log_metric("intervention/paraphrase_worst/macro_srcc", worst_paraphrase)
            print(f"paraphrase_worst: macro SRCC {worst_paraphrase:.4f}", flush=True)
        images_per_second = final_validation_scores["images"] / final_validation_scores["elapsed_seconds"]
        def forward_once():
            images = torch.randn((5, 3, image_size, image_size), device=device) if args.method in {"multiview_baseline", "multiview_interaction", "multiview_uniform_interaction"} else torch.randn(1, 3, image_size, image_size, device=device)
            if args.method in {"multiview_baseline", "multiview_interaction", "multiview_uniform_interaction"}:
                features = embed(backbone, images).unsqueeze(0)
            else:
                features = (embed(backbone, images), embed_patches(backbone, images)) if args.method in {"global_patch_residual", "global_text_patch_residual", "global_text_cross_attention"} else (embed_patches(backbone, images) if args.method in {"patch_weighted", "patch_interaction"} else embed(backbone, images))
            if prompts is None:
                return head(features)
            return head(features, torch.randn(1, text_dim, device=device))

        latency_p50_ms, latency_p95_ms, peak_memory_mb = measure_latency_memory(
            forward_once, device
        )
        flops = measure_flops(forward_once)
        image_throughput = 1000 / latency_p50_ms if latency_p50_ms else images_per_second
        report_rows = [
            {
                "run_id": run_id,
                "experiment": args.mlflow_experiment if args.mlflow else "",
                "run_name": args.mlflow_run_name or "",
                "evaluation": "validation",
                "dataset": name,
                "backbone": args.backbone,
                "method": args.method,
                "seed": args.seed,
                "epochs": args.epochs,
                "latency_p50_ms": latency_p50_ms,
                "latency_p95_ms": latency_p95_ms,
                "peak_memory_mb": peak_memory_mb,
                "images": row["n"],
                "srcc": row["srcc"],
                "plcc": row["plcc"],
                "srcc_per_reference": final_validation_scores["srcc_per_reference"],
                "images_per_second": images_per_second,
                "milliseconds_per_image": 1000 / images_per_second,
                "head_size_mb": size_megabytes(head),
                "model_parameter_size_mb": size_megabytes(backbone) + size_megabytes(head),
                "config_path": args.config or "",
            }
            for name, row in final_validation_scores["per_dataset"].items()
        ]
        try:
            reporter.append(report_rows)
        except RuntimeError as error:
            print(f"results-table export failed after local save: {error}", file=sys.stderr)
            if tracker.mlflow is not None:
                tracker.mlflow.set_tag("results_export_error", str(error))
        if tracker.mlflow is not None:
            tracker.mlflow.log_metrics({
                "system/validation_images_per_second": images_per_second,
                "system/head_size_mb": size_megabytes(head),
                "system/model_parameter_size_mb": size_megabytes(backbone) + size_megabytes(head),
                "system/latency_p50_ms": latency_p50_ms,
                "system/latency_p95_ms": latency_p95_ms,
                "system/peak_memory_mb": peak_memory_mb,
                "system/image_throughput": image_throughput,
                "system/flops": flops,
            })
        tracker.log_checkpoint(head, args, vision_dim)
        if args.out:
            torch.save({"head": head.state_dict(), "backbone": args.backbone, "method": args.method}, args.out)
    except Exception:
        tracker.close("FAILED")
        raise
    else:
        tracker.close()


if __name__ == "__main__":
    main()
