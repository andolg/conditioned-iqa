"""Evaluate a saved IQA head on one or more complete prepared datasets.

The frozen vision backbone and trained head are loaded once, then reused for
every ``labels.csv`` passed to ``--data``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import IQADataset
from arniqa import (
    ARNIQA_CROP_SIZE,
    ARNIQA_FEATURE_DIM,
    embed_arniqa,
    load_arniqa_encoder,
)
from label_and_embed_conditioning.train import (
    LABEL_FUSION_HEADS,
    ARNIQAConditionedQualityMLP,
    CONDITION_GROUPS,
    LabelConditionedQualityMLP,
    QualityMLP,
    embed,
    encode_groups,
    evaluate,
    load_backbone,
    make_label_conditioned_head,
)


def _head_from_checkpoint(checkpoint: dict, device: torch.device):
    """Reconstruct a baseline, label-, or ARNIQA-conditioned head."""
    state = checkpoint["head"]
    conditioning = checkpoint.get("conditioning", "none")
    feature_dim = int(checkpoint["feature_dim"])

    if conditioning == "label":
        saved_groups = checkpoint.get("groups")
        if saved_groups is not None and tuple(saved_groups) != CONDITION_GROUPS:
            raise ValueError(
                "checkpoint group vocabulary differs from the current prepared-data vocabulary"
            )
        label_dim = int(checkpoint.get("label_dim") or state["group_embedding.weight"].shape[1])
        hidden_dim = int(checkpoint.get("hidden_dim") or state["net.0.weight"].shape[0])
        label_fusion = checkpoint.get("label_fusion") or "concat"
        if label_fusion not in LABEL_FUSION_HEADS:
            raise ValueError(f"unsupported label fusion: {label_fusion!r}")
        head = make_label_conditioned_head(
            label_fusion,
            feature_dim,
            len(CONDITION_GROUPS),
            hidden_dim=hidden_dim,
            label_dim=label_dim,
            condition_dropout=0.0,
            low_rank_dim=int(checkpoint.get("low_rank_dim") or 4),
        )
    elif conditioning == "arniqa":
        arniqa_dim = int(checkpoint.get("arniqa_feature_dim") or ARNIQA_FEATURE_DIM)
        condition_dim = int(
            checkpoint.get("condition_dim")
            or state["condition_projection.0.weight"].shape[0]
        )
        hidden_dim = int(checkpoint.get("hidden_dim") or state["net.0.weight"].shape[0])
        head = ARNIQAConditionedQualityMLP(
            feature_dim,
            arniqa_dim=arniqa_dim,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            condition_dropout=0.0,
        )
    elif conditioning == "none":
        # Older baseline checkpoints predate hidden_dim metadata.
        hidden_dim = int(checkpoint.get("hidden_dim") or state["net.1.weight"].shape[0])
        head = QualityMLP(feature_dim, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"unsupported checkpoint conditioning mode: {conditioning!r}")

    head.load_state_dict(state)
    return head.eval().to(device), conditioning, feature_dim


@torch.no_grad()
def _model_complexity(
    backbone,
    head,
    image_size: int,
    device: torch.device,
    conditioned: bool,
    arniqa_encoder=None,
    arniqa_batch_size: int = 64,
) -> dict:
    """Parameter counts and counted FLOPs for one normal-path image."""
    modules = (
        (backbone, head)
        if arniqa_encoder is None else (backbone, arniqa_encoder, head)
    )
    total_parameters = sum(
        parameter.numel() for module in modules for parameter in module.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    head_parameters = sum(parameter.numel() for parameter in head.parameters())

    gflops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode

        image = torch.randn(1, 3, image_size, image_size, device=device)
        with FlopCounterMode(display=False) as counter:
            features = embed(
                backbone,
                image,
                patch_tokens=getattr(head, "requires_patch_tokens", False),
            )
            if arniqa_encoder is not None:
                arniqa_image = torch.randn(
                    1, 5, 3, ARNIQA_CROP_SIZE, ARNIQA_CROP_SIZE, device=device
                )
                condition = embed_arniqa(
                    arniqa_encoder,
                    arniqa_image,
                    arniqa_image,
                    chunk_size=arniqa_batch_size,
                )
                head(features, condition)
            elif conditioned:
                head(features, encode_groups(["authentic"], device))
            else:
                head(features)
        gflops = float(counter.get_total_flops() / 1e9)
    except (ImportError, NotImplementedError, RuntimeError):
        # Complexity reporting must not prevent evaluation on a PyTorch build
        # whose FLOP counter does not support an operation in the backbone.
        pass

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "head_parameters": head_parameters,
        "gflops_per_image": gflops,
    }


def _print_scores(csv: Path, scores: dict) -> None:
    print(f"\n{csv.parent.name}  ({csv})")
    for name, row in sorted(scores["per_dataset"].items()):
        print(
            f"  {name:<14s} n {row['n']:>6d}   "
            f"SRCC {row['srcc']:.4f}   PLCC {row['plcc']:.4f}"
        )
    if scores["srcc_per_reference"] is not None:
        print(
            f"  {'within-ref':<14s} {'':>8s}   SRCC {scores['srcc_per_reference']:.4f}   "
            f"({scores['n_references']} references)"
        )
    if "condition_ablations" in scores:
        print("  condition ablations:")
        for mode, mode_scores in scores["condition_ablations"].items():
            print(
                f"    {mode:<10s} macro SRCC {mode_scores['macro_srcc']:.4f}   "
                f"PLCC {mode_scores['macro_plcc']:.4f}"
            )
    peak_memory = (
        f"{scores['peak_memory_mb']:.1f} MB"
        if scores["peak_memory_mb"] is not None else "N/A"
    )
    print(
        f"  performance: latency p50 {scores['latency_p50_ms']:.3f} ms/img   "
        f"p95 {scores['latency_p95_ms']:.3f} ms/img   "
        f"peak memory {peak_memory}   "
        f"throughput {scores['images_per_second']:.2f} img/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, help="prepared labels.csv files")
    parser.add_argument("--checkpoint", required=True, help="saved head checkpoint")
    parser.add_argument("--weights", default=None, help="local frozen vision checkpoint")
    parser.add_argument("--arniqa-weights", default=None,
                        help="local official ARNIQA.pth encoder; downloads it when omitted")
    parser.add_argument("--arniqa-batch-size", type=int, default=64,
                        help="maximum full/half ARNIQA crops encoded at once")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-column", default="scaled_subjective_score")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for deterministic shuffled-condition evaluation")
    args = parser.parse_args()
    if args.arniqa_batch_size < 1:
        parser.error("--arniqa-batch-size must be positive")

    device = torch.device(
        args.device if args.device != "auto"
        else "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone_name = checkpoint["backbone"]
    backbone, image_size, backbone_dim = load_backbone(backbone_name, args.weights, device)
    head, conditioning, feature_dim = _head_from_checkpoint(checkpoint, device)
    arniqa_encoder = (
        load_arniqa_encoder(args.arniqa_weights, device)
        if conditioning == "arniqa" else None
    )
    if backbone_dim != feature_dim:
        raise ValueError(
            f"checkpoint expects {feature_dim}-d features, but {backbone_name} produces "
            f"{backbone_dim}"
        )

    family = "siglip" if backbone_name.startswith("siglip") else "clip"
    complexity = _model_complexity(
        backbone,
        head,
        image_size,
        device,
        conditioning == "label",
        arniqa_encoder=arniqa_encoder,
        arniqa_batch_size=args.arniqa_batch_size,
    )
    gflops = (
        f"{complexity['gflops_per_image']:.3f}"
        if complexity["gflops_per_image"] is not None else "N/A"
    )
    print(
        f"checkpoint {checkpoint_path}  backbone {backbone_name}  "
        f"conditioning {conditioning}  device {device}"
    )
    print(
        f"parameters: total {complexity['total_parameters'] / 1e6:.3f}M   "
        f"trainable {complexity['trainable_parameters'] / 1e6:.3f}M   "
        f"head {complexity['head_parameters'] / 1e6:.3f}M   "
        f"compute {gflops} GFLOPs/image"
    )
    for value in args.data:
        csv = Path(value).expanduser()
        dataset = IQADataset(
            csv, image_size=image_size, backbone=family,
            score_column=args.score_column,
            arniqa=conditioning == "arniqa",
            arniqa_crop_size=int(checkpoint.get("arniqa_crop_size") or ARNIQA_CROP_SIZE),
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.workers,
        )
        scores = evaluate(
            backbone, head, loader, device,
            conditioned=conditioning != "none",
            seed=args.seed,
            conditioning=conditioning,
            arniqa_encoder=arniqa_encoder,
            arniqa_batch_size=args.arniqa_batch_size,
        )
        _print_scores(csv, scores)


if __name__ == "__main__":
    main()
