"""Score a completed image-only or text-conditioned checkpoint on held-out IQA data."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from result_reporting import (
    ResultReporter,
    add_reporting_arguments,
    measure_flops,
    measure_latency_memory,
    size_megabytes,
)
from text_conditioning.data import ConditionedIQADataset
from text_conditioning.models import AdapterTextFusionHead, DatasetScaleHead, GlobalPatchResidualHead, GlobalTextCrossAttentionHead, GlobalTextPatchResidualHead, MDTVSFAHead, MultiViewQualityHead, MultiViewTextFusionHead, MultiViewUniformTextFusionHead, PatchWeightedHead, ResidualTextHead, TextFusionHead, TextPatchWeightedHead, FiLMTextHead
from text_conditioning.text_encoder import load_frozen_text_encoder
from train import BACKBONES, QualityMLP, embed, embed_patches, load_backbone
from train_text_conditioned import PromptBank, evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional evaluation YAML")
    parser.add_argument("--source-run-id", default=None, help="MLflow run containing checkpoint/config artifacts")
    parser.add_argument("--checkpoint", default=None, help="local quality-head checkpoint (use with --config)")
    parser.add_argument("--data", nargs="+", required=True, help="prepared held-out labels.csv files")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--text-weights", default=None)
    parser.add_argument("--text-encoder-id", default=None)
    parser.add_argument("--method", choices=("baseline", "concat", "interaction", "residual", "film", "patch_weighted", "patch_interaction", "global_patch_residual", "global_text_patch_residual", "global_text_cross_attention", "multiview_baseline", "multiview_interaction", "multiview_uniform_interaction", "adapter_interaction"), default=None)
    parser.add_argument(
        "--dataset-objective", choices=("global", "mdtvsfa", "mdtvsfa_faithful"), default=None
    )
    parser.add_argument("--calibration-datasets", nargs="+", default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--fusion-dim", type=int, default=None)
    parser.add_argument("--mlp-layers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--score-column", default="scaled_subjective_score")
    parser.add_argument(
        "--preprocessing", choices=("stretch", "resize_center_crop", "multiscale"), default=None,
        help="image preprocessing; defaults to the source run's setting",
    )
    parser.add_argument(
        "--path-parent", default=None,
        help="evaluate only rows whose image path's immediate parent has this name",
    )
    parser.add_argument("--include-groups", nargs="+", default=None,
                        help="evaluate only these broad condition groups (E8)")
    parser.add_argument(
        "--condition-mode", choices=("correct", "zero", "generic", "wrong", "heldout"),
        default="correct", help="prompt intervention used during held-out scoring",
    )
    parser.add_argument("--paraphrase-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", default="conditioned-iqa-external-eval")
    parser.add_argument("--mlflow-run-name", default=None)
    add_reporting_arguments(parser)
    return parser


def source_defaults(run_id: str, tracking_uri: str) -> tuple[dict, Path, str]:
    """Download the source manifest and head artifact from MLflow."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri)
    run = client.get_run(run_id)
    config_path = Path(client.download_artifacts(run_id, f"configs/{run_id}.yaml"))
    checkpoint_path = Path(client.download_artifacts(run_id, "checkpoints/quality_head.pt"))
    with config_path.open(encoding="utf-8") as stream:
        defaults = yaml.safe_load(stream) or {}
    return defaults, checkpoint_path, run.data.tags.get("mlflow.runName", "")


def parse_args() -> tuple[argparse.Namespace, Path, str]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    bootstrap.add_argument("--source-run-id")
    bootstrap.add_argument("--checkpoint")
    bootstrap.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    bootstrap_args, _ = bootstrap.parse_known_args()
    if bootstrap_args.source_run_id and bootstrap_args.checkpoint:
        bootstrap.error("use only one of --source-run-id and --checkpoint")
    if not bootstrap_args.source_run_id and not bootstrap_args.checkpoint:
        bootstrap.error("one of --source-run-id or --checkpoint is required")

    if bootstrap_args.source_run_id:
        defaults, checkpoint_path, source_name = source_defaults(
            bootstrap_args.source_run_id, bootstrap_args.mlflow_tracking_uri
        )
    else:
        defaults = {}
        checkpoint_path = Path(bootstrap_args.checkpoint).expanduser()
        source_name = checkpoint_path.stem
    parser = build_parser()
    if bootstrap_args.config:
        with Path(bootstrap_args.config).open(encoding="utf-8") as stream:
            defaults.update(yaml.safe_load(stream) or {})
    # Source manifests describe training.  They must not silently redirect a
    # new test run's data, MLflow experiment, or results destination.
    for key in (
        "data",
        "mlflow_experiment",
        "mlflow_run_name",
        "results_csv",
        "google_sheet_id",
        "google_worksheet",
        "google_service_account_file",
        "google_service_account_json",
    ):
        defaults.pop(key, None)
    valid = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in defaults.items() if key in valid})
    args = parser.parse_args()
    if args.source_run_id and args.checkpoint:
        parser.error("use only one of --source-run-id or --checkpoint")
    if not args.source_run_id and not args.checkpoint:
        parser.error("one of --source-run-id or --checkpoint is required")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.is_file():
            parser.error(f"local checkpoint does not exist: {checkpoint_path}")
        if not args.config:
            parser.error("--config is required when evaluating a local checkpoint")
    # Source manifests created before this option existed use the historical
    # stretch transform, so they remain evaluable and reproducible.
    args.preprocessing = args.preprocessing or "stretch"
    return args, checkpoint_path, source_name


def build_head(args, vision_dim: int, text_dim: int | None, device: torch.device):
    if args.method in {"baseline", "multiview_baseline"}:
        head = (MultiViewQualityHead(vision_dim, args.fusion_dim, args.hidden_dim).to(device)
                if args.method == "multiview_baseline" else
                QualityMLP(vision_dim, args.hidden_dim, mlp_layers=args.mlp_layers or 1).to(device))
    elif args.method == "patch_weighted":
        head = PatchWeightedHead(vision_dim, args.hidden_dim).to(device)
    elif args.method == "patch_interaction":
        head = TextPatchWeightedHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
    elif args.method == "multiview_interaction":
        head = MultiViewTextFusionHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
    elif args.method == "multiview_uniform_interaction":
        head = MultiViewUniformTextFusionHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
    elif args.method == "adapter_interaction":
        head = AdapterTextFusionHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim, mlp_layers=args.mlp_layers or 1).to(device)
    elif args.method == "film":
        head = FiLMTextHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
    elif args.method == "global_patch_residual":
        head = GlobalPatchResidualHead(vision_dim, args.hidden_dim).to(device)
    elif args.method == "global_text_patch_residual":
        head = GlobalTextPatchResidualHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
    elif args.method == "global_text_cross_attention":
        head = GlobalTextCrossAttentionHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
    elif args.method == "residual":
        head = ResidualTextHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
    else:
        head = TextFusionHead(
            vision_dim, text_dim, args.fusion_dim, args.hidden_dim,
            args.method == "interaction", mlp_layers=args.mlp_layers or 1,
        ).to(device)
    if args.dataset_objective == "mdtvsfa":
        names = args.calibration_datasets or [f"calibration_{index}" for index in range(4)]
        head = DatasetScaleHead(head, names).to(device)
    elif args.dataset_objective == "mdtvsfa_faithful":
        names = args.calibration_datasets or [f"calibration_{index}" for index in range(4)]
        head = MDTVSFAHead(head, names).to(device)
    return head


def safe_args(args) -> dict:
    values = vars(args).copy()
    if values.get("google_service_account_json"):
        values["google_service_account_json"] = "<redacted>"
    return values


def main() -> None:
    args, checkpoint_path, source_name = parse_args()
    required = ("backbone", "method", "hidden_dim")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"source run manifest lacks required settings: {', '.join(missing)}")
    import mlflow

    # Start before model loading/inference so a long external benchmark is
    # visible as RUNNING in the UI and fills in one dataset at a time.
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    evaluation_run_name = args.mlflow_run_name or f"external-{source_name}"
    run = mlflow.start_run(run_name=evaluation_run_name)
    mlflow.log_params({key: str(value) for key, value in safe_args(args).items()})
    if args.source_run_id:
        mlflow.set_tag("source_run_id", args.source_run_id)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    backbone, image_size, vision_dim = load_backbone(args.backbone, args.weights, device)
    prompts = None
    text_dim = None
    if args.method not in {"baseline", "multiview_baseline", "patch_weighted", "global_patch_residual"}:
        model_id = args.text_encoder_id or BACKBONES[args.backbone][0]
        text_weights = args.text_weights or (None if args.text_encoder_id else args.weights)
        tokenizer, text_encoder, text_dim = load_frozen_text_encoder(
            model_id, text_weights, device, native=args.text_encoder_id is None
        )
        prompts = PromptBank(tokenizer, text_encoder, device)
        del text_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    head = build_head(args, vision_dim, text_dim, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head.load_state_dict(checkpoint["head"])
    head.eval()
    def forward_once():
        images = torch.randn((5, 3, image_size, image_size), device=device) if args.method in {"multiview_baseline", "multiview_interaction", "multiview_uniform_interaction"} else torch.randn(1, 3, image_size, image_size, device=device)
        if args.method in {"multiview_baseline", "multiview_interaction", "multiview_uniform_interaction"}:
            features = embed(backbone, images).unsqueeze(0)
        else:
            features = (embed(backbone, images), embed_patches(backbone, images)) if args.method in {"global_patch_residual", "global_text_patch_residual", "global_text_cross_attention"} else (embed_patches(backbone, images) if args.method in {"patch_weighted", "patch_interaction"} else embed(backbone, images))
        if args.method in {"baseline", "multiview_baseline", "patch_weighted", "global_patch_residual"}:
            return head(features)
        return head(features, torch.randn(1, text_dim, device=device))

    try:
        latency_p50_ms, latency_p95_ms, peak_memory_mb = measure_latency_memory(
            forward_once, device
        )
    except RuntimeError as error:
        # Throughput reporting is auxiliary.  A CUDA-event failure must not
        # discard a completed held-out metric evaluation.
        print(f"latency measurement failed; continuing without latency metrics: {error}", file=sys.stderr)
        latency_p50_ms = latency_p95_ms = peak_memory_mb = float("nan")
    flops = measure_flops(forward_once)
    rows = []
    for data_path in args.data:
        family = "siglip" if args.backbone.startswith("siglip") else "clip"
        dataset = ConditionedIQADataset(
            data_path, image_size=image_size, backbone=family, score_column=args.score_column,
            preprocessing=args.preprocessing,
        )
        if args.path_parent:
            subset_rows = dataset.rows[dataset.rows["path"].map(
                lambda path: Path(path).parent.name == args.path_parent
            )]
            if subset_rows.empty:
                raise ValueError(f"no rows in {data_path} have parent directory {args.path_parent!r}")
            dataset = dataset.subset(subset_rows)
        if args.include_groups:
            groups = dataset.rows["group"].fillna("authentic").astype(str).replace({"color": "colour"})
            selected = dataset.rows[groups.isin(set(args.include_groups))]
            if selected.empty:
                raise ValueError(f"no rows in {data_path} match groups {args.include_groups}")
            dataset = dataset.subset(selected)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        scores = evaluate(
            backbone, head, loader, device, prompts,
            mode=args.condition_mode, paraphrase_index=args.paraphrase_index,
        )
        images_per_second = scores["images"] / scores["elapsed_seconds"]
        for dataset_name, value in scores["per_dataset"].items():
            rows.append({
                "run_id": run.info.run_id,
                "source_run_id": args.source_run_id,
                "experiment": args.mlflow_experiment,
                "run_name": evaluation_run_name,
                "evaluation": "held_out_test",
                "dataset": dataset_name,
                "backbone": args.backbone,
                "method": args.method,
                "seed": args.seed if hasattr(args, "seed") else "",
                "epochs": args.epochs if args.epochs is not None else "",
                "latency_p50_ms": latency_p50_ms,
                "latency_p95_ms": latency_p95_ms,
                "peak_memory_mb": peak_memory_mb,
                "images": value["n"],
                "srcc": value["srcc"],
                "plcc": value["plcc"],
                "srcc_per_reference": scores["srcc_per_reference"],
                "images_per_second": images_per_second,
                "milliseconds_per_image": 1000 / images_per_second,
                "head_size_mb": size_megabytes(head),
                "model_parameter_size_mb": size_megabytes(backbone) + size_megabytes(head),
                "config_path": args.config or f"mlflow:{args.source_run_id}",
            })
            print(
                f"{dataset_name}: SRCC {value['srcc']:.4f} PLCC {value['plcc']:.4f}; "
                f"{images_per_second:.1f} images/s"
            )
            mlflow.log_metrics({
                f"test/{dataset_name}/srcc": value["srcc"],
                f"test/{dataset_name}/plcc": value["plcc"],
                f"system/{dataset_name}/images_per_second": images_per_second,
            })
    mlflow.log_metrics({
        "system/latency_p50_ms": latency_p50_ms,
        "system/latency_p95_ms": latency_p95_ms,
        "system/peak_memory_mb": peak_memory_mb,
        "system/image_throughput": 1000 / latency_p50_ms if latency_p50_ms else 0,
        "system/flops": flops,
        "system/head_size_mb": size_megabytes(head),
        "system/model_parameter_size_mb": size_megabytes(backbone) + size_megabytes(head),
    })
    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / "external_evaluation.yaml"
        with config_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(safe_args(args), stream, sort_keys=True)
        mlflow.log_artifact(str(config_path), artifact_path="configs")
    reporter = ResultReporter.from_args(args)
    try:
        reporter.append(rows)
    except RuntimeError as error:
        print(f"results-table export failed after local save: {error}", file=sys.stderr)
        mlflow.set_tag("results_export_error", str(error))
    mlflow.end_run()


if __name__ == "__main__":
    main()
