"""Measure KADID-10k image throughput for a completed checkpoint.

Unlike ``evaluate_text_conditioned.py``, this script intentionally does not
compute SRCC/PLCC.  KADID-10k is a training dataset for most models, so a
correlation reported on it as a held-out test would be misleading.  The script
only records end-to-end inference throughput plus latency, memory, FLOPs and
parameter count, which is what the summary table needs for its ``FPS`` column.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from evaluate_text_conditioned import build_head, source_defaults
from result_reporting import (
    ResultReporter,
    add_reporting_arguments,
    measure_flops,
    measure_latency_memory,
    size_megabytes,
)
from text_conditioning.data import ConditionedIQADataset
from text_conditioning.text_encoder import load_frozen_text_encoder
from train import BACKBONES, embed, load_backbone
from train_text_conditioned import PromptBank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--data", required=True, help="prepared KADID labels.csv")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--text-weights", default=None)
    parser.add_argument("--method", choices=("baseline", "concat", "interaction", "residual"), default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--fusion-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", default="conditioned-iqa-external-benchmark")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    add_reporting_arguments(parser)
    return parser


def parse_args() -> tuple[argparse.Namespace, Path, str]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    bootstrap.add_argument("--source-run-id", required=True)
    bootstrap.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults, checkpoint_path, source_name = source_defaults(
        bootstrap_args.source_run_id, bootstrap_args.mlflow_tracking_uri
    )
    parser = build_parser()
    if bootstrap_args.config:
        with Path(bootstrap_args.config).open(encoding="utf-8") as stream:
            defaults.update(yaml.safe_load(stream) or {})
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
    return parser.parse_args(), checkpoint_path, source_name


def main() -> None:
    args, checkpoint_path, source_name = parse_args()
    required = ("backbone", "weights", "method", "hidden_dim")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"source manifest lacks required settings: {', '.join(missing)}")

    import mlflow

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run_name = args.mlflow_run_name or f"external-kadid-benchmark-{source_name}"
    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tag("source_run_id", args.source_run_id)
    mlflow.set_tag("benchmark_dataset", "kadid10k")

    device = torch.device(
        args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    )
    backbone, image_size, vision_dim = load_backbone(args.backbone, args.weights, device)
    prompts = None
    text_dim = None
    if args.method != "baseline":
        model_id = BACKBONES[args.backbone][0]
        tokenizer, text_encoder, text_dim = load_frozen_text_encoder(
            model_id, args.text_weights or args.weights, device
        )
        prompts = PromptBank(tokenizer, text_encoder, device)
        del text_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    head = build_head(args, vision_dim, text_dim, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    head.load_state_dict(checkpoint["head"])
    head.eval()

    def forward_once() -> None:
        features = embed(backbone, torch.randn(1, 3, image_size, image_size, device=device))
        if args.method == "baseline":
            head(features)
        else:
            head(features, torch.randn(1, text_dim, device=device))

    latency_p50_ms, latency_p95_ms, peak_memory_mb = measure_latency_memory(forward_once, device)
    flops = measure_flops(forward_once)

    family = "siglip" if args.backbone.startswith("siglip") else "clip"
    dataset = ConditionedIQADataset(
        args.data, image_size=image_size, backbone=family, score_column="scaled_subjective_score"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    total_images = 0
    with torch.no_grad():
        for batch in loader:
            vision = embed(backbone, batch["image"].to(device))
            if prompts is None:
                head(vision)
            else:
                head(vision, prompts.for_groups(batch["group"], device, "correct"))
            total_images += len(batch["image"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    images_per_second = total_images / elapsed if elapsed else 0.0

    row = {
        "run_id": run.info.run_id,
        "source_run_id": args.source_run_id,
        "experiment": args.mlflow_experiment,
        "run_name": run_name,
        "evaluation": "held_out_test",
        "dataset": "kadid10k",
        "backbone": args.backbone,
        "method": args.method,
        "seed": "",
        "epochs": args.epochs if args.epochs is not None else "",
        "latency_p50_ms": latency_p50_ms,
        "latency_p95_ms": latency_p95_ms,
        "peak_memory_mb": peak_memory_mb,
        "images": total_images,
        "srcc": "",
        "plcc": "",
        "srcc_per_reference": "",
        "images_per_second": images_per_second,
        "milliseconds_per_image": 1000 / images_per_second if images_per_second else "",
        "head_size_mb": size_megabytes(head),
        "model_parameter_size_mb": size_megabytes(backbone) + size_megabytes(head),
        "config_path": f"mlflow:{args.source_run_id}",
    }

    mlflow.log_metrics({
        "system/kadid10k/images_per_second": images_per_second,
        "system/kadid10k/image_count": total_images,
        "system/latency_p50_ms": latency_p50_ms,
        "system/latency_p95_ms": latency_p95_ms,
        "system/peak_memory_mb": peak_memory_mb,
        "system/image_throughput": images_per_second,
        "system/flops": flops,
        "system/head_size_mb": size_megabytes(head),
        "system/model_parameter_size_mb": size_megabytes(backbone) + size_megabytes(head),
    })
    reporter = ResultReporter.from_args(args)
    reporter.append([row])
    print(
        f"{source_name}: {total_images} KADID images in {elapsed:.1f}s "
        f"({images_per_second:.2f} images/s, latency p50 {latency_p50_ms:.2f} ms)"
    )
    mlflow.end_run()


if __name__ == "__main__":
    main()
