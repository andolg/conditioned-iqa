"""Run pooled text-conditioned IQA experiments without changing ``train.py``.

The vision backbone and text tower are frozen. Methods are ``baseline`` for the
unconditioned pooled MLP, ``concat`` for image/text concatenation, and
``interaction`` for concatenation plus elementwise image-text interaction.
"""

from __future__ import annotations

import argparse
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
from result_reporting import ResultReporter, add_reporting_arguments, size_megabytes
from text_conditioning.data import ConditionedIQADataset
from text_conditioning.models import ResidualTextHead, TextFusionHead
from text_conditioning.prompts import (
    GENERIC_PROMPT,
    GROUP_PROMPTS,
    GROUPS,
    HELD_OUT_GROUP_PROMPTS,
    wrong_group,
)
from text_conditioning.text_encoder import encode_prompts, load_frozen_text_encoder
from train import BACKBONES, MLflowTracker, QualityMLP, embed, load_backbone


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--backbone", default="clip-base", choices=sorted(BACKBONES))
    parser.add_argument("--weights", default=None, help="local frozen vision checkpoint")
    parser.add_argument("--text-weights", default=None, help="local frozen text checkpoint; defaults to --weights")
    parser.add_argument("--method", choices=["baseline", "concat", "interaction", "residual"], default="concat")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--condition-dropout", type=float, default=0.0,
                        help="probability of replacing a training condition with zero text")
    parser.add_argument("--split", choices=["reference", "random"], default="reference")
    parser.add_argument("--score-column", default="scaled_subjective_score")
    parser.add_argument("--sampler", choices=["random", "balanced", "by_level", "by_dataset"], default="random")
    parser.add_argument("--workers", type=int, default=4)
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


class PromptBank:
    def __init__(self, tokenizer, text_encoder, device: torch.device):
        prompts = (
            [GROUP_PROMPTS[group] for group in GROUPS]
            + [HELD_OUT_GROUP_PROMPTS[group] for group in GROUPS]
            + [GENERIC_PROMPT]
        )
        vectors = encode_prompts(tokenizer, text_encoder, prompts, device)
        self.groups = GROUPS
        self.index = {group: position for position, group in enumerate(self.groups)}
        self.vectors = vectors
        self.held_out_offset = len(self.groups)
        self.generic_index = len(self.groups) * 2

    def for_groups(self, groups: list[str], device: torch.device, mode: str = "correct") -> torch.Tensor:
        if mode == "zero":
            return torch.zeros((len(groups), self.vectors.shape[1]), device=device)
        if mode == "generic":
            indices = [self.generic_index] * len(groups)
        elif mode == "heldout":
            indices = [self.held_out_offset + self.index.get(group, self.index["authentic"]) for group in groups]
        elif mode == "wrong":
            indices = [self.index[wrong_group(group)] for group in groups]
        else:
            indices = [self.index.get(group, self.index["authentic"]) for group in groups]
        return self.vectors[indices].to(device)


def evaluate(backbone, head, loader, device, prompts: PromptBank | None, mode: str = "correct") -> dict:
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
            vision = embed(backbone, batch["image"].to(device))
            if prompts is None:
                prediction = head(vision)
            else:
                groups = batch["group"] if shuffled_groups is None else shuffled_groups[offset:offset + len(batch["group"])]
                prediction = head(vision, prompts.for_groups(groups, device, "correct" if mode == "shuffled" else mode))
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


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    backbone, image_size, vision_dim = load_backbone(args.backbone, args.weights, device)
    family = "siglip" if args.backbone.startswith("siglip") else "clip"
    dataset = ConditionedIQADataset(args.data, image_size=image_size, backbone=family, score_column=args.score_column)
    train_set, val_set = split_by(dataset, args.split, fraction=0.2, seed=args.seed)
    if args.limit and args.limit < len(train_set.rows):
        train_set = train_set.subset(train_set.rows.sample(args.limit, random_state=args.seed))
    sampler = make_sampler(train_set, args.sampler, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    prompts = None
    if args.method == "baseline":
        head = QualityMLP(vision_dim, args.hidden_dim).to(device)
    else:
        model_id = BACKBONES[args.backbone][0]
        tokenizer, text_encoder, text_dim = load_frozen_text_encoder(model_id, args.text_weights or args.weights, device)
        prompts = PromptBank(tokenizer, text_encoder, device)
        del text_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if args.method == "residual":
            head = ResidualTextHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim).to(device)
        else:
            head = TextFusionHead(vision_dim, text_dim, args.fusion_dim, args.hidden_dim, args.method == "interaction").to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = torch.nn.SmoothL1Loss(beta=0.1)
    tracker = MLflowTracker(args)
    reporter = ResultReporter.from_args(args)
    run_id = tracker.mlflow.active_run().info.run_id if tracker.mlflow is not None else ""
    if tracker.mlflow is not None:
        tracker.mlflow.set_tags({"conditioning/method": args.method, "conditioning/text_encoder": BACKBONES[args.backbone][0] if prompts else "none"})
        tracker.mlflow.log_params({
            "fusion_dim": args.fusion_dim,
            "condition_dropout": args.condition_dropout,
            "text_weights": args.text_weights or args.weights or "mirror",
        })
    print(f"{args.method}: {args.backbone} on {device}; train {len(train_set)}, validation {len(val_set)}")
    try:
        tracker.log_dataset(len(train_set), len(val_set), vision_dim, sum(parameter.numel() for parameter in head.parameters()), device)
        step = 0
        for epoch in range(args.epochs):
            losses = []
            for batch in train_loader:
                vision = embed(backbone, batch["image"].to(device))
                if prompts is None:
                    prediction = head(vision)
                else:
                    text = prompts.for_groups(batch["group"], device)
                    if args.condition_dropout:
                        text[torch.rand(len(text), device=device) < args.condition_dropout] = 0
                    prediction = head(vision, text)
                loss = loss_fn(prediction, batch["target"].to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                value = float(loss.detach())
                losses.append(value)
                tracker.log_train_step(value, step)
                step += 1
            scores = evaluate(backbone, head, val_loader, device, prompts)
            tracker.log_epoch(epoch, float(np.mean(losses)), scores)
            print(f"epoch {epoch}: loss {np.mean(losses):.4f} macro SRCC {scores['macro_srcc']:.4f}", flush=True)
        final_validation_scores = scores
        if prompts is not None:
            for mode in ("zero", "generic", "heldout", "wrong", "shuffled"):
                intervention_scores = evaluate(backbone, head, val_loader, device, prompts, mode)
                if tracker.mlflow is not None and intervention_scores["macro_srcc"] is not None:
                    tracker.mlflow.log_metric(f"intervention/{mode}/macro_srcc", intervention_scores["macro_srcc"])
                print(f"{mode}: macro SRCC {intervention_scores['macro_srcc']:.4f}", flush=True)
        images_per_second = final_validation_scores["images"] / final_validation_scores["elapsed_seconds"]
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
