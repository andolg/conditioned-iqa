"""Evaluate a zero-shot CLIP-IQA-style good-versus-bad prompt reference.

This is intentionally separate from ``train_text_conditioned.py``: it uses
CLIP's image/text projection and no learned quality head.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from torch.nn import functional
from torch.utils.data import DataLoader

from dataset import split_by
from text_conditioning.data import ConditionedIQADataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", required=False)
    parser.add_argument("--weights", required=False, help="local full CLIP snapshot")
    parser.add_argument("--good-prompt", default="a good quality image")
    parser.add_argument("--bad-prompt", default="a bad quality image")
    parser.add_argument("--split", choices=("reference", "random"), default="reference")
    parser.add_argument("--score-column", default="scaled_subjective_score")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--mlflow-experiment", default="conditioned-iqa-text")
    parser.add_argument("--mlflow-run-name", default="clip-iqa-zero-shot")
    parser.add_argument("--config-dir", default="runs/configs")
    return parser


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    initial, _ = bootstrap.parse_known_args()
    parser = build_parser()
    if initial.config:
        with Path(initial.config).open(encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}
        valid = {action.dest for action in parser._actions}
        parser.set_defaults(**{key: value for key, value in values.items() if key in valid})
    args = parser.parse_args()
    if not args.data or not args.weights:
        parser.error("--data and --weights are required")
    return args


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    values = []
    for _, rows in frame.groupby("dataset"):
        if len(rows) > 1 and rows.target.nunique() > 1:
            values.append(float(stats.spearmanr(rows.score, rows.target).correlation))
    return {"macro_srcc": float(np.mean(values)), "macro_plcc": float(stats.pearsonr(frame.score, frame.target).statistic)}


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer, CLIPModel

    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    snapshot = Path(args.weights).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = CLIPModel.from_pretrained(snapshot, local_files_only=True).eval().to(device)
    dataset = ConditionedIQADataset(args.data, image_size=224, backbone="clip", score_column=args.score_column)
    _, validation = split_by(dataset, args.split, fraction=0.2, seed=args.seed)
    loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers)
    with torch.no_grad():
        tokens = tokenizer([args.good_prompt, args.bad_prompt], padding=True, return_tensors="pt")
        text_output = model.text_model(**{key: value.to(device) for key, value in tokens.items()})
        text = model.text_projection(text_output.pooler_output)
        text = functional.normalize(text, dim=-1)
        predictions, targets, datasets = [], [], []
        for batch in loader:
            image_output = model.vision_model(pixel_values=batch["image"].to(device))
            image = functional.normalize(model.visual_projection(image_output.pooler_output), dim=-1)
            similarity = image @ text.T
            predictions.append((similarity[:, 0] - similarity[:, 1]).cpu().numpy())
            targets.append(batch["target"].numpy())
            datasets.extend(batch["dataset"])
    frame = pd.DataFrame({"score": np.concatenate(predictions), "target": np.concatenate(targets), "dataset": datasets})
    result = metrics(frame)
    print(f"zero-shot CLIP-IQA: macro SRCC {result['macro_srcc']:.4f} PLCC {result['macro_plcc']:.4f}")
    if args.mlflow:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.mlflow_run_name):
            mlflow.set_tags({"method": "zero_shot_clip_iqa", "backbone": "clip-base"})
            mlflow.log_params({key: str(value) for key, value in vars(args).items()})
            mlflow.log_metrics({f"validation/{key}": value for key, value in result.items()})
            config_dir = Path(args.config_dir)
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / f"{mlflow.active_run().info.run_id}.yaml"
            with config_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(vars(args), file, sort_keys=True)
            mlflow.log_artifact(str(config_path), artifact_path="configs")
            prediction_path = config_dir / f"{mlflow.active_run().info.run_id}-predictions.csv"
            frame.to_csv(prediction_path, index=False)
            mlflow.log_artifact(str(prediction_path), artifact_path="predictions")


if __name__ == "__main__":
    main()
