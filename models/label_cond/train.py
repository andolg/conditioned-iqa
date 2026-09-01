import argparse
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_sampler, split_by
from models.dist_classifier.model import DistortionClassifier
from models.label_cond.dataset import GROUPS, LabelConditionedDataset
from models.label_cond.model import (
    LabelConditionedMetric,
    encode_images,
    load_image_encoder,
)


def hard_condition(targets: torch.Tensor) -> torch.Tensor:
    return F.one_hot(targets, num_classes=len(GROUPS)).float()


def predicted_condition(logits: torch.Tensor, label_type: str) -> torch.Tensor:
    if label_type == "soft":
        return logits.softmax(dim=1)
    return hard_condition(logits.argmax(dim=1))


def train_hard(metric, encoder, loader, optimizer, quality_loss, device):
    metric.train()
    losses = []
    for batch in tqdm(loader, desc="train/hard", leave=False):
        features = encode_images(encoder, batch["image"].to(device))
        targets = batch["target"].to(device)
        condition = hard_condition(batch["group"].to(device))
        loss = quality_loss(metric(features, condition), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return {"loss": float(np.mean(losses)), "iqa_loss": float(np.mean(losses))}


def train_frozen(
    metric, classifier, encoder, loader, optimizer, quality_loss, label_type, device
):
    metric.train()
    classifier.eval()
    losses = []
    for batch in tqdm(loader, desc="train/frozen", leave=False):
        features = encode_images(encoder, batch["image"].to(device))
        with torch.no_grad():
            logits = classifier(batch["classifier_image"].to(device))
            condition = predicted_condition(logits, label_type)
        loss = quality_loss(
            metric(features, condition), batch["target"].to(device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return {"loss": float(np.mean(losses)), "iqa_loss": float(np.mean(losses))}


def train_joint(
    metric,
    classifier,
    encoder,
    loader,
    optimizer,
    quality_loss,
    classification_loss,
    label_type,
    lambda_iqa,
    lambda_cls,
    device,
):
    metric.train()
    classifier.train()
    total_losses, iqa_losses, cls_losses = [], [], []
    for batch in tqdm(loader, desc="train/joint", leave=False):
        group_targets = batch["group"].to(device)
        logits = classifier(batch["classifier_image"].to(device))
        condition = predicted_condition(logits, label_type)
        features = encode_images(encoder, batch["image"].to(device))
        iqa_loss = quality_loss(
            metric(features, condition), batch["target"].to(device)
        )
        cls_loss = classification_loss(logits, group_targets)
        loss = lambda_iqa * iqa_loss + lambda_cls * cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_losses.append(loss.item())
        iqa_losses.append(iqa_loss.item())
        cls_losses.append(cls_loss.item())
    return {
        "loss": float(np.mean(total_losses)),
        "iqa_loss": float(np.mean(iqa_losses)),
        "cls_loss": float(np.mean(cls_losses)),
    }


@torch.no_grad()
def evaluate(metric, classifier, encoder, loader, mode, label_type, device):
    metric.eval()
    if classifier is not None:
        classifier.eval()

    predictions, targets, datasets = [], [], []
    cls_correct = cls_count = 0
    for batch in tqdm(loader, desc="validation", leave=False):
        group_targets = batch["group"].to(device)
        if mode == "hard":
            condition = hard_condition(group_targets)
        else:
            logits = classifier(batch["classifier_image"].to(device))
            condition = predicted_condition(logits, label_type)
            cls_correct += (logits.argmax(1) == group_targets).sum().item()
            cls_count += len(group_targets)

        features = encode_images(encoder, batch["image"].to(device))
        predictions.append(metric(features, condition).cpu().numpy())
        targets.append(batch["target"].numpy())
        datasets.extend(batch["dataset"])

    frame = pd.DataFrame(
        {
            "prediction": np.concatenate(predictions),
            "target": np.concatenate(targets),
            "dataset": datasets,
        }
    )
    per_dataset = {}
    for name, rows in frame.groupby("dataset"):
        per_dataset[name] = {
            "srcc": float(
                stats.spearmanr(rows["prediction"], rows["target"]).correlation
            ),
            "plcc": float(
                stats.pearsonr(rows["prediction"], rows["target"]).statistic
            ),
        }
    return {
        "per_dataset": per_dataset,
        "macro_srcc": float(np.mean([row["srcc"] for row in per_dataset.values()])),
        "macro_plcc": float(np.mean([row["plcc"] for row in per_dataset.values()])),
        "classifier_accuracy": cls_correct / cls_count if cls_count else None,
    }


def load_classifier(config, device):
    classifier = DistortionClassifier(
        len(GROUPS), config.get("classifier_imagenet_pretrained", False)
    )
    checkpoint_path = config.get("classifier_checkpoint")
    if checkpoint_path:
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False
        )
        classifier.load_state_dict(checkpoint["model"])
    return classifier.to(device)


def make_optimizer(metric, classifier, config):
    groups = [{"params": metric.parameters(), "lr": config["lr"]}]
    if config["training_mode"] == "joint":
        groups.append(
            {
                "params": classifier.parameters(),
                "lr": config.get("classifier_lr", config["lr"]),
            }
        )
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def load_config(paths):
    config = {}
    for path in paths:
        config.update(yaml.safe_load(path.read_text()) or {})
    return config


def mlflow_context(config):
    if not config.get("mlflow", True):
        return nullcontext(), None

    import mlflow

    uri = str(config["mlflow_tracking_uri"])
    if "://" not in uri:
        uri = Path(uri).expanduser().resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["exp_name"])
    return mlflow.start_run(run_name=config["run_name"]), mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", nargs="+", required=True,
        help="base YAML followed by optional patch YAMLs",
    )
    args = parser.parse_args()
    config_paths = [Path(path) for path in args.config]
    config = load_config(config_paths)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = config["device"]
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto"
        else requested_device
    )

    mode = config["training_mode"]
    assert mode in {"hard", "frozen", "joint"}
    label_type = config.get("classifier_labels", "soft")
    assert label_type in {"hard", "soft"}

    encoder, image_size, feature_dim = load_image_encoder(
        config["backbone"], config.get("weights"), device
    )
    family = "siglip" if config["backbone"].startswith("siglip") else "clip"
    dataset = LabelConditionedDataset(
        config["data"],
        image_size,
        family,
        config["score_column"],
        config.get("classifier_image_size"),
    )
    train_set, val_set = split_by(
        dataset, config["split"], config["val_fraction"], seed
    )
    if config.get("limit") and config["limit"] < len(train_set):
        train_set = train_set.subset(
            train_set.rows.sample(config["limit"], random_state=seed)
        )

    sampler = make_sampler(train_set, config["sampler"], seed)
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
    )

    metric = LabelConditionedMetric(
        feature_dim,
        len(GROUPS),
        config["hidden_dim"],
        config["dropout"],
        config["fusion"],
    ).to(device)
    classifier = load_classifier(config, device) if mode != "hard" else None
    if mode == "frozen":
        classifier.eval().requires_grad_(False)

    quality_loss = nn.SmoothL1Loss(beta=0.1)
    counts = torch.bincount(
        torch.tensor(train_set.labels), minlength=len(GROUPS)
    )
    weights = torch.zeros(len(GROUPS))
    present = counts > 0
    weights[present] = len(train_set) / (len(GROUPS) * counts[present])
    classification_loss = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = make_optimizer(metric, classifier, config)

    print(
        f"{mode=} {config['fusion']=} {label_type=} "
        f"train={len(train_set)} val={len(val_set)} device={device}"
    )

    run_context, mlflow = mlflow_context(config)
    with run_context:
        if mlflow:
            for config_path in config_paths:
                mlflow.log_artifact(str(config_path), artifact_path="config")
            mlflow.log_params(
                {
                    **{
                        key: "none" if value is None else value
                        for key, value in config.items()
                    },
                    "train_size": len(train_set),
                    "val_size": len(val_set),
                    "feature_dim": feature_dim,
                }
            )

        checkpoint_dir = (
            Path(config["checkpoint_dir"]).expanduser()
            / config["exp_name"]
            / config["run_name"]
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_srcc = -float("inf")

        for epoch in range(1, config["epochs"] + 1):
            if mode == "hard":
                train_metrics = train_hard(
                    metric, encoder, train_loader, optimizer, quality_loss, device
                )
            elif mode == "frozen":
                train_metrics = train_frozen(
                    metric,
                    classifier,
                    encoder,
                    train_loader,
                    optimizer,
                    quality_loss,
                    label_type,
                    device,
                )
            else:
                train_metrics = train_joint(
                    metric,
                    classifier,
                    encoder,
                    train_loader,
                    optimizer,
                    quality_loss,
                    classification_loss,
                    label_type,
                    config["lambda_iqa"],
                    config["lambda_cls"],
                    device,
                )

            val_metrics = evaluate(
                metric, classifier, encoder, val_loader, mode, label_type, device
            )
            metrics = {
                **{f"train/{key}": value for key, value in train_metrics.items()},
                "val/macro_srcc": val_metrics["macro_srcc"],
                "val/macro_plcc": val_metrics["macro_plcc"],
            }
            for name, values in val_metrics["per_dataset"].items():
                metrics[f"val/{name}/srcc"] = values["srcc"]
                metrics[f"val/{name}/plcc"] = values["plcc"]
            if val_metrics["classifier_accuracy"] is not None:
                metrics["val/classifier_accuracy"] = val_metrics[
                    "classifier_accuracy"
                ]
            if mlflow:
                mlflow.log_metrics(metrics, step=epoch)

            summary = " ".join(
                f"{key}={value:.4f}" for key, value in metrics.items()
                if key in {"train/loss", "val/macro_srcc", "val/classifier_accuracy"}
            )
            print(f"epoch {epoch:02d} {summary}", flush=True)

            checkpoint = {
                "epoch": epoch,
                "metric": metric.state_dict(),
                "classifier": classifier.state_dict() if classifier else None,
                "optimizer": optimizer.state_dict(),
                "config": config,
                "classes": GROUPS,
                "feature_dim": feature_dim,
            }
            torch.save(checkpoint, checkpoint_dir / "last.pt")
            if val_metrics["macro_srcc"] > best_srcc:
                best_srcc = val_metrics["macro_srcc"]
                torch.save(checkpoint, checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
