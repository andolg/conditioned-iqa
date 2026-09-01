import argparse
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from models.dist_classifier.dataset import DistortionDataset, split_by_reference
from models.dist_classifier.model import DistortionClassifier

# From prepare_data.py
GROUPS = (
    "compression", "generative", "blur", "noise",
    "color", "tone", "spatial", "authentic",
)
CLASS_NAMES = GROUPS


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        count += len(targets)
        total_loss += loss.item() * len(targets)
        correct += (logits.argmax(1) == targets).sum().item()

    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
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

    train_rows, val_rows = split_by_reference(
        config["data"], config["val_fraction"], seed
    )
    group_to_label = {group: index for index, group in enumerate(GROUPS)}
    train_set = DistortionDataset(
        train_rows, group_to_label, config.get("image_size"), train=True
    )
    val_set = DistortionDataset(
        val_rows, group_to_label, config.get("image_size"), train=False
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        pin_memory=device.type == "cuda",
    )

    num_classes = len(CLASS_NAMES)
    model = DistortionClassifier(num_classes, config["pretrained"]).to(device)
    counts = torch.bincount(torch.tensor(train_set.labels), minlength=num_classes)
    class_weights = torch.zeros(num_classes)
    present = counts > 0
    class_weights[present] = len(train_set) / (num_classes * counts[present])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    tracking_uri = Path(config["mlflow_tracking_uri"]).expanduser().resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["exp_name"])

    with mlflow.start_run(run_name=config["run_name"]) as run:
        checkpoint_dir = Path("weights") / config["exp_name"] / config["run_name"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_artifact(str(config_path), artifact_path="config")
        mlflow.log_params({
            **config,
            "device_used": str(device),
            "train_size": len(train_set),
            "val_size": len(val_set),
            "classes": ",".join(CLASS_NAMES),
        })

        best_accuracy = -1.0
        for epoch in range(1, config["epochs"] + 1):
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_loss, val_accuracy = run_epoch(
                model, val_loader, criterion, device
            )
            metrics = {
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "val/loss": val_loss,
                "val/accuracy": val_accuracy,
            }
            mlflow.log_metrics(metrics, step=epoch)
            print(
                f"epoch {epoch:02d} "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "classes": CLASS_NAMES,
            }
            torch.save(checkpoint, checkpoint_dir / "last.pt")
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save(checkpoint, checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
