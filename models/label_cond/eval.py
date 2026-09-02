import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader, default_collate
from torchvision import transforms
from tqdm import tqdm

from dataset import IQADataset
from models.dist_classifier.model import DistortionClassifier
from models.label_cond.dataset import GROUPS, GROUP_TO_LABEL
from models.label_cond.model import LabelConditionedMetric, encode_images, load_image_encoder


TABLE_COLUMNS = [
    "Design", "Description", "Backbone", "Train datasets", "Epochs", "Seed",
    "Baseline", "Latency p50 (ms)", "Latency p95 (ms)", "Peak memory (MB)", "FPS",
    "KADID-10k SRCC", "KADID-10k PLCC", "SPAQ SRCC", "SPAQ PLCC",
    "GFIQA-20k SRCC", "GFIQA-20k PLCC", "PIPAL SRCC", "PIPAL PLCC",
    "AIGCIQA2023 SRCC", "AIGCIQA2023 PLCC", "TID2013 SRCC", "TID2013 PLCC",
    "CSIQ SRCC", "CSIQ PLCC", "CID2013 SRCC", "CID2013 PLCC",
    "KonIQ-10k SRCC", "KonIQ-10k PLCC", "CLIVE SRCC", "CLIVE PLCC",
    "AGIQA-3K SRCC", "AGIQA-3K PLCC", "UHD-IQA SRCC", "UHD-IQA PLCC",
    "Avg validation SRCC", "Avg validation PLCC", "Avg test SRCC", "Avg test PLCC",
    "Avg val+test SRCC", "Avg val+test PLCC", "Parameters", "GFLOPs", "run_id", "run_name",
]


class EvaluationDataset(IQADataset):
    def __init__(self, csv_path, image_size, backbone, score_column, use_classifier):
        rows = pd.read_csv(Path(csv_path).expanduser())
        super().__init__(rows, image_size, backbone, score_column)
        groups = rows["group"] if "group" in rows else pd.Series("authentic", index=rows.index)
        self.labels = [
            GROUP_TO_LABEL.get(str(group), GROUP_TO_LABEL["authentic"])
            if pd.notna(group) else GROUP_TO_LABEL["authentic"]
            for group in groups
        ]
        self.use_classifier = use_classifier
        self.classifier_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["group"] = torch.tensor(self.labels[index])
        if self.use_classifier:
            image = Image.open(self.rows.iloc[index]["path"]).convert("RGB")
            sample["classifier_image"] = self.classifier_transform(image)
        return sample


def collate_native(batch):
    classifier_images = [sample.pop("classifier_image", None) for sample in batch]
    result = default_collate(batch)
    if classifier_images[0] is not None:
        result["classifier_image"] = classifier_images
    return result


def one_hot(labels):
    return F.one_hot(labels, num_classes=len(GROUPS)).float()


def classify_native(classifier, images, device):
    return torch.cat([
        classifier(image.unsqueeze(0).to(device)) for image in images
    ])


def condition_for(batch, source, classifier, device):
    labels = batch["group"].to(device)
    if source.get("zero_labels", False):
        return torch.zeros(len(labels), len(GROUPS), device=device)
    if source["training_mode"] == "hard":
        return one_hot(labels)
    logits = classify_native(classifier, batch["classifier_image"], device)
    if source.get("classifier_labels", "soft") == "soft":
        return logits.softmax(dim=1)
    return one_hot(logits.argmax(dim=1))


@torch.no_grad()
def evaluate_dataset(encoder, metric, classifier, loader, source, device):
    predictions, targets = [], []
    started = time.perf_counter()
    for batch in tqdm(loader, desc=loader.dataset.name, leave=False):
        features = encode_images(encoder, batch["image"].to(device))
        condition = condition_for(batch, source, classifier, device)
        predictions.append(metric(features, condition).cpu().numpy())
        targets.append(batch["target"].numpy())
    elapsed = time.perf_counter() - started
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    return {
        "images": len(target),
        "elapsed_seconds": elapsed,
        "images_per_second": len(target) / elapsed,
        "srcc": float(stats.spearmanr(prediction, target).correlation),
        "plcc": float(stats.pearsonr(prediction, target).statistic),
    }


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction / 100 * len(ordered)) - 1))
    return ordered[index]


def measure_latency_memory(forward, device, warmup=10, repeats=50):
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        forward()
    if use_cuda:
        torch.cuda.synchronize(device)

    samples = []
    for _ in range(repeats):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            forward()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(start.elapsed_time(end))
        else:
            started = time.perf_counter()
            forward()
            samples.append((time.perf_counter() - started) * 1000)
    memory = torch.cuda.max_memory_allocated(device) / 1024**2 if use_cuda else math.nan
    return round(percentile(samples, 50), 3), round(percentile(samples, 95), 3), round(memory, 1)


def measure_flops(forward):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        forward()
    return float(counter.get_total_flops())


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def metric_value(client, run_id, key, step):
    return next(item.value for item in client.get_metric_history(run_id, key) if item.step == step)


def source_runs(config):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    from mlflow import MlflowClient

    uri = str(config["mlflow_tracking_uri"])
    if "://" not in uri:
        uri = Path(uri).expanduser().resolve().as_uri()
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(config["experiment"])
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )
    result = []
    for run in runs:
        history = client.get_metric_history(run.info.run_id, "val/macro_srcc")
        if not history:
            continue
        best = max(history, key=lambda item: item.value)
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        checkpoint_path = (
            Path(config["checkpoint_root"]).expanduser()
            / config["experiment"] / name / "best.pt"
        )
        result.append({
            "run_id": run.info.run_id,
            "run_name": name,
            "best_epoch": best.step,
            "best_srcc": best.value,
            "val_srcc": metric_value(client, run.info.run_id, "val/kadid10k/srcc", best.step),
            "val_plcc": metric_value(client, run.info.run_id, "val/kadid10k/plcc", best.step),
            "checkpoint": str(checkpoint_path),
        })
    return result


def load_model(run, device):
    checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != run["best_epoch"]:
        raise ValueError(
            f'{run["run_name"]}: MLflow best epoch {run["best_epoch"]} '
            f'does not match checkpoint epoch {checkpoint["epoch"]}'
        )
    source = checkpoint["config"]
    encoder, image_size, feature_dim = load_image_encoder(
        source["backbone"], source.get("weights"), device
    )
    metric = LabelConditionedMetric(
        feature_dim, len(GROUPS), source["hidden_dim"], source["dropout"], source["fusion"]
    ).to(device)
    metric.load_state_dict(checkpoint["metric"])
    metric.eval()

    classifier = None
    if source["training_mode"] != "hard":
        classifier = DistortionClassifier(len(GROUPS), pretrained=False).to(device)
        classifier.load_state_dict(checkpoint["classifier"])
        classifier.eval()
    return checkpoint, source, encoder, metric, classifier, image_size


def system_metrics(config, source, encoder, metric, classifier, image_size, device):
    @torch.no_grad()
    def forward():
        images = torch.randn(1, 3, image_size, image_size, device=device)
        features = encode_images(encoder, images)
        if source.get("zero_labels", False):
            condition = torch.zeros(1, len(GROUPS), device=device)
        elif source["training_mode"] == "hard":
            condition = one_hot(torch.zeros(1, dtype=torch.long, device=device))
        else:
            classifier_images = torch.randn(
                1, 3, *config["classifier_benchmark_size"], device=device
            )
            logits = classifier(classifier_images)
            condition = (
                logits.softmax(dim=1)
                if source.get("classifier_labels", "soft") == "soft"
                else one_hot(logits.argmax(dim=1))
            )
        return metric(features, condition)

    latency_p50, latency_p95, peak_memory = measure_latency_memory(
        forward, device, config["latency_warmup"], config["latency_repeats"]
    )
    modules = [encoder, metric] + ([classifier] if classifier is not None else [])
    return {
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "peak_memory_mb": peak_memory,
        "parameters": sum(parameter.numel() for module in modules for parameter in module.parameters()),
        "gflops": measure_flops(forward) / 1e9,
    }


def mean(values):
    return float(np.mean(values)) if values else None


def rounded(value, digits=4):
    return "" if value is None else round(value, digits)


def aggregate(config, runs):
    intermediate = Path(config["intermediate_root"]) / config["experiment"]
    rows = []
    for run in runs:
        key = run["key"]
        root = intermediate / key
        metadata_path, system_path = root / "metadata.json", root / "system.json"
        if not metadata_path.exists() or not system_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        system = json.loads(system_path.read_text())
        source = metadata["source"]
        scores = {}
        for dataset_key, dataset in config["datasets"].items():
            path = root / f"{dataset_key}.json"
            if path.exists():
                scores[dataset_key] = json.loads(path.read_text())

        row = {column: "" for column in TABLE_COLUMNS}
        row.update({
            "Backbone": config["backbone_names"].get(source["backbone"], source["backbone"]),
            "Train datasets": ", ".join(config["train_datasets"]),
            "Epochs": source["epochs"],
            "Seed": source["seed"],
            "Baseline": "baseline" if source.get("zero_labels", False) else "variant",
            "Latency p50 (ms)": rounded(system["latency_p50_ms"], 1),
            "Latency p95 (ms)": rounded(system["latency_p95_ms"], 1),
            "Peak memory (MB)": rounded(system["peak_memory_mb"], 1),
            "FPS": rounded(scores.get("kadid10k", {}).get("images_per_second"), 2),
            "Parameters": f'{system["parameters"] / 1e6:.2f}M',
            "GFLOPs": rounded(system["gflops"]),
            "run_id": metadata["run_id"],
            "run_name": metadata["run_name"],
        })

        validation_srcc, validation_plcc, test_srcc, test_plcc = [], [], [], []
        for dataset_key, dataset in config["datasets"].items():
            if dataset_key == "kadid10k":
                srcc, plcc = metadata["validation_srcc"], metadata["validation_plcc"]
            elif dataset_key in scores:
                srcc, plcc = scores[dataset_key]["srcc"], scores[dataset_key]["plcc"]
            else:
                continue
            row[f'{dataset["name"]} SRCC'] = rounded(srcc)
            row[f'{dataset["name"]} PLCC'] = rounded(plcc)
            if dataset.get("enabled", True):
                if dataset["group"] == "validation":
                    validation_srcc.append(srcc)
                    validation_plcc.append(plcc)
                else:
                    test_srcc.append(srcc)
                    test_plcc.append(plcc)

        avg_val_srcc, avg_val_plcc = mean(validation_srcc), mean(validation_plcc)
        avg_test_srcc, avg_test_plcc = mean(test_srcc), mean(test_plcc)
        row.update({
            "Avg validation SRCC": rounded(avg_val_srcc),
            "Avg validation PLCC": rounded(avg_val_plcc),
            "Avg test SRCC": rounded(avg_test_srcc),
            "Avg test PLCC": rounded(avg_test_plcc),
            "Avg val+test SRCC": rounded(mean([x for x in (avg_val_srcc, avg_test_srcc) if x is not None])),
            "Avg val+test PLCC": rounded(mean([x for x in (avg_val_plcc, avg_test_plcc) if x is not None])),
        })
        rows.append(row)

    output = Path(config["table_root"]) / config["experiment"] / "aggregated.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def slug(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def evaluate_run(task):
    config, run = task
    device = torch.device(config["device"])
    root = Path(config["intermediate_root"]) / config["experiment"] / run["key"]
    metadata_path = root / "metadata.json"
    system_path = root / "system.json"
    enabled = {
        key: value for key, value in config["datasets"].items()
        if value.get("enabled", True)
    }
    pending = [key for key in enabled if not (root / f"{key}.json").exists()]
    if metadata_path.exists() and system_path.exists() and not pending:
        return run["run_name"]

    print(f'{run["run_name"]}: {len(pending)} datasets pending', flush=True)
    checkpoint, source, encoder, metric, classifier, image_size = load_model(run, device)
    family = "siglip" if source["backbone"].startswith("siglip") else "clip"
    use_classifier = classifier is not None

    if not metadata_path.exists():
        atomic_json(metadata_path, {
            "run_id": run["run_id"],
            "run_name": run["run_name"],
            "best_epoch": run["best_epoch"],
            "best_validation_srcc": run["best_srcc"],
            "validation_srcc": run["val_srcc"],
            "validation_plcc": run["val_plcc"],
            "checkpoint": run["checkpoint"],
            "source": source,
        })
    if not system_path.exists():
        atomic_json(
            system_path,
            system_metrics(config, source, encoder, metric, classifier, image_size, device),
        )

    for dataset_key in pending:
        dataset_config = enabled[dataset_key]
        dataset = EvaluationDataset(
            dataset_config["path"], image_size, family, source["score_column"],
            use_classifier,
        )
        dataset.name = f'{run["run_name"]}/{dataset_config["name"]}'
        loader = DataLoader(
            dataset, batch_size=config["batch_size"], num_workers=config["workers"],
            pin_memory=device.type == "cuda", collate_fn=collate_native,
        )
        result = evaluate_dataset(encoder, metric, classifier, loader, source, device)
        result["dataset"] = dataset_key
        atomic_json(root / f"{dataset_key}.json", result)
        print(
            f'{run["run_name"]}/{dataset_config["name"]}: '
            f'SRCC {result["srcc"]:.4f} PLCC {result["plcc"]:.4f}',
            flush=True,
        )

    del checkpoint, encoder, metric, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run["run_name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/label_cond/eval.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    runs = source_runs(config)
    for run in runs:
        run["key"] = f'{slug(run["run_name"])}__{run["run_id"][:8]}'

    context = mp.get_context("spawn")
    pool_size = min(config["mp_pool_size"], len(runs))
    with ProcessPoolExecutor(max_workers=pool_size, mp_context=context) as pool:
        futures = [pool.submit(evaluate_run, (config, run)) for run in runs]
        for completed, future in enumerate(as_completed(futures), 1):
            print(f'[{completed}/{len(runs)}] {future.result()}: complete', flush=True)

    aggregate(config, runs)
    print(Path(config["table_root"]) / config["experiment"] / "aggregated.csv")


if __name__ == "__main__":
    main()
