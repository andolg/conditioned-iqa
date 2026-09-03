"""Evaluate the official pretrained ARNIQA metric on prepared IQA datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from arniqa import (
    ARNIQA_CROP_SIZE,
    ARNIQA_REGRESSOR_METADATA,
    embed_arniqa,
    load_arniqa_encoder,
    load_arniqa_regressor,
    scale_arniqa_score,
)
from dataset import IQADataset
from train import (
    _evaluation_progress_name,
    _quality_metrics,
    _reset_peak_memory,
    _peak_memory_mb,
    _synchronize,
)


@torch.no_grad()
def evaluate_pretrained_arniqa(
    encoder,
    regressor,
    loader,
    device: torch.device,
    regressor_dataset: str = "kadid10k",
    arniqa_batch_size: int = 64,
) -> dict:
    predictions, targets, references, datasets = [], [], [], []
    latency_ms_per_image = []
    inference_seconds = 0.0
    inference_images = 0
    _reset_peak_memory(device)

    for batch in tqdm(
        loader,
        desc=f"ARNIQA {_evaluation_progress_name(loader).removeprefix('evaluate ')}",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch_size = len(batch["target"])
        _synchronize(device)
        started = time.perf_counter()
        embedding = embed_arniqa(
            encoder,
            batch["arniqa_image"].to(device),
            batch["arniqa_image_ds"].to(device),
            chunk_size=arniqa_batch_size,
        )
        raw_score = regressor(embedding).reshape(-1)
        prediction = scale_arniqa_score(raw_score, regressor_dataset)
        _synchronize(device)
        elapsed = time.perf_counter() - started

        predictions.append(prediction.cpu().numpy())
        targets.append(batch["target"].numpy())
        references.extend(batch["reference"])
        datasets.extend(batch["dataset"])
        inference_seconds += elapsed
        inference_images += batch_size
        latency_ms_per_image.extend([1000.0 * elapsed / batch_size] * batch_size)

    scores = _quality_metrics(predictions, targets, references, datasets)
    scores.update({
        "latency_p50_ms": float(np.percentile(latency_ms_per_image, 50)),
        "latency_p95_ms": float(np.percentile(latency_ms_per_image, 95)),
        "peak_memory_mb": _peak_memory_mb(device),
        "images_per_second": float(inference_images / inference_seconds),
    })
    return scores


def _print_scores(csv: Path, scores: dict) -> None:
    print(f"\n{csv.parent.name}  ({csv})")
    for name, row in sorted(scores["per_dataset"].items()):
        print(
            f"  {name:<14s} n {row['n']:>6d}   "
            f"SRCC {row['srcc']:.4f}   PLCC {row['plcc']:.4f}"
        )
    if scores["srcc_per_reference"] is not None:
        print(
            f"  {'within-ref':<14s} {'':>8s}   "
            f"SRCC {scores['srcc_per_reference']:.4f} "
            f"({scores['n_references']} references)"
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
    parser.add_argument("--arniqa-weights", default=None,
                        help="local official ARNIQA.pth encoder; downloads it when omitted")
    parser.add_argument("--regressor-weights", default=None,
                        help="local official ARNIQA regressor; downloads it when omitted")
    parser.add_argument("--regressor-dataset", default="kadid10k",
                        choices=sorted(ARNIQA_REGRESSOR_METADATA),
                        help="IQA dataset used for the official pretrained regressor")
    parser.add_argument("--arniqa-batch-size", type=int, default=64,
                        help="maximum full/half ARNIQA crops encoded at once")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--score-column", default="scaled_subjective_score")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.arniqa_batch_size < 1:
        parser.error("--arniqa-batch-size must be positive")

    device = torch.device(
        args.device if args.device != "auto"
        else "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    encoder = load_arniqa_encoder(args.arniqa_weights, device)
    regressor = load_arniqa_regressor(
        args.regressor_weights, args.regressor_dataset, device
    )
    total_parameters = sum(
        parameter.numel()
        for module in (encoder, regressor)
        for parameter in module.parameters()
    )
    print(
        f"official ARNIQA with {args.regressor_dataset} regressor on {device}  "
        f"parameters {total_parameters / 1e6:.3f}M"
    )

    for value in args.data:
        csv = Path(value).expanduser()
        dataset = IQADataset(
            csv,
            image_size=ARNIQA_CROP_SIZE,
            backbone="clip",
            score_column=args.score_column,
            arniqa=True,
            arniqa_crop_size=ARNIQA_CROP_SIZE,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.workers
        )
        scores = evaluate_pretrained_arniqa(
            encoder,
            regressor,
            loader,
            device,
            regressor_dataset=args.regressor_dataset,
            arniqa_batch_size=args.arniqa_batch_size,
        )
        _print_scores(csv, scores)


if __name__ == "__main__":
    main()
