"""Evaluate a released classifier-conditioned checkpoint without MLflow or training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    # These helpers live with the label-conditioning implementation.  Keeping
    # this adapter small means the released checkpoint remains self-contained:
    # its config and both metric/classifier state dicts are loaded directly.
    from models.label_cond.eval import EvaluationDataset, collate_native, evaluate_dataset, load_model

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = checkpoint["config"]
    device = torch.device(
        args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    )
    run = {"checkpoint": str(args.checkpoint), "best_epoch": checkpoint["epoch"]}
    _, source, encoder, metric, classifier, image_size = load_model(run, device)
    family = "siglip" if source["backbone"].startswith("siglip") else "clip"
    scores = []
    try:
        for csv_path in args.data:
            dataset = EvaluationDataset(
                csv_path, image_size, family, source["score_column"], classifier is not None
            )
            dataset.name = Path(csv_path).parent.name
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate_native,
            )
            result = evaluate_dataset(encoder, metric, classifier, loader, source, device)
            scores.append(result)
            print(f"{Path(csv_path).parent.name}: SRCC {result['srcc']:.4f}  PLCC {result['plcc']:.4f}")
    finally:
        del encoder, metric, classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(
        f"macro: SRCC {np.mean([x['srcc'] for x in scores]):.4f}  "
        f"PLCC {np.mean([x['plcc'] for x in scores]):.4f}"
    )


if __name__ == "__main__":
    main()
