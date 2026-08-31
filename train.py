"""Train an MLP on frozen CLIP features to predict image quality.

    python prepare_data.py ~/iqa-data/kadid10k          # once, writes labels.csv
    python train.py --data ~/iqa-data/kadid10k/labels.csv
    python train.py --data ~/iqa-data/kadid10k/labels.csv --sampler balanced

    image -> frozen CLIP -> pooled embedding -> MLP -> quality score

The backbone never trains; only the MLP does, which is a few hundred
thousand parameters over a representation that costs nothing to keep. That
makes this the row every other design is measured against: if a change does
not beat it, the change is not doing anything.

Reports SRCC and PLCC on the held-out split each epoch, one row per dataset
plus their macro and the worst of them. SRCC is the number IQA papers report
— it only cares about ranking, which is what a quality metric is for.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader

from dataset import IQADataset, make_sampler, split_by
from hf_mirror_utils import load_transformers_model_from_mirrors

BACKBONES = {
    "clip-base": ("openai/clip-vit-base-patch16", 224),
    "clip-large": ("openai/clip-vit-large-patch14-336", 336),
    "siglip": ("google/siglip-large-patch16-256", 256),
    "siglip2-base": ("google/siglip2-base-patch16-224", 224),
    "siglip2-large": ("google/siglip2-large-patch16-256", 256),
}


class QualityMLP(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear -> one number."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_backbone(name: str, weights: str | None, device: torch.device):
    """The frozen encoder. `weights` is a local directory, if you have one."""
    from transformers import CLIPVisionModel, SiglipVisionModel

    hf_id, image_size = BACKBONES[name]
    model_class = SiglipVisionModel if name.startswith("siglip") else CLIPVisionModel
    if weights:
        model = model_class.from_pretrained(weights, local_files_only=True)
    else:
        model = load_transformers_model_from_mirrors(model_class, hf_id)
    model = model.eval().requires_grad_(False).to(device)
    return model, image_size, model.config.hidden_size


@torch.no_grad()
def embed(backbone, images: torch.Tensor) -> torch.Tensor:
    return backbone(pixel_values=images).pooler_output.float()


def evaluate(backbone, head, loader, device) -> dict:
    """SRCC and PLCC per dataset, their macro, and SRCC within each reference.

    Per dataset rather than pooled, because pooling measures something else.
    Two releases put their subjects on different scales and score different
    pictures, so a correlation over the union partly measures the offset
    between them: on a KADID + KonIQ + SPAQ run the pooled SRCC came out at
    0.766, below every one of the three sets it is made of. The macro is the
    mean of the per-dataset numbers and `worst` is the lowest of them — a mean
    alone hides a collapse on one set.

    The second number exists because of PIPAL. Its scores are Elo ratings
    from pairwise comparisons, and every image starts at 1400 — so a score
    says how a restoration ranks against other restorations *of the same
    picture*, not how good the picture is. Measured on the data: 99.9% of
    the variance sits inside a reference, and the 200 reference means span
    22 points against a 622-point spread within one. Correlating across
    references mixes two different questions; averaging the per-reference
    correlations asks only the one the ratings can answer.
    """
    head.eval()
    predictions, targets, references, datasets = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            features = embed(backbone, batch["image"].to(device))
            predictions.append(head(features).cpu().numpy())
            targets.append(batch["target"].numpy())
            references.extend(batch["reference"])
            datasets.extend(batch["dataset"])
    head.train()
    p, t = np.concatenate(predictions), np.concatenate(targets)
    frame = pd.DataFrame({"p": p, "t": t, "ref": references, "dataset": datasets})

    per_dataset = {}
    for name, group in frame.groupby("dataset"):
        if len(group) < 2 or group["t"].nunique() < 2:
            continue
        per_dataset[name] = {
            "srcc": float(stats.spearmanr(group["p"], group["t"]).correlation),
            "plcc": float(stats.pearsonr(group["p"], group["t"]).statistic),
            "n": int(len(group)),
        }

    per_reference = []
    for _, group in frame.groupby("ref"):
        if len(group) >= 8 and group["t"].nunique() > 1:
            per_reference.append(stats.spearmanr(group["p"], group["t"]).correlation)

    srccs = [scores["srcc"] for scores in per_dataset.values()]
    return {
        "per_dataset": per_dataset,
        "macro_srcc": float(np.mean(srccs)) if srccs else None,
        "macro_plcc": float(np.mean([s["plcc"] for s in per_dataset.values()])) if srccs else None,
        "worst_srcc": float(min(srccs)) if srccs else None,
        "worst_dataset": min(per_dataset, key=lambda k: per_dataset[k]["srcc"]) if srccs else None,
        "srcc_per_reference": float(np.mean(per_reference)) if per_reference else None,
        "n_references": len(per_reference),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the CSV prepare_data.py wrote")
    ap.add_argument("--backbone", default="clip-base", choices=sorted(BACKBONES))
    ap.add_argument("--weights", default=None, help="local checkpoint directory, if not the hub")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--split", default="reference", choices=["reference", "random"])
    ap.add_argument("--score-column", default="scaled_subjective_score",
                    help="which column of the CSV to regress")
    ap.add_argument("--sampler", default="random",
                    choices=["random", "balanced", "by_level", "by_dataset"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N training images, drawn at random")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="save the trained head here")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device if args.device != "auto"
        else "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    backbone, image_size, feature_dim = load_backbone(args.backbone, args.weights, device)
    family = "siglip" if args.backbone.startswith("siglip") else "clip"

    dataset = IQADataset(args.data, image_size=image_size, backbone=family,
                         score_column=args.score_column)
    train_set, val_set = split_by(dataset, args.split, fraction=0.2, seed=args.seed)
    if args.limit and args.limit < len(train_set.rows):
        # Sampled, not the first N rows: the CSV is ordered by dataset and then
        # by reference, so a head() would train on one dataset and a handful of
        # its pictures without saying so.
        train_set = train_set.subset(
            train_set.rows.sample(args.limit, random_state=args.seed)
        )

    sampler = make_sampler(train_set, args.sampler, seed=args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=args.workers,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    head = QualityMLP(feature_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    print(f"{args.backbone} at {image_size}px on {device}, {feature_dim}-d features")
    print(f"train {len(train_set)}  held out {len(val_set)}  "
          f"(split by {args.split}, sampling {args.sampler})")
    print(f"{sum(p.numel() for p in head.parameters()):,} trainable parameters "
          "— the backbone is frozen")

    for epoch in range(args.epochs):
        losses = []
        for batch in train_loader:
            features = embed(backbone, batch["image"].to(device))
            loss = loss_fn(head(features), batch["target"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scores = evaluate(backbone, head, val_loader, device)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}", flush=True)
        for name, row in sorted(scores["per_dataset"].items()):
            print(f"    {name:<14s} n {row['n']:>6d}   "
                  f"SRCC {row['srcc']:.4f}   PLCC {row['plcc']:.4f}")
        if len(scores["per_dataset"]) > 1:
            print(f"    {'macro':<14s} {'':>8s}   SRCC {scores['macro_srcc']:.4f}   "
                  f"PLCC {scores['macro_plcc']:.4f}   "
                  f"worst {scores['worst_srcc']:.4f} on {scores['worst_dataset']}")
        if scores["srcc_per_reference"] is not None:
            print(f"    {'within-ref':<14s} {'':>8s}   SRCC {scores['srcc_per_reference']:.4f}"
                  f"   ({scores['n_references']} references)")

    if args.out:
        torch.save({"head": head.state_dict(), "backbone": args.backbone,
                    "feature_dim": feature_dim}, args.out)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
