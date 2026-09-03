"""Generate the report's speed--accuracy Pareto plots from selected result rows.

The values below are a compact snapshot of the project's shared results sheet.
They are kept in the script so the figures can be regenerated without network
access. Pareto dominance maximizes both reported external macro SRCC and FPS.
Point area encodes total parameter count and colour encodes peak GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


@dataclass(frozen=True)
class Result:
    label: str
    fps: float
    srcc: float
    parameters_m: float
    memory_mb: float


TEXT_HEADS = [
    Result("Image-only", 308.60, 0.5928, 86.00, 347.4),
    Result("Native interaction", 333.92, 0.6191, 86.33, 349.3),
    Result("Calibrated", 292.51, 0.6188, 86.33, 364.1),
    Result("MDTVSFA", 231.91, 0.6169, 86.33, 364.3),
    Result("Residual", 325.69, 0.5882, 86.53, 350.8),
    Result("Patch residual", 106.47, 0.6190, 86.93, 375.5),
    Result("Cross-attention", 107.69, 0.6224, 86.99, 377.4),
    Result("FiLM", 207.38, 0.6112, 86.33, 364.1),
    Result("Two-layer", 205.63, 0.6246, 86.39, 365.4),
    Result("Patch scoring", 140.53, 0.4461, 86.40, 365.3),
]

TEXT_FINALISTS = [
    Result("B pooled image", 308.60, 0.5928, 86.00, 347.4),
    Result("B pooled text", 333.92, 0.6191, 86.33, 349.3),
    Result("B 5-view image", 96.60, 0.6310, 86.13, 401.0),
    Result("B 5-view text", 99.37, 0.6539, 86.53, 408.4),
    Result("B 5-view calibrated", 79.93, 0.6612, 86.53, 408.4),
    Result("L pooled text", 47.07, 0.6292, 304.17, 1224.0),
    Result("L 5-view text", 10.48, 0.6475, 304.37, 1382.9),
    Result("SigLIP pooled text", 103.72, 0.6258, 316.68, 1249.2),
    Result("SigLIP 5-view text", 22.68, 0.6699, 316.88, 1304.0),
]

DATASET_LABEL = [
    Result("B image-only", 335.65, 0.4861, 86.00, 347.9),
    Result("L image-only", 47.40, 0.5111, 303.77, 1206.6),
    Result("KADID additive", 552.91, 0.4817, 86.02, 661.2),
    Result("KADID input FiLM", 484.81, 0.5081, 86.05, 661.4),
    Result("KADID residual gate", 559.96, 0.5230, 86.20, 651.9),
    Result("Mix input FiLM", 454.35, 0.6136, 86.05, 661.4),
    Result("Mix residual gate", 529.77, 0.6314, 86.20, 663.2),
    Result("Mix hypernetwork", 353.22, 0.6380, 86.00, 660.8),
    Result("Mix hidden FiLM", 558.61, 0.6106, 86.02, 661.0),
    Result("Mix patch attention", 391.43, 0.6107, 86.00, 679.7),
]

CLASSIFIER = [
    Result("Zero condition", 109.57, 0.4981, 86.00, 346.6),
    Result("Oracle hard label", 110.15, 0.5025, 86.00, 346.6),
    Result("Predicted hard label", 101.09, 0.4753, 97.18, 407.2),
    Result("Predicted soft label", 79.55, 0.4775, 97.18, 407.2),
    Result("Layer 3 projected", 74.44, 0.5200, 97.26, 407.5),
    Result("Layer 4 projected", 69.27, 0.4907, 97.27, 407.6),
    Result("Layer 3 normalized", 81.28, 0.5081, 97.31, 407.7),
    Result("Layer 4 normalized", 81.42, 0.5068, 97.38, 408.7),
    Result("CLIP-B + ARNIQA", 26.86, 0.5977, 109.65, 1445.8),
    Result("CLIP-L + ARNIQA", 17.31, 0.5645, 327.43, 2455.2),
    Result("Standalone ARNIQA", 35.28, 0.5859, 23.51, 1105.0),
]

LABEL_OFFSETS = {
    "Image-only": (-7, 7),
    "Native interaction": (-7, -12),
    "Calibrated": (-7, 8),
    "MDTVSFA": (7, -15),
    "Residual": (-7, -13),
    "FiLM": (7, -15),
    "B pooled image": (-6, -10),
    "B pooled text": (-6, -10),
    "B 5-view image": (7, 5),
    "B 5-view text": (7, -13),
    "B 5-view calibrated": (7, 7),
    "L pooled text": (7, -12),
    "L 5-view text": (7, 5),
    "SigLIP pooled text": (7, -13),
    "SigLIP 5-view text": (7, 5),
    "B image-only": (7, 5),
    "L image-only": (7, -13),
    "KADID additive": (-7, -10),
    "KADID input FiLM": (-7, -13),
    "KADID residual gate": (7, 5),
    "Mix input FiLM": (7, -14),
    "Mix residual gate": (7, 6),
    "Mix hypernetwork": (7, -15),
    "Mix hidden FiLM": (7, 6),
    "Mix patch attention": (7, -18),
    "Zero condition": (7, -14),
    "Oracle hard label": (7, 6),
    "Predicted hard label": (7, 5),
    "Predicted soft label": (7, -15),
    "Layer 3 projected": (7, 6),
    "Layer 4 projected": (7, -12),
    "Layer 3 normalized": (7, 8),
    "Layer 4 normalized": (7, -18),
    "CLIP-B + ARNIQA": (7, 5),
    "CLIP-L + ARNIQA": (7, -14),
    "Standalone ARNIQA": (7, 5),
}


def pareto_front(results: list[Result]) -> list[Result]:
    front = []
    for candidate in results:
        dominated = any(
            other.fps >= candidate.fps
            and other.srcc >= candidate.srcc
            and (other.fps > candidate.fps or other.srcc > candidate.srcc)
            for other in results
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda result: result.fps)


def point_size(parameters_m: float) -> float:
    return 42.0 + 0.55 * parameters_m


def draw_panel(ax, results: list[Result], title: str, normalizer: Normalize) -> None:
    scatter = ax.scatter(
        [result.fps for result in results],
        [result.srcc for result in results],
        s=[point_size(result.parameters_m) for result in results],
        c=[result.memory_mb for result in results],
        cmap="viridis",
        norm=normalizer,
        edgecolor="black",
        linewidth=0.65,
        alpha=0.9,
        zorder=3,
    )
    front = pareto_front(results)
    ax.plot(
        [result.fps for result in front],
        [result.srcc for result in front],
        color="#c43c39",
        linestyle="--",
        linewidth=1.5,
        marker="none",
        label="FPS--SRCC Pareto front",
        zorder=2,
    )
    for index, result in enumerate(results):
        offset_x, offset_y = LABEL_OFFSETS.get(result.label, (5, 5 if index % 2 == 0 else -10))
        ax.annotate(
            result.label,
            (result.fps, result.srcc),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8.0,
            ha="right" if offset_x < 0 else "left",
            va="bottom" if offset_y > 0 else "top",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Throughput (images/s, logarithmic scale)")
    ax.set_ylabel("External macro SRCC")
    ax.set_title(title)
    ax.grid(True, which="both", color="#d7d7d7", linewidth=0.6, alpha=0.8)
    ax.margins(x=0.13, y=0.10)
    ax.legend(loc="best", fontsize=8, frameon=True)
    return scatter


def add_size_legend(ax) -> None:
    handles = [
        ax.scatter([], [], s=point_size(value), facecolor="none", edgecolor="black", linewidth=0.7)
        for value in (25, 100, 300)
    ]
    first = ax.get_legend()
    legend = ax.legend(handles, ["25M", "100M", "300M"], title="Total parameters", loc="lower left", fontsize=7.5, title_fontsize=8, frameon=True)
    if first is not None:
        ax.add_artist(first)
    ax.add_artist(legend)


def save_single(results: list[Result], title: str, filename: str, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    normalizer = Normalize(min(result.memory_mb for result in results), max(result.memory_mb for result in results))
    scatter = draw_panel(axis, results, title, normalizer)
    add_size_legend(axis)
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("Peak GPU memory (MB)")
    figure.savefig(output_dir / filename, dpi=240, bbox_inches="tight")
    plt.close(figure)


def save_text(output_dir: Path) -> None:
    combined = TEXT_HEADS + TEXT_FINALISTS
    normalizer = Normalize(min(result.memory_mb for result in combined), max(result.memory_mb for result in combined))
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 10.0), constrained_layout=True)
    first = draw_panel(axes[0], TEXT_HEADS, "Pooled-head and objective ablations", normalizer)
    draw_panel(axes[1], TEXT_FINALISTS, "Backbone and multi-view finalists", normalizer)
    add_size_legend(axes[1])
    colorbar = figure.colorbar(first, ax=axes, pad=0.015)
    colorbar.set_label("Peak GPU memory (MB)")
    figure.savefig(output_dir / "text_conditioning_tradeoff.png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    save_text(output_dir)
    save_single(DATASET_LABEL, "Dataset-label conditioning", "dataset_label_tradeoff.png", output_dir)
    save_single(CLASSIFIER, "Classifier and ARNIQA conditioning", "classifier_conditioning_tradeoff.png", output_dir)


if __name__ == "__main__":
    main()
