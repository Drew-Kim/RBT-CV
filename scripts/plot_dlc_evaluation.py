"""Create a compact chart from one DeepLabCut evaluation-results CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    if len(results) != 1:
        raise ValueError(f"Expected one evaluation row, found {len(results)} in {args.results_csv}")
    row = results.iloc[0]

    train_fraction = float(row["%Training dataset"])
    shuffle = int(row["Shuffle number"])
    epochs = int(row["Training epochs"])
    pcutoff = float(row["pcutoff"])

    figure, (error_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10.6, 5.6))
    figure.patch.set_facecolor("white")
    figure.suptitle(
        f"RBT-CV six-landmark model evaluation — snapshot-best-{epochs}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.925,
        f"PyTorch · ResNet-50 · shuffle {shuffle} · {train_fraction:.0%} train split · p-cutoff {pcutoff:.2f}",
        ha="center",
        color="#485260",
        fontsize=10,
    )

    labels = ["Train", "Test"]
    colors = ["#65a30d", "#2563eb"]

    rmse = [float(row["train rmse"]), float(row["test rmse"])]
    bars = error_axis.bar(labels, rmse, color=colors, width=0.58)
    error_axis.set_title("Localization error", fontweight="bold")
    error_axis.set_ylabel("RMSE (pixels) — lower is better")
    error_axis.set_ylim(0, max(rmse) * 1.35)
    error_axis.grid(axis="y", alpha=0.2)
    error_axis.set_axisbelow(True)
    for bar, value in zip(bars, rmse):
        error_axis.text(bar.get_x() + bar.get_width() / 2, value + max(rmse) * 0.04, f"{value:.2f}", ha="center", fontweight="bold")

    metrics = ["mAP", "mAR"]
    train_values = [float(row["train mAP"]), float(row["train mAR"])]
    test_values = [float(row["test mAP"]), float(row["test mAR"])]
    positions = range(len(metrics))
    width = 0.34
    train_bars = accuracy_axis.bar([x - width / 2 for x in positions], train_values, width, color=colors[0], label="Train")
    test_bars = accuracy_axis.bar([x + width / 2 for x in positions], test_values, width, color=colors[1], label="Test")
    accuracy_axis.set_title("Detection quality", fontweight="bold")
    accuracy_axis.set_ylabel("Percent — higher is better")
    accuracy_axis.set_xticks(list(positions), metrics)
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.grid(axis="y", alpha=0.2)
    accuracy_axis.set_axisbelow(True)
    accuracy_axis.legend(frameon=False, loc="lower right")
    for bars in (train_bars, test_bars):
        for bar in bars:
            value = bar.get_height()
            accuracy_axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}%", ha="center", fontsize=9, fontweight="bold")

    figure.text(
        0.5,
        0.03,
        "Evaluated checkpoint: snapshot-best-180.pt. Metrics are DeepLabCut's reported held-out evaluation metrics.",
        ha="center",
        color="#485260",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.02, 0.08, 0.98, 0.89))

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(args.output_stem.with_suffix(".svg"))
    print(args.output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
