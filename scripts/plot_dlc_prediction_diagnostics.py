"""Plot confidence and manual-label error for selected DLC evaluation trials."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BODY_PARTS = ["back_paw", "front_paw", "tail_S", "tail_M", "tail_E", "body_center"]
COLORS = {
    "back_paw": "#7c3aed",
    "front_paw": "#0891b2",
    "tail_S": "#65a30d",
    "tail_M": "#ea580c",
    "tail_E": "#dc2626",
    "body_center": "#2563eb",
}


def first_scorer(frame: pd.DataFrame) -> str:
    return str(frame.columns.get_level_values("scorer").unique()[0])


def prediction_and_labels(project: Path, evaluation: pd.DataFrame, folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ground_truth_path = project / "labeled-data" / folder / "CollectedData_RBT_CV.h5"
    labels = pd.read_hdf(ground_truth_path, key="df_with_missing")
    predicted = evaluation.loc[[index for index in evaluation.index if index[1] == folder]]
    if predicted.empty:
        raise ValueError(f"No evaluation predictions found for {folder}")
    predicted = predicted.reindex(labels.index)
    if predicted.isna().all().all():
        raise ValueError(f"Evaluation rows do not align with manual labels for {folder}")
    return predicted, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("evaluation_h5", type=Path)
    parser.add_argument("output_stem", type=Path)
    parser.add_argument("trial", nargs=2, help="Two labeled-data folder names to compare")
    args = parser.parse_args()

    evaluation = pd.read_hdf(args.evaluation_h5)
    trials = [(folder, *prediction_and_labels(args.project, evaluation, folder)) for folder in args.trial]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col")
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Six-landmark prediction diagnostics — snapshot-best-180",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.952,
        "Confidence uses DeepLabCut likelihood; error is Euclidean distance from the manual label.",
        ha="center",
        color="#485260",
        fontsize=10,
    )

    for row, (folder, prediction, labels) in enumerate(trials):
        pred_scorer = first_scorer(prediction)
        label_scorer = first_scorer(labels)
        confidence_axis, error_axis = axes[row]
        frame_positions = np.arange(1, len(labels) + 1)
        errors: list[float] = []

        for bodypart in BODY_PARTS:
            pred = prediction[(pred_scorer, "animal", bodypart)]
            truth = labels[(label_scorer, bodypart)]
            confidence_axis.plot(
                frame_positions,
                pred["likelihood"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=COLORS[bodypart],
                label=bodypart,
            )
            valid = truth[["x", "y"]].notna().all(axis=1)
            error = np.hypot(pred.loc[valid, "x"] - truth.loc[valid, "x"], pred.loc[valid, "y"] - truth.loc[valid, "y"])
            errors.extend(error.tolist())
            error_axis.plot(
                frame_positions[valid.to_numpy()],
                error,
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=COLORS[bodypart],
                label=bodypart,
            )

        confidence_axis.axhline(0.60, color="#64748b", linestyle="--", linewidth=1, label="GUI cutoff (0.60)")
        confidence_axis.set_ylim(0, 1.05)
        confidence_axis.set_ylabel("Likelihood")
        confidence_axis.set_title(f"{folder}\nConfidence", fontsize=11, fontweight="bold")
        confidence_axis.grid(axis="y", alpha=0.2)

        mean_error = float(np.mean(errors)) if errors else float("nan")
        median_error = float(np.median(errors)) if errors else float("nan")
        error_axis.set_ylabel("Error (pixels)")
        error_axis.set_title(
            f"{folder}\nPrediction vs. manual label — mean {mean_error:.2f}px, median {median_error:.2f}px",
            fontsize=11,
            fontweight="bold",
        )
        error_axis.grid(axis="y", alpha=0.2)

    for axis in axes[-1]:
        axis.set_xlabel("Labeled frame order")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.01))
    figure.tight_layout(rect=(0.02, 0.08, 0.98, 0.93))

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(args.output_stem.with_suffix(".svg"))
    print(args.output_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
