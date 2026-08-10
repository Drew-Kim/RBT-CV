from __future__ import annotations

import argparse
from pathlib import Path

LABELS = ("visible_back_paw", "visible_front_paw", "tail_end", "body_center")
NAMES = {"visible_back_paw": "Visible back paw", "visible_front_paw": "Visible front paw", "tail_end": "Tail end", "body_center": "Body center"}


def plot(csv_path: Path, pcutoff: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    labels = [label for label in LABELS if label in set(data.columns.get_level_values(1))]
    if not labels:
        raise ValueError(f"No mouse-tracking labels in {csv_path.name}")
    frame = pd.to_numeric(data.index, errors="coerce")
    values = {"x": {}, "y": {}, "likelihood": {}}
    for label in labels:
        bodypart = data.xs(label, axis=1, level=1)
        for coordinate in values:
            values[coordinate][label] = pd.to_numeric(bodypart.xs(coordinate, axis=1, level=1).iloc[:, 0], errors="coerce")
    title = csv_path.stem.split("DLC_")[0].rstrip("_- ")
    report = csv_path.parent / "tracking-reports" / title
    report.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(12, 6))
    for label in labels:
        axis.plot(frame, values["likelihood"][label], label=NAMES[label], linewidth=1)
    axis.axhline(pcutoff, color="black", linestyle="--", linewidth=1, label=f"Confidence cutoff ({pcutoff:.2f})")
    axis.set(title=f"Mouse-tracking confidence by video frame - {title}", xlabel="Video frame index (frame number in the analyzed trial)", ylabel="DeepLabCut landmark confidence (likelihood, 0 to 1)", ylim=(0, 1.05))
    axis.legend(loc="center left", bbox_to_anchor=(1, .5), title="Mouse landmark")
    fig.tight_layout(); fig.savefig(report / "tracking_confidence_over_time.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label in labels:
        axes[0].plot(frame, values["x"][label], label=NAMES[label], linewidth=1)
        axes[1].plot(frame, values["y"][label], label=NAMES[label], linewidth=1)
    axes[0].set(title=f"Mouse landmark coordinates by video frame - {title}", ylabel="Horizontal image coordinate, x (pixels; larger = farther right)")
    axes[1].set(xlabel="Video frame index (frame number in the analyzed trial)", ylabel="Vertical image coordinate, y (pixels; larger = lower in image)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1, .5), title="Mouse landmark")
    fig.tight_layout(); fig.savefig(report / "tracking_positions_over_time.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    means = [values["likelihood"][label].mean() for label in labels]
    axis.bar([NAMES[label] for label in labels], means, color="#2f7f9d")
    axis.axhline(pcutoff, color="black", linestyle="--", linewidth=1, label=f"Confidence cutoff ({pcutoff:.2f})")
    axis.set(title=f"Average mouse-tracking confidence by landmark - {title}", xlabel="Tracked mouse landmark", ylabel="Mean DeepLabCut confidence across all analyzed frames (0 to 1)", ylim=(0, 1.05))
    axis.tick_params(axis="x", rotation=15); axis.legend(); fig.tight_layout(); fig.savefig(report / "tracking_average_confidence.png", dpi=180); plt.close(fig)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--pcutoff", type=float, default=.6)
    args = parser.parse_args()
    for csv_path in args.csv:
        print(plot(csv_path, args.pcutoff))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())