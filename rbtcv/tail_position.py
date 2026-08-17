"""Tail-height recording and normalized daily SVG plots for RBT trials."""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .dataset import ROOT, TrialVideo
from .detection import DLCTracking
from .ticks import BeamCalibration, tick_line_y_at_x


OUTPUT_ROOT = ROOT / "outputs"
RAW_FILENAME = "tail_position_measurements.csv"
GRID = tuple(range(101))
COLORS = ("#1769aa", "#d95f02", "#2e7d32", "#7b1fa2", "#c62828", "#00838f")
TailRecord = tuple[TrialVideo, DLCTracking, BeamCalibration, int, int]


class TailPositionStore:
    """Keep one normalized tail trace per trial and produce daily SVG plots."""

    fieldnames = (
        "relative_video",
        "dataset",
        "day",
        "group",
        "subject",
        "trial",
        "frame",
        "progress_percent",
        "tail_y_px",
        "fall_boundary_y_px",
        "tail_offset_px",
    )

    def __init__(self, output_root: Path = OUTPUT_ROOT) -> None:
        self.output_root = output_root

    def result_dir(self, dataset: str) -> Path:
        return self.output_root / f"{dataset} Results"

    def record_trial(
        self,
        video: TrialVideo,
        tracking: DLCTracking,
        calibration: BeamCalibration,
        start_frame: int,
        end_frame: int,
        *,
        refresh_plot: bool = True,
    ) -> Path | None:
        """Upsert a single trace, retaining the historic single-trial API."""
        if end_frame <= start_frame:
            return None

        points = self._measure(video, tracking, calibration, start_frame, end_frame)
        csv_path = self.result_dir(video.dataset) / RAW_FILENAME
        rows = [
            row
            for row in self._read(csv_path)
            if row["relative_video"] != video.relative_path
        ]
        rows.extend(points)
        self._write(csv_path, rows)

        if not points:
            return None
        return self._write_day_plot(video.dataset, video.day, rows) if refresh_plot else csv_path

    def record_trials(
        self,
        records: Iterable[TailRecord],
        *,
        refresh_plots: bool,
    ) -> dict[tuple[str, str], Path | None]:
        """Upsert every available trace in one CSV write per dataset.

        Batch analysis can process dozens of trials. This avoids repeatedly
        reopening and rewriting the same tail-measurement CSV after each video.
        """
        by_dataset: dict[str, list[TailRecord]] = defaultdict(list)
        for record in records:
            by_dataset[record[0].dataset].append(record)

        plots: dict[tuple[str, str], Path | None] = {}
        for dataset, dataset_records in by_dataset.items():
            csv_path = self.result_dir(dataset) / RAW_FILENAME
            replacing = {video.relative_path for video, *_rest in dataset_records}
            rows = [
                row
                for row in self._read(csv_path)
                if row["relative_video"] not in replacing
            ]
            days = set()
            for video, tracking, calibration, start_frame, end_frame in dataset_records:
                days.add(video.day)
                if end_frame > start_frame:
                    rows.extend(
                        self._measure(
                            video,
                            tracking,
                            calibration,
                            start_frame,
                            end_frame,
                        )
                    )
            self._write(csv_path, rows)

            if refresh_plots:
                for day in days:
                    plots[(dataset, day)] = self._write_day_plot(dataset, day, rows)
        return plots

    def refresh_day_plot(self, dataset: str, day: str) -> Path | None:
        rows = self._read(self.result_dir(dataset) / RAW_FILENAME)
        return self._write_day_plot(dataset, day, rows)

    def _measure(
        self,
        video: TrialVideo,
        tracking: DLCTracking,
        calibration: BeamCalibration,
        start_frame: int,
        end_frame: int,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        span = end_frame - start_frame
        for frame, prediction in sorted(tracking.frames.items()):
            if frame < start_frame or frame > end_frame:
                continue
            tail = prediction.tail_end
            if tail is None:
                continue
            boundary_y = tick_line_y_at_x(calibration, tail.x)
            if boundary_y is None:
                continue
            rows.append(
                {
                    "relative_video": video.relative_path,
                    "dataset": video.dataset,
                    "day": video.day,
                    "group": video.group,
                    "subject": video.subject,
                    "trial": str(video.trial),
                    "frame": str(frame),
                    "progress_percent": f"{100 * (frame - start_frame) / span:.5f}",
                    "tail_y_px": f"{tail.y:.5f}",
                    "fall_boundary_y_px": f"{boundary_y:.5f}",
                    # Positive values are above the calibrated fall boundary.
                    "tail_offset_px": f"{boundary_y - tail.y:.5f}",
                }
            )
        return rows

    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _write(self, path: Path, rows: Iterable[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            with temporary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(
                    sorted(
                        rows,
                        key=lambda row: (
                            row["day"],
                            row["group"],
                            row["subject"],
                            int(row["trial"]),
                            int(row["frame"]),
                        ),
                    )
                )
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _write_day_plot(
        self,
        dataset: str,
        day: str,
        rows: Iterable[dict[str, str]],
    ) -> Path | None:
        path = self.result_dir(dataset) / f"tail_position_{day}.svg"
        traces = self._animal_traces(
            row for row in rows if row["dataset"] == dataset and row["day"] == day
        )
        if not traces:
            if path.exists():
                path.unlink()
            return None

        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            temporary_path.write_text(self._svg(dataset, day, traces), encoding="utf-8")
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return path

    @staticmethod
    def _animal_traces(
        rows: Iterable[dict[str, str]],
    ) -> list[tuple[tuple[str, str], int, list[float | None]]]:
        trials: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            trials[(row["group"], row["subject"])][row["trial"]].append(
                (float(row["progress_percent"]), float(row["tail_offset_px"]))
            )

        traces = []
        for key, by_trial in sorted(trials.items()):
            curves = [_resample(points) for points in by_trial.values()]
            average: list[float | None] = []
            for index in GRID:
                values = [
                    curve[index] for curve in curves if curve[index] is not None
                ]
                average.append(sum(values) / len(values) if values else None)
            traces.append((key, len(by_trial), average))
        return traces

    @staticmethod
    def _svg(
        dataset: str,
        day: str,
        traces: list[tuple[tuple[str, str], int, list[float | None]]],
    ) -> str:
        width, height = 1200, 700
        left, right, top, bottom = 100, 330, 65, 105
        graph_width, graph_height = width - left - right, height - top - bottom
        values = [
            value
            for _key, _count, curve in traces
            for value in curve
            if value is not None
        ]
        low, high = min(values + [0.0, -20.0]), max(values + [0.0, 20.0])
        padding = max(5.0, (high - low) * 0.08)
        low, high = low - padding, high + padding

        def x(value: float) -> float:
            return left + graph_width * value / 100

        def y(value: float) -> float:
            return top + graph_height * (high - value) / (high - low)

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{left}" y="30" font-family="Arial" font-size="22" font-weight="bold">Tail position relative to fall boundary - {html.escape(dataset)} {html.escape(day)}</text>',
            f'<text x="{left}" y="52" font-family="Arial" font-size="13" fill="#444">One line per animal: average of available normalized T1-T3 traces. Positive values are above the fall boundary.</text>',
        ]
        for percent in range(0, 101, 20):
            px = x(percent)
            parts.extend(
                (
                    f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + graph_height}" stroke="#e6e6e6"/>',
                    f'<text x="{px:.1f}" y="{top + graph_height + 25}" text-anchor="middle" font-family="Arial" font-size="12">{percent}%</text>',
                )
            )
        for index in range(6):
            value = low + (high - low) * index / 5
            py = y(value)
            parts.extend(
                (
                    f'<line x1="{left}" y1="{py:.1f}" x2="{left + graph_width}" y2="{py:.1f}" stroke="#e6e6e6"/>',
                    f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.0f}</text>',
                )
            )

        zero = y(0)
        parts.extend(
            (
                f'<line x1="{left}" y1="{zero:.1f}" x2="{left + graph_width}" y2="{zero:.1f}" stroke="#d32f2f" stroke-width="2" stroke-dasharray="7 5"/>',
                f'<text x="{left + graph_width + 8}" y="{zero + 4:.1f}" font-family="Arial" font-size="12" fill="#d32f2f">fall boundary (0)</text>',
            )
        )
        for index, ((group, subject), trial_count, curve) in enumerate(traces):
            color = COLORS[index % len(COLORS)]
            segments, segment = [], []
            for progress, value in enumerate(curve):
                if value is None:
                    if segment:
                        segments.append(segment)
                        segment = []
                    continue
                segment.append(f"{x(progress):.1f},{y(value):.1f}")
            if segment:
                segments.append(segment)
            for points in segments:
                parts.append(
                    f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
                )
            label_y = top + 25 + index * 24
            parts.extend(
                (
                    f'<line x1="{left + graph_width + 35}" y1="{label_y - 5}" x2="{left + graph_width + 57}" y2="{label_y - 5}" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{left + graph_width + 65}" y="{label_y}" font-family="Arial" font-size="13">{html.escape(group)}_{html.escape(subject)} ({trial_count}/3 trials)</text>',
                )
            )
        parts.extend(
            (
                f'<text x="{left + graph_width / 2:.1f}" y="{height - 35}" text-anchor="middle" font-family="Arial" font-size="15">Normalized trial progress (back paw: 0 cm start to terminal event)</text>',
                f'<text x="25" y="{top + graph_height / 2:.1f}" text-anchor="middle" font-family="Arial" font-size="15" transform="rotate(-90 25 {top + graph_height / 2:.1f})">Calibrated fall-boundary Y - tail Y (pixels)</text>',
                "</svg>",
            )
        )
        return "\n".join(parts)


def _resample(points: list[tuple[float, float]]) -> list[float | None]:
    points = sorted(points)
    values: list[float | None] = []
    for target in GRID:
        if not points or target < points[0][0] or target > points[-1][0]:
            values.append(None)
            continue
        left = next((point for point in reversed(points) if point[0] <= target), points[0])
        right = next((point for point in points if point[0] >= target), points[-1])
        if left[0] == right[0]:
            values.append(left[1])
            continue
        fraction = (target - left[0]) / (right[0] - left[0])
        values.append(left[1] + fraction * (right[1] - left[1]))
    return values
