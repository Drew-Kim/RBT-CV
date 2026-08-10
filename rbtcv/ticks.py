from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv
import json

import numpy as np

from .dataset import ROOT, TrialVideo
from .scoring import BEAM_LENGTH_CM, BEAM_TICK_MARKS_CM, FALL_DISTANCE_STEP_CM


TICK_CALIBRATIONS_FILE = ROOT / "outputs" / "tick_calibrations.csv"


@dataclass(frozen=True)
class BeamTick:
    distance_cm: int
    x: int
    y: int


@dataclass(frozen=True)
class BeamCalibration:
    key: str
    dataset: str
    day: str
    cage: str
    subject: str
    source_video: str
    source_trial: int
    frame_numbers: tuple[int, ...]
    ticks: tuple[BeamTick, ...]
    confirmed_at: str


@dataclass(frozen=True)
class TickDetectionResult:
    ticks: tuple[BeamTick, ...]
    frame_numbers: tuple[int, ...]
    message: str


def calibration_key(video: TrialVideo) -> str:
    return f"{video.dataset}|{video.day}|{video.cage_number}_{video.rat_id}"


def calibration_from_detection(
    video: TrialVideo,
    detection: TickDetectionResult,
    confirmed_at: str,
) -> BeamCalibration:
    return BeamCalibration(
        key=calibration_key(video),
        dataset=video.dataset,
        day=video.day,
        cage=video.cage_number,
        subject=video.rat_id,
        source_video=video.relative_path,
        source_trial=video.trial,
        frame_numbers=detection.frame_numbers,
        ticks=detection.ticks,
        confirmed_at=confirmed_at,
    )


def calibration_with_replaced_tick(
    calibration: BeamCalibration,
    distance_cm: int,
    x: int,
    y: int,
) -> BeamCalibration:
    replacement = BeamTick(distance_cm, x, y)
    ticks = [replacement if tick.distance_cm == distance_cm else tick for tick in calibration.ticks]
    if all(tick.distance_cm != distance_cm for tick in calibration.ticks):
        ticks.append(replacement)
    return replace(calibration, ticks=tuple(sorted(ticks, key=lambda tick: tick.distance_cm)))


def estimate_distance_from_point(calibration: BeamCalibration, x: int, y: int) -> int:
    ticks = sorted(calibration.ticks, key=lambda item: item.distance_cm)

    # Need at least two points to estimate a position along the beam.
    if len(ticks) < 2:
        return 0

    point = np.array([float(x), float(y)], dtype=np.float64)
    best_distance_cm = 0.0
    best_error = float("inf")

    # Try every neighboring pair of ticks and keep the closest beam segment.
    for start, end in zip(ticks, ticks[1:]):
        a = np.array([float(start.x), float(start.y)], dtype=np.float64)
        b = np.array([float(end.x), float(end.y)], dtype=np.float64)
        segment = b - a
        segment_length_sq = float(np.dot(segment, segment))

        # Skip bad duplicate tick points.
        if segment_length_sq <= 0:
            continue

        t = float(np.dot(point - a, segment) / segment_length_sq)
        t = max(0.0, min(1.0, t))
        projection = a + (segment * t)
        error = float(np.sum((point - projection) ** 2))

        if error < best_error:
            best_error = error
            best_distance_cm = start.distance_cm + ((end.distance_cm - start.distance_cm) * t)

    # Fall distances are scored to the nearest 5 cm.
    rounded = int(round(best_distance_cm / FALL_DISTANCE_STEP_CM) * FALL_DISTANCE_STEP_CM)
    return max(0, min(BEAM_LENGTH_CM, rounded))


def interval_midpoint_distance_from_point(calibration: BeamCalibration, x: int, y: int) -> int:
    """Return the midpoint of the beam-tick interval containing a tracked point."""
    ticks = sorted(calibration.ticks, key=lambda item: item.distance_cm)
    if len(ticks) < 2:
        return 0

    point = np.array([float(x), float(y)], dtype=np.float64)
    best: tuple[BeamTick, BeamTick, float] | None = None
    best_error = float("inf")
    for start, end in zip(ticks, ticks[1:]):
        a = np.array([float(start.x), float(start.y)], dtype=np.float64)
        b = np.array([float(end.x), float(end.y)], dtype=np.float64)
        segment = b - a
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 0:
            continue
        fraction = max(0.0, min(1.0, float(np.dot(point - a, segment) / length_squared)))
        projection = a + (segment * fraction)
        error = float(np.sum((point - projection) ** 2))
        if error < best_error:
            best_error = error
            best = (start, end, fraction)

    if best is None:
        return 0
    start, end, fraction = best
    if fraction <= 0:
        return start.distance_cm
    if fraction >= 1:
        return end.distance_cm
    return (start.distance_cm + end.distance_cm) // 2

def point_for_distance(calibration: BeamCalibration, distance_cm: int) -> BeamTick | None:
    """Return the video point for a distance on the calibrated beam."""
    ticks = sorted(calibration.ticks, key=lambda item: item.distance_cm)
    if not ticks:
        return None

    for tick in ticks:
        if tick.distance_cm == distance_cm:
            return tick

    for start, end in zip(ticks, ticks[1:]):
        if not (start.distance_cm <= distance_cm <= end.distance_cm):
            continue

        distance_span = end.distance_cm - start.distance_cm
        if distance_span <= 0:
            continue

        fraction = (distance_cm - start.distance_cm) / distance_span
        x = start.x + ((end.x - start.x) * fraction)
        y = start.y + ((end.y - start.y) * fraction)
        return BeamTick(distance_cm, int(round(x)), int(round(y)))

    if distance_cm <= ticks[0].distance_cm:
        return ticks[0]
    return ticks[-1]



def tick_line_y_at_x(calibration: BeamCalibration, x: float) -> float | None:
    """Return the calibrated tick-center line's y position at image coordinate ``x``.

    The beam can be slightly tilted in the recording, so fall detection should use
    the local line through the tick centers instead of one fixed image y coordinate.
    Points beyond either end use the nearest end tick's y value.
    """
    ticks = sorted(calibration.ticks, key=lambda tick: tick.x)
    if not ticks:
        return None
    if len(ticks) == 1:
        return float(ticks[0].y)

    if x <= ticks[0].x:
        return float(ticks[0].y)
    if x >= ticks[-1].x:
        return float(ticks[-1].y)

    for start, end in zip(ticks, ticks[1:]):
        if not (start.x <= x <= end.x):
            continue

        span = end.x - start.x
        # Duplicate tick x values cannot define a line segment; keep looking for
        # the next usable pair.
        if span <= 0:
            continue

        fraction = (x - start.x) / span
        return float(start.y + ((end.y - start.y) * fraction))

    # This is only reachable with duplicate x values between otherwise valid
    # endpoints. Returning the closest following tick keeps the result stable.
    for tick in ticks:
        if tick.x >= x:
            return float(tick.y)
    return float(ticks[-1].y)

class TickCalibrationStore:
    fieldnames = [
        "key",
        "dataset",
        "day",
        "cage",
        "subject",
        "source_video",
        "source_trial",
        "frame_numbers",
        "ticks_json",
        "confirmed_at",
    ]

    def __init__(self, path: Path = TICK_CALIBRATIONS_FILE) -> None:
        self.path = path

    def load_by_key(self) -> dict[str, BeamCalibration]:
        if not self.path.exists():
            return {}

        calibrations: dict[str, BeamCalibration] = {}
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                calibration = self._from_row(row)
                if calibration is not None:
                    calibrations[calibration.key] = calibration
        return calibrations

    def save(self, calibration: BeamCalibration) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        calibrations = self.load_by_key()
        calibrations[calibration.key] = calibration

        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for key in sorted(calibrations):
                writer.writerow(self._to_row(calibrations[key]))

    def _to_row(self, calibration: BeamCalibration) -> dict[str, str]:
        ticks = [{"distance_cm": tick.distance_cm, "x": tick.x, "y": tick.y} for tick in calibration.ticks]
        return {
            "key": calibration.key,
            "dataset": calibration.dataset,
            "day": calibration.day,
            "cage": calibration.cage,
            "subject": calibration.subject,
            "source_video": calibration.source_video,
            "source_trial": str(calibration.source_trial),
            "frame_numbers": json.dumps(list(calibration.frame_numbers), separators=(",", ":")),
            "ticks_json": json.dumps(ticks, separators=(",", ":")),
            "confirmed_at": calibration.confirmed_at,
        }

    def _from_row(self, row: dict[str, str]) -> BeamCalibration | None:
        try:
            ticks = tuple(
                BeamTick(int(item["distance_cm"]), int(item["x"]), int(item["y"]))
                for item in json.loads(row.get("ticks_json", "[]"))
            )
            frame_numbers = tuple(int(item) for item in json.loads(row.get("frame_numbers", "[]")))
            return BeamCalibration(
                key=row["key"],
                dataset=row["dataset"],
                day=row["day"],
                cage=row["cage"],
                subject=row["subject"],
                source_video=row.get("source_video", ""),
                source_trial=int(row.get("source_trial", 1)),
                frame_numbers=frame_numbers,
                ticks=ticks,
                confirmed_at=row.get("confirmed_at", ""),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None



DEFAULT_TICK_DLC_OUTPUT_DIR = ROOT / "outputs" / "dlc_tick_predictions"
DEFAULT_TICK_DLC_LIKELIHOOD_CUTOFF = 0.30


class DLCTickDetector:
    """Loads high-confidence tick predictions from the current DLC calibration run."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_TICK_DLC_OUTPUT_DIR,
        likelihood_cutoff: float = DEFAULT_TICK_DLC_LIKELIHOOD_CUTOFF,
    ) -> None:
        self.output_dir = output_dir
        self.likelihood_cutoff = likelihood_cutoff

    def detect_for_video(self, video: TrialVideo) -> TickDetectionResult:
        csv_path = self.find_for_video(video)
        if csv_path is None:
            return TickDetectionResult(
                (),
                (),
                "No current tick-model CSV found. Run Detect ticks with DLC again.",
            )
        return self.detect_from_csv(csv_path)

    def find_for_video(self, video: TrialVideo) -> Path | None:
        if not self.output_dir.exists():
            return None

        matches = list(self.output_dir.glob(f"{video.path.stem}*DLC_*.csv"))
        if not matches:
            return None
        return max(matches, key=lambda path: path.stat().st_mtime)

    def detect_from_csv(self, csv_path: Path) -> TickDetectionResult:
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        except OSError:
            return TickDetectionResult((), (), f"Could not read {csv_path.name}")

        if len(rows) < 4 or len(rows[1]) < 2 or len(rows[2]) < 2:
            return TickDetectionResult((), (), f"{csv_path.name} is not a DLC prediction CSV")

        columns = self._tick_columns(rows[1], rows[2])
        if not columns:
            return TickDetectionResult((), (), f"{csv_path.name} has no tick labels")

        candidates: list[tuple[int, dict[int, tuple[float, float, float]], float]] = []
        for row_number, row in enumerate(rows[3:]):
            try:
                frame_number = int(float(row[0]))
            except (IndexError, ValueError):
                frame_number = row_number

            predictions: dict[int, tuple[float, float, float]] = {}
            for distance_cm, indexes in columns.items():
                try:
                    x = float(row[indexes["x"]])
                    y = float(row[indexes["y"]])
                    likelihood = float(row[indexes["likelihood"]])
                except (IndexError, ValueError):
                    continue
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(likelihood):
                    predictions[distance_cm] = (x, y, likelihood)

            score = self._clear_frame_score(predictions)
            if score is not None:
                candidates.append((frame_number, predictions, score))

        # Keep up to the five clearest candidates. If only one or two frames are
        # unobstructed, they are still safer than blending in a bad prediction.
        selected = sorted(candidates, key=lambda candidate: candidate[2], reverse=True)[:5]
        if not selected:
            return TickDetectionResult(
                (),
                (),
                "No clear non-overlapping calibration frames were found.",
            )
        ticks: list[BeamTick] = []
        for distance_cm in BEAM_TICK_MARKS_CM:
            predictions = [candidate[1][distance_cm] for candidate in selected]
            ticks.append(
                BeamTick(
                    distance_cm,
                    int(round(np.median([prediction[0] for prediction in predictions]))),
                    int(round(np.median([prediction[1] for prediction in predictions]))),
                )
            )

        if not self._has_clear_tick_spacing(ticks):
            return TickDetectionResult(
                (),
                tuple(frame_number for frame_number, _, _ in selected),
                "The selected calibration frames produced overlapping ticks; calibration was not saved.",
            )

        return TickDetectionResult(
            tuple(ticks),
            tuple(frame_number for frame_number, _, _ in selected),
            f"DLC tick model found {len(ticks)}/{len(BEAM_TICK_MARKS_CM)} ticks from {len(selected)} clear non-overlapping calibration frame(s).",
        )

    def _clear_frame_score(self, predictions: dict[int, tuple[float, float, float]]) -> float | None:
        """Score only complete frames whose predicted ticks remain well separated."""
        if set(predictions) != set(BEAM_TICK_MARKS_CM):
            return None
        if any(prediction[2] < self.likelihood_cutoff for prediction in predictions.values()):
            return None

        x_values = [predictions[distance_cm][0] for distance_cm in BEAM_TICK_MARKS_CM]
        if not self._has_clear_tick_spacing_from_x(x_values):
            return None

        likelihood = float(np.mean([predictions[distance_cm][2] for distance_cm in BEAM_TICK_MARKS_CM]))
        direction = 1.0 if x_values[-1] > x_values[0] else -1.0
        gaps = [direction * (end - start) for start, end in zip(x_values, x_values[1:])]
        # Confidence ranks otherwise valid frames; the small spacing term breaks ties
        # in favour of frames where nearby tick marks are easiest to distinguish.
        return likelihood + (0.05 * min(gaps) / float(np.median(gaps)))

    @staticmethod
    def _has_clear_tick_spacing(ticks: list[BeamTick]) -> bool:
        return DLCTickDetector._has_clear_tick_spacing_from_x([tick.x for tick in ticks])

    @staticmethod
    def _has_clear_tick_spacing_from_x(x_values: list[float]) -> bool:
        if len(x_values) < 2 or x_values[-1] == x_values[0]:
            return False
        direction = 1.0 if x_values[-1] > x_values[0] else -1.0
        gaps = [direction * (end - start) for start, end in zip(x_values, x_values[1:])]
        if any(gap <= 0 for gap in gaps):
            return False
        median_gap = float(np.median(gaps))
        # Reject merged neighbours (such as 70/80) but do not alter their locations.
        return median_gap > 0 and min(gaps) >= (0.35 * median_gap)
    def _tick_columns(self, bodyparts: list[str], coords: list[str]) -> dict[int, dict[str, int]]:
        columns: dict[int, dict[str, int]] = {}
        for index, coordinate in enumerate(coords):
            if coordinate.strip().lower() not in {"x", "y", "likelihood"} or index >= len(bodyparts):
                continue
            name = bodyparts[index].strip().lower()
            if not name.startswith("tick_"):
                continue
            try:
                distance_cm = int(name.removeprefix("tick_"))
            except ValueError:
                continue
            if distance_cm not in BEAM_TICK_MARKS_CM:
                continue
            columns.setdefault(distance_cm, {})[coordinate.strip().lower()] = index
        return {
            distance_cm: indexes
            for distance_cm, indexes in columns.items()
            if {"x", "y", "likelihood"}.issubset(indexes)
        }

class BeamTickDetector:
    """Legacy GUI compatibility wrapper.

    Automatic tick calibration is now performed by ``DLCTickDetector`` after the
    trained DLC tick model writes a CSV. The older frame-only detector deliberately
    returns a clear message instead of inventing unsafe calibration points.
    """

    def detect_from_video(self, video_path: Path) -> TickDetectionResult:
        if not video_path.exists():
            return TickDetectionResult((), (), "Video file was not found.")
        return TickDetectionResult(
            (),
            (),
            "Run Detect Trial's Ticks with the trained DLC tick model before confirming intervals.",
        )
