from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import re

import cv2

from .dataset import ROOT, TrialVideo
from .research_angle import TailAngleMeasurement


DEFAULT_DLC_PREDICTIONS_DIR = ROOT / "outputs" / "dlc_predictions"
DEFAULT_DLC_PROJECTS_DIR = ROOT / "models" / "dlc_tracking"

# The legacy model labels visible paws and tail_end; the current model uses
# back_paw/front_paw plus tail_S, tail_M, and tail_E. Keep both CSV schemas
# readable so historical outputs stay usable.
# Unrelated labels such as head and nose are ignored by the GUI.
TRACKING_BODYPARTS = (
    "visible_back_paw",
    "visible_front_paw",
    "back_paw",
    "front_paw",
    "tail_S",
    "tail_M",
    "tail_E",
    "tail_end",
    "body_center",
)
TAIL_START_ALIASES = {"tails", "tailstart", "tailbase"}
TAIL_MIDDLE_ALIASES = {"tailm", "tailmiddle", "tailmid"}
TAIL_END_ALIASES = {"taile", "tailend", "tailtip"}
BODY_CENTER_ALIASES = {"bodycenter", "bodycentre", "mousebody"}
BACK_PAW_ALIASES = {
    "visiblebackpaw",
    "backpaw",
    "hindpaw",
    "rearpaw",
    "visiblehindpaw",
    "visiblerearpaw",
    "hindlimbpaw",
    "backlimbpaw",
}
FRONT_PAW_ALIASES = {
    "visiblefrontpaw",
    "frontpaw",
    "forepaw",
    "visibleforepaw",
    "forelimbpaw",
    "frontlimbpaw",
}

# A lower threshold keeps the live overlay responsive; scoring remains conservative.
DISPLAY_LIKELIHOOD_CUTOFF = 0.20
SCORING_LIKELIHOOD_CUTOFF = 0.60
DEFAULT_LIKELIHOOD_CUTOFF = SCORING_LIKELIHOOD_CUTOFF


@dataclass(frozen=True)
class DLCPoint:
    name: str
    kind: str
    x: float
    y: float
    likelihood: float


@dataclass(frozen=True)
class DLCFramePrediction:
    frame: int
    points: tuple[DLCPoint, ...]


    def _first_alias(self, aliases: set[str]) -> DLCPoint | None:
        return next(
            (point for point in self.points if normalize_bodypart_name(point.name) in aliases),
            None,
        )

    @property

    def visible_back_paw(self) -> DLCPoint | None:
        return self._first_alias(BACK_PAW_ALIASES)

    @property

    def visible_front_paw(self) -> DLCPoint | None:
        return self._first_alias(FRONT_PAW_ALIASES)

    @property

    def body_center(self) -> DLCPoint | None:
        return self._first_alias(BODY_CENTER_ALIASES)

    @property

    def tail_end(self) -> DLCPoint | None:
        return self._first_alias(TAIL_END_ALIASES)

    @property
    def tail_start(self) -> DLCPoint | None:
        return self._first_alias(TAIL_START_ALIASES)

    @property
    def tail_middle(self) -> DLCPoint | None:
        return self._first_alias(TAIL_MIDDLE_ALIASES)

    @property

    def paw_count(self) -> int:
        return sum(point.kind == "paw" for point in self.points)

    @property

    def tail_count(self) -> int:
        return sum(1 for point in self.points if point.kind == "tail")

    @property

    def body_points(self) -> tuple[DLCPoint, ...]:
        return tuple(point for point in self.points if point.kind == "body")


@dataclass(frozen=True)
class DLCTracking:
    csv_path: Path
    frames: dict[int, DLCFramePrediction]
    likelihood_cutoff: float = DEFAULT_LIKELIHOOD_CUTOFF

    def points_for_frame(self, frame_number: int) -> DLCFramePrediction | None:
        return self.frames.get(frame_number)
    def filtered(self, likelihood_cutoff: float) -> DLCTracking:
        """Create a stricter view without reading the same CSV again."""
        if likelihood_cutoff < self.likelihood_cutoff:
            raise ValueError("A filtered tracking view cannot restore discarded points.")
        frames = {
            frame_number: DLCFramePrediction(
                frame_number,
                tuple(point for point in prediction.points if point.likelihood >= likelihood_cutoff),
            )
            for frame_number, prediction in self.frames.items()
        }
        return DLCTracking(self.csv_path, frames, likelihood_cutoff)


class DLCPredictionStore:

    def __init__(
        self,
        predictions_dir: Path = DEFAULT_DLC_PREDICTIONS_DIR,
        likelihood_cutoff: float = DEFAULT_LIKELIHOOD_CUTOFF,
    ) -> None:
        self.predictions_dir = predictions_dir
        self.likelihood_cutoff = likelihood_cutoff

    def load(self, csv_path: Path) -> DLCTracking:
        """Read one DeepLabCut CSV file into simple frame-by-frame points."""
        rows = self._read_rows(csv_path)
        bodyparts, coords = self._header_rows(rows, csv_path)
        columns = self._tracked_columns(bodyparts, coords)

        # If the CSV only has labels like body/head/nose, this GUI should not use it.
        if not columns:
            raise ValueError("No supported paw, tail, or body-center bodyparts found in the DLC CSV.")

        frames: dict[int, DLCFramePrediction] = {}
        for row_number, row in enumerate(rows[3:]):
            frame_number = self._frame_number(row, row_number)
            points: list[DLCPoint] = []
            for name, indexes in columns.items():
                point = self._point_from_row(row, name, indexes)

                # DeepLabCut gives every point a likelihood. Keep only reliable points.
                if point is not None and point.likelihood >= self.likelihood_cutoff:
                    points.append(point)
            frames[frame_number] = DLCFramePrediction(frame_number, tuple(points))

        return DLCTracking(csv_path=csv_path, frames=frames, likelihood_cutoff=self.likelihood_cutoff)

    def find_for_video(self, video: TrialVideo) -> Path | None:
        """Find the DLC CSV that belongs to a video.

        Search the app output directory first, then the source video directory,
        then every DLC project's ``videos`` directory. Within a directory, use
        the newest matching CSV so re-analysis naturally takes precedence.
        """
        project_video_folders = sorted(DEFAULT_DLC_PROJECTS_DIR.glob("*/videos"))
        for folder in (self.predictions_dir, video.path.parent, *project_video_folders):
            # Some projects will not have prediction files yet.
            if not folder.exists():
                continue

            matches = list(folder.glob(f"{video.path.stem}*.csv"))
            if matches:
                return max(matches, key=lambda path: path.stat().st_mtime)
        return None

    def _read_rows(self, csv_path: Path) -> list[list[str]]:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))

        # A normal DLC CSV starts with 3 header rows, then one row per frame.
        if len(rows) < 4:
            raise ValueError("DLC CSV is missing the expected three header rows.")
        return rows

    def _header_rows(self, rows: list[list[str]], csv_path: Path) -> tuple[list[str], list[str]]:
        bodyparts = rows[1]
        coords = rows[2]

        first_coord_cell = ""
        if coords:
            first_coord_cell = coords[0].strip().lower()

        if first_coord_cell != "coords":
            raise ValueError(f"{csv_path.name} does not look like a DeepLabCut prediction CSV.")

        return bodyparts, coords

    def _tracked_columns(self, bodyparts: list[str], coords: list[str]) -> dict[str, dict[str, int]]:
        """Map each tracked bodypart to its x/y/likelihood column numbers."""
        columns: dict[str, dict[str, int]] = {}
        for index, coord in enumerate(coords):
            coord_name = coord.strip().lower()

            # DLC repeats x, y, likelihood for every bodypart.
            if coord_name not in {"x", "y", "likelihood"}:
                continue

            if index >= len(bodyparts):
                continue

            name = bodyparts[index].strip()
            kind = tracked_bodypart_kind(name)

            # Ignore bodyparts outside the simplified tracking plan.
            if kind is None:
                continue

            columns.setdefault(name, {})[coord_name] = index

        complete_columns: dict[str, dict[str, int]] = {}
        for name, indexes in columns.items():
            has_x = "x" in indexes
            has_y = "y" in indexes
            has_likelihood = "likelihood" in indexes
            if has_x and has_y and has_likelihood:
                complete_columns[name] = indexes

        return complete_columns

    def _frame_number(self, row: list[str], fallback: int) -> int:
        # DLC usually stores something like "labeled-data/.../img000123.png".
        # Taking the first number from that text gives the video frame number.
        if row:
            match = re.search(r"\d+", row[0])
            if match:
                return int(match.group())
        return fallback

    def _point_from_row(self, row: list[str], name: str, indexes: dict[str, int]) -> DLCPoint | None:
        try:
            x = float(row[indexes["x"]])
            y = float(row[indexes["y"]])
            likelihood = float(row[indexes["likelihood"]])
        except (IndexError, ValueError):
            return None

        if math.isnan(x) or math.isnan(y) or math.isnan(likelihood):
            return None

        kind = tracked_bodypart_kind(name)
        if kind is None:
            return None
        return DLCPoint(name=name, kind=kind, x=x, y=y, likelihood=likelihood)


def normalize_bodypart_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def tracked_bodypart_kind(name: str) -> str | None:
    normalized = normalize_bodypart_name(name)

    # Accept the named tail landmarks, but not a generic whole-tail label.
    if normalized in TAIL_START_ALIASES | TAIL_MIDDLE_ALIASES | TAIL_END_ALIASES:
        return "tail"

    if normalized in BODY_CENTER_ALIASES:
        return "body"
    # Any bodypart with "paw" in the name is treated as one visible paw.
    if "paw" in normalized:
        return "paw"

    return None


def draw_tracking_overlay(
    frame_bgr,
    prediction: DLCFramePrediction | None,
    tail_angle: TailAngleMeasurement | None = None,
):
    overlay = frame_bgr.copy()
    if prediction is None:
        cv2.putText(
            overlay,
            "no DLC tracking for frame",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    # Draw the tail skeleton first so the landmark dots remain visible on top.
    for start, end in (
        (prediction.tail_start, prediction.tail_middle),
        (prediction.tail_middle, prediction.tail_end),
    ):
        if start is not None and end is not None:
            cv2.line(
                overlay,
                (int(round(start.x)), int(round(start.y))),
                (int(round(end.x)), int(round(end.y))),
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    if tail_angle is not None:
        _draw_dotted_line(
            overlay,
            (tail_angle.tail_start_x, tail_angle.tail_start_y),
            (tail_angle.tail_end_x, tail_angle.tail_end_y),
            (255, 255, 0),
        )
        tail_axis = _line_to_frame_bounds(
            (tail_angle.tail_start_x, tail_angle.tail_start_y),
            (tail_angle.tail_end_x, tail_angle.tail_end_y),
            overlay.shape[1],
            overlay.shape[0],
        )
        if tail_axis is not None:
            _draw_dashed_line(overlay, tail_axis[0], tail_axis[1], (0, 255, 0))
        label_origin = (
            int(round(tail_angle.tail_start_x + 10)),
            max(18, int(round(tail_angle.tail_start_y + 22))),
        )
        cv2.putText(
            overlay,
            f"Tail angle: {tail_angle.angle_degrees:+.1f} deg",
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for point in prediction.points:
        center = (int(round(point.x)), int(round(point.y)))

        # Magenta marks paws, yellow marks the tail endpoints, orange marks tail_M,
        # and cyan marks body center.
        if point.kind == "paw":
            color = (255, 0, 255)
        elif point.kind == "tail":
            color = (0, 140, 255) if normalize_bodypart_name(point.name) in TAIL_MIDDLE_ALIASES else (0, 220, 255)
        else:
            color = (255, 220, 0)

        cv2.circle(overlay, center, 3, color, -1)
        cv2.putText(
            overlay,
            f"{point.name} {point.likelihood:.2f}",
            (center[0] + 8, max(16, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return overlay


def _draw_dotted_line(frame_bgr, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int]) -> None:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = math.hypot(delta_x, delta_y)
    if length <= 1e-9:
        return
    for distance in range(0, int(length) + 1, 7):
        fraction = min(1.0, distance / length)
        cv2.circle(
            frame_bgr,
            (int(round(start[0] + (fraction * delta_x))), int(round(start[1] + (fraction * delta_y)))),
            1,
            color,
            -1,
            cv2.LINE_AA,
        )


def _draw_dashed_line(frame_bgr, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int]) -> None:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = math.hypot(delta_x, delta_y)
    if length <= 1e-9:
        return
    dash_length_px = 7.0
    gap_length_px = 5.0
    distance = 0.0
    while distance < length:
        dash_end = min(length, distance + dash_length_px)
        start_fraction = distance / length
        end_fraction = dash_end / length
        cv2.line(
            frame_bgr,
            (int(round(start[0] + (start_fraction * delta_x))), int(round(start[1] + (start_fraction * delta_y)))),
            (int(round(start[0] + (end_fraction * delta_x))), int(round(start[1] + (end_fraction * delta_y)))),
            color,
            1,
            cv2.LINE_AA,
        )
        distance = dash_end + gap_length_px


def _line_to_frame_bounds(
    start: tuple[float, float],
    end: tuple[float, float],
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Extend a non-zero line through two points to the visible image bounds."""
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    if math.hypot(delta_x, delta_y) <= 1e-9 or width <= 0 or height <= 0:
        return None

    intersections: list[tuple[float, float, float]] = []
    if abs(delta_x) > 1e-9:
        for edge_x in (0.0, float(width - 1)):
            fraction = (edge_x - start[0]) / delta_x
            edge_y = start[1] + (fraction * delta_y)
            if 0.0 <= edge_y <= height - 1:
                intersections.append((fraction, edge_x, edge_y))
    if abs(delta_y) > 1e-9:
        for edge_y in (0.0, float(height - 1)):
            fraction = (edge_y - start[1]) / delta_y
            edge_x = start[0] + (fraction * delta_x)
            if 0.0 <= edge_x <= width - 1:
                intersections.append((fraction, edge_x, edge_y))

    if len(intersections) < 2:
        return None
    first = min(intersections, key=lambda item: item[0])
    last = max(intersections, key=lambda item: item[0])
    return (first[1], first[2]), (last[1], last[2])
