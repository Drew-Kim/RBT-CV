from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import re

import cv2

from .dataset import ROOT, TrialVideo


DEFAULT_DLC_PREDICTIONS_DIR = ROOT / "outputs" / "dlc_predictions"
DEFAULT_DLC_PROJECTS_DIR = ROOT / "models" / "dlc_tracking"

# The mouse model labels the visible paws, tail end, and body center.
# Unrelated labels such as head and nose are ignored by the GUI.
TRACKING_BODYPARTS = ("visible_back_paw", "visible_front_paw", "tail_end", "body_center")
TAIL_END_ALIASES = {"tailend", "tailtip"}
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

    def body_center(self) -> DLCPoint | None:
        return self._first_alias(BODY_CENTER_ALIASES)

    @property

    def tail_end(self) -> DLCPoint | None:
        return self._first_alias(TAIL_END_ALIASES)

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
            raise ValueError("No paw, tail_end, or body_center bodyparts found in the DLC CSV.")

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

    # Accept tail_end and common tail-tip naming, but do not accept whole-tail labels.
    if normalized in TAIL_END_ALIASES:
        return "tail"

    if normalized in BODY_CENTER_ALIASES:
        return "body"
    # Any bodypart with "paw" in the name is treated as one visible paw.
    if "paw" in normalized:
        return "paw"

    return None


def draw_tracking_overlay(frame_bgr, prediction: DLCFramePrediction | None):
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

    for point in prediction.points:
        center = (int(round(point.x)), int(round(point.y)))

        # Magenta marks paws, yellow marks the tail end, and cyan marks body center.
        if point.kind == "paw":
            color = (255, 0, 255)
        elif point.kind == "tail":
            color = (0, 220, 255)
        else:
            color = (255, 220, 0)

        cv2.circle(overlay, center, 5, color, -1)
        cv2.circle(overlay, center, 8, (255, 255, 255), 1)
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

@dataclass(frozen=True)
class MouseDetection:
    """Compatibility result used by the legacy GUI overlay."""

    found: bool
    bbox: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    limb_candidates: tuple[tuple[int, int], ...] = ()
    message: str = ""


class MouseLimbDetector:
    """Lightweight fallback overlay detector for the legacy review interface.

    DLC CSV tracking remains the authoritative scorer. This detector only keeps the
    older live bounding-box overlay functional when no DLC CSV is loaded.
    """

    def detect(self, frame) -> MouseDetection:
        if frame is None or frame.size == 0:
            return MouseDetection(False, message="No video frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = [contour for contour in contours if cv2.contourArea(contour) >= 250]
        if not candidates:
            return MouseDetection(False, message="Mouse not detected")
        contour = max(candidates, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        area_ratio = min(1.0, cv2.contourArea(contour) / max(1.0, frame.shape[0] * frame.shape[1] * 0.08))
        return MouseDetection(True, (x, y, width, height), area_ratio, (), "")


def draw_detection_overlay(frame, detection: MouseDetection):
    output = frame.copy()
    if detection.found and detection.bbox is not None:
        x, y, width, height = detection.bbox
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 220, 255), 2)
        cv2.putText(output, f"mouse {detection.confidence:.2f}", (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    return output
