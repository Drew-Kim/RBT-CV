"""Pure frame-wise tail-curvature measurements for the research layer.

Curvature is the unsigned bend at ``tail_M`` between the tail_S -> tail_M and
tail_M -> tail_E segments.  It is a geometric angle, so a straight tail is
0 degrees and a tighter bend produces a larger value up to 180 degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .research_angle import (
    RESEARCH_LIKELIHOOD_CUTOFF,
    RESEARCH_MAX_DISTANCE_CM,
    RESEARCH_MIN_DISTANCE_CM,
)
from .ticks import BeamCalibration, local_beam_segment_for_point, tick_line_y_at_x

if TYPE_CHECKING:
    from .dataset import TrialVideo
    from .detection import DLCFramePrediction, DLCPoint, DLCTracking


@dataclass(frozen=True)
class TailCurvatureMeasurement:
    """One independently valid tail bend and its calibrated beam position."""

    curvature_degrees: float
    back_paw_position_cm: float


@dataclass(frozen=True)
class TailCurvatureFrameRecord:
    """One audit-friendly tail-curvature value for a source video frame."""

    relative_video: str
    dataset: str
    day: str
    cage: str
    animal: str
    group: str
    trial: int
    frame: int
    time_seconds: float
    back_paw_position_cm: float
    tail_curvature_degrees: float


def calculate_tail_curvature(
    prediction: DLCFramePrediction | None,
    calibration: BeamCalibration | None,
    *,
    likelihood_cutoff: float = RESEARCH_LIKELIHOOD_CUTOFF,
) -> TailCurvatureMeasurement | None:
    """Return a valid 0--180 degree tail bend for one tracked frame.

    Curvature follows the common research-frame validity rule: the back paw
    must project into 0--90 cm, body center must be strictly above the local
    fall boundary, and every point required for this metric must meet the
    likelihood cutoff.  ``front_paw`` is intentionally not required because it
    is not part of the curvature geometry.
    """
    if prediction is None or calibration is None:
        return None

    back_paw = _high_confidence(prediction.visible_back_paw, likelihood_cutoff)
    body_center = _high_confidence(prediction.body_center, likelihood_cutoff)
    tail_start = _high_confidence(prediction.tail_start, likelihood_cutoff)
    tail_middle = _high_confidence(prediction.tail_middle, likelihood_cutoff)
    tail_end = _high_confidence(prediction.tail_end, likelihood_cutoff)
    required_points = (back_paw, body_center, tail_start, tail_middle, tail_end)
    if any(point is None or not _finite_point(point) for point in required_points):
        return None

    assert back_paw is not None
    assert body_center is not None
    assert tail_start is not None
    assert tail_middle is not None
    assert tail_end is not None

    back_paw_segment = local_beam_segment_for_point(calibration, back_paw.x, back_paw.y)
    if back_paw_segment is None or not (
        RESEARCH_MIN_DISTANCE_CM
        <= back_paw_segment.distance_cm
        <= RESEARCH_MAX_DISTANCE_CM
    ):
        return None

    boundary_y = tick_line_y_at_x(calibration, body_center.x)
    if boundary_y is None or body_center.y >= boundary_y:
        return None

    first_x = tail_middle.x - tail_start.x
    first_y = tail_middle.y - tail_start.y
    second_x = tail_end.x - tail_middle.x
    second_y = tail_end.y - tail_middle.y
    first_length = math.hypot(first_x, first_y)
    second_length = math.hypot(second_x, second_y)
    if first_length <= 1e-9 or second_length <= 1e-9:
        return None

    # atan2(|cross|, dot) is stable near straight and reverse configurations.
    # It directly returns the change in direction: 0 degrees for a straight
    # tail, 90 degrees for a right-angle bend, and 180 degrees for reversal.
    cross = (first_x * second_y) - (first_y * second_x)
    dot = (first_x * second_x) + (first_y * second_y)
    curvature_degrees = math.degrees(math.atan2(abs(cross), dot))
    if not math.isfinite(curvature_degrees):
        return None

    return TailCurvatureMeasurement(
        curvature_degrees=max(0.0, min(180.0, curvature_degrees)),
        back_paw_position_cm=back_paw_segment.distance_cm,
    )


def tail_curvature_frame_records(
    video: TrialVideo,
    tracking: DLCTracking,
    calibration: BeamCalibration,
    fps: float,
) -> list[TailCurvatureFrameRecord]:
    """Measure each independently valid frame in saved DLC tracking data.

    Invalid/low-confidence frames receive no record.  A later valid frame can
    resume the metric, but no value is interpolated across the missing gap.
    """
    frame_rate = fps if math.isfinite(fps) and fps > 0 else 15.0
    records: list[TailCurvatureFrameRecord] = []
    for frame, prediction in sorted(tracking.frames.items()):
        measurement = calculate_tail_curvature(prediction, calibration)
        if measurement is None:
            continue
        records.append(
            TailCurvatureFrameRecord(
                relative_video=video.relative_path,
                dataset=video.dataset,
                day=video.day,
                cage=video.cage_number,
                animal=video.rat_id,
                group=video.group,
                trial=video.trial,
                frame=frame,
                time_seconds=frame / frame_rate,
                back_paw_position_cm=measurement.back_paw_position_cm,
                tail_curvature_degrees=measurement.curvature_degrees,
            )
        )
    return records


def _high_confidence(point: DLCPoint | None, likelihood_cutoff: float) -> DLCPoint | None:
    if point is None or point.likelihood < likelihood_cutoff:
        return None
    return point


def _finite_point(point: DLCPoint) -> bool:
    return math.isfinite(point.x) and math.isfinite(point.y)
