"""Pure calibrated rear-paw to body-center distance measurements.

This research metric uses only the two landmarks needed for its geometry:
``back_paw`` and ``body_center``.  A valid frame must still be within the
calibrated 0--90 cm research window and above the local fall boundary, but a
weak tail or front-paw label does not discard an otherwise usable posture.
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
class BackPawBodyDistanceMeasurement:
    """One valid local-scale rear-paw/body-center distance and beam position."""

    distance_cm: float
    back_paw_position_cm: float


@dataclass(frozen=True)
class BackPawBodyDistanceFrameRecord:
    """One audit-friendly rear-paw/body-center value for a source video frame."""

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
    back_paw_body_distance_cm: float


def calculate_back_paw_body_distance(
    prediction: DLCFramePrediction | None,
    calibration: BeamCalibration | None,
    *,
    likelihood_cutoff: float = RESEARCH_LIKELIHOOD_CUTOFF,
) -> BackPawBodyDistanceMeasurement | None:
    """Return a valid rear-paw/body-center Euclidean distance in calibrated cm.

    The confirmed 10 cm beam segment nearest the midpoint of the two landmarks
    supplies the local pixels-to-centimetres scale.  The back paw alone defines
    the inclusive 0--90 cm research window.  ``body_center`` must be strictly
    above its local fall-boundary line; a frame on or below the line is omitted.
    """
    if prediction is None or calibration is None:
        return None

    back_paw = _high_confidence(prediction.visible_back_paw, likelihood_cutoff)
    body_center = _high_confidence(prediction.body_center, likelihood_cutoff)
    required_points = (back_paw, body_center)
    if any(point is None or not _finite_point(point) for point in required_points):
        return None

    assert back_paw is not None
    assert body_center is not None

    back_paw_segment = local_beam_segment_for_point(calibration, back_paw.x, back_paw.y)
    if back_paw_segment is None or not (
        RESEARCH_MIN_DISTANCE_CM <= back_paw_segment.distance_cm <= RESEARCH_MAX_DISTANCE_CM
    ):
        return None

    boundary_y = tick_line_y_at_x(calibration, body_center.x)
    if boundary_y is None or body_center.y >= boundary_y:
        return None

    midpoint_x = (back_paw.x + body_center.x) / 2.0
    midpoint_y = (back_paw.y + body_center.y) / 2.0
    scale_segment = local_beam_segment_for_point(calibration, midpoint_x, midpoint_y)
    if scale_segment is None:
        return None

    segment_pixels = math.hypot(
        scale_segment.end.x - scale_segment.start.x,
        scale_segment.end.y - scale_segment.start.y,
    )
    segment_cm = scale_segment.end.distance_cm - scale_segment.start.distance_cm
    if segment_pixels <= 1e-9 or segment_cm <= 0:
        return None

    distance_pixels = math.hypot(body_center.x - back_paw.x, body_center.y - back_paw.y)
    distance_cm = distance_pixels * (segment_cm / segment_pixels)
    if not math.isfinite(distance_cm):
        return None

    return BackPawBodyDistanceMeasurement(
        distance_cm=distance_cm,
        back_paw_position_cm=back_paw_segment.distance_cm,
    )


def back_paw_body_distance_frame_records(
    video: TrialVideo,
    tracking: DLCTracking,
    calibration: BeamCalibration,
    fps: float,
) -> list[BackPawBodyDistanceFrameRecord]:
    """Measure every independently valid frame in a saved DLC tracking stream.

    There is deliberately no interpolation: invalid frames produce no record,
    while a later valid frame can resume the metric.
    """
    frame_rate = fps if math.isfinite(fps) and fps > 0 else 15.0
    records: list[BackPawBodyDistanceFrameRecord] = []
    for frame, prediction in sorted(tracking.frames.items()):
        measurement = calculate_back_paw_body_distance(prediction, calibration)
        if measurement is None:
            continue
        records.append(
            BackPawBodyDistanceFrameRecord(
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
                back_paw_body_distance_cm=measurement.distance_cm,
            )
        )
    return records


def _high_confidence(point: DLCPoint | None, likelihood_cutoff: float) -> DLCPoint | None:
    if (
        point is None
        or not math.isfinite(point.likelihood)
        or point.likelihood < likelihood_cutoff
    ):
        return None
    return point


def _finite_point(point: DLCPoint) -> bool:
    return math.isfinite(point.x) and math.isfinite(point.y)
