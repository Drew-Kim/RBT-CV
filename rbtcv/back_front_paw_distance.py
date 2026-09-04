"""Pure calibrated back-paw to front-paw distance measurements.

This research metric uses only the landmarks needed for the measurement: a
high-confidence back paw and front paw in the calibrated 0--90 cm window. It
therefore does not discard an otherwise valid stance merely because a tail point
is weak. The geometry remains independent of the GUI, Excel, and legacy
first-fall scorer.
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
from .ticks import BeamCalibration, local_beam_segment_for_point

if TYPE_CHECKING:
    from .dataset import TrialVideo
    from .detection import DLCFramePrediction, DLCPoint, DLCTracking


@dataclass(frozen=True)
class BackFrontPawDistanceMeasurement:
    """One valid local-scale paw separation and its back-paw beam position."""

    distance_cm: float
    back_paw_position_cm: float


@dataclass(frozen=True)
class BackFrontPawDistanceFrameRecord:
    """One frame-wise, audit-friendly back/front paw distance measurement."""

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
    back_front_paw_distance_cm: float


def calculate_back_front_paw_distance(
    prediction: DLCFramePrediction | None,
    calibration: BeamCalibration | None,
    *,
    likelihood_cutoff: float = RESEARCH_LIKELIHOOD_CUTOFF,
) -> BackFrontPawDistanceMeasurement | None:
    """Return a valid back-paw/front-paw Euclidean distance in calibrated cm.

    The local image-to-cm scale is taken from the confirmed 10 cm beam segment
    nearest the midpoint of the two paws. The back paw independently defines
    the 0--90 cm research window, so the metric remains comparable across
    recordings with slightly tilted or mirrored beams.
    """
    if prediction is None or calibration is None:
        return None

    back_paw = _high_confidence(prediction.visible_back_paw, likelihood_cutoff)
    front_paw = _high_confidence(prediction.visible_front_paw, likelihood_cutoff)
    required_points = (back_paw, front_paw)
    if any(point is None or not _finite_point(point) for point in required_points):
        return None

    assert back_paw is not None
    assert front_paw is not None

    back_paw_segment = local_beam_segment_for_point(calibration, back_paw.x, back_paw.y)
    if back_paw_segment is None or not (
        RESEARCH_MIN_DISTANCE_CM <= back_paw_segment.distance_cm <= RESEARCH_MAX_DISTANCE_CM
    ):
        return None

    midpoint_x = (back_paw.x + front_paw.x) / 2.0
    midpoint_y = (back_paw.y + front_paw.y) / 2.0
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

    paw_pixels = math.hypot(front_paw.x - back_paw.x, front_paw.y - back_paw.y)
    distance_cm = paw_pixels * (segment_cm / segment_pixels)
    if not math.isfinite(distance_cm):
        return None

    return BackFrontPawDistanceMeasurement(
        distance_cm=distance_cm,
        back_paw_position_cm=back_paw_segment.distance_cm,
    )


def back_front_paw_distance_frame_records(
    video: TrialVideo,
    tracking: DLCTracking,
    calibration: BeamCalibration,
    fps: float,
) -> list[BackFrontPawDistanceFrameRecord]:
    """Measure every independently valid frame in a saved DLC tracking stream."""
    frame_rate = fps if math.isfinite(fps) and fps > 0 else 15.0
    records: list[BackFrontPawDistanceFrameRecord] = []
    for frame, prediction in sorted(tracking.frames.items()):
        measurement = calculate_back_front_paw_distance(prediction, calibration)
        if measurement is None:
            continue
        records.append(
            BackFrontPawDistanceFrameRecord(
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
                back_front_paw_distance_cm=measurement.distance_cm,
            )
        )
    return records


def _high_confidence(point: DLCPoint | None, likelihood_cutoff: float) -> DLCPoint | None:
    if point is None or point.likelihood < likelihood_cutoff:
        return None
    return point


def _finite_point(point: DLCPoint) -> bool:
    return math.isfinite(point.x) and math.isfinite(point.y)
