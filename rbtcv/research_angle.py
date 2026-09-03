"""Pure signed-tail-angle geometry for the six-landmark research model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .ticks import BeamCalibration, local_beam_segment_for_point, tick_line_y_at_x

if TYPE_CHECKING:
    from .dataset import TrialVideo
    from .detection import DLCFramePrediction, DLCPoint
    from .detection import DLCTracking


RESEARCH_LIKELIHOOD_CUTOFF = 0.60
RESEARCH_MIN_DISTANCE_CM = 0.0
RESEARCH_MAX_DISTANCE_CM = 90.0


@dataclass(frozen=True)
class TailAngleMeasurement:
    """A valid signed tail angle plus the geometry needed for a display overlay."""

    angle_degrees: float
    back_paw_distance_cm: float
    tail_start_x: float
    tail_start_y: float
    tail_end_x: float
    tail_end_y: float
    beam_tangent_x: float
    beam_tangent_y: float


@dataclass(frozen=True)
class TailAngleFrameRecord:
    """One valid, audit-friendly tail-angle measurement for an Excel row."""

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
    signed_tail_angle_degrees: float


def calculate_tail_angle(
    prediction: DLCFramePrediction | None,
    calibration: BeamCalibration | None,
    *,
    likelihood_cutoff: float = RESEARCH_LIKELIHOOD_CUTOFF,
) -> TailAngleMeasurement | None:
    """Return the valid 0--90 cm signed tail angle for one tracked frame.

    The beam is treated as an undirected line: both tail directions parallel to it
    are zero degrees. The sign instead represents whether the tail points toward
    the upper (+) or lower (-) side of the video.
    """
    if prediction is None or calibration is None:
        return None

    back_paw = _high_confidence(prediction.visible_back_paw, likelihood_cutoff)
    front_paw = _high_confidence(prediction.visible_front_paw, likelihood_cutoff)
    body_center = _high_confidence(prediction.body_center, likelihood_cutoff)
    tail_start = _high_confidence(prediction.tail_start, likelihood_cutoff)
    tail_middle = _high_confidence(prediction.tail_middle, likelihood_cutoff)
    tail_end = _high_confidence(prediction.tail_end, likelihood_cutoff)
    if None in (back_paw, front_paw, body_center, tail_start, tail_middle, tail_end):
        return None

    # The None check above narrows these values for runtime correctness; the
    # assignments keep the following geometry simple and explicit.
    assert back_paw is not None
    assert body_center is not None
    assert tail_start is not None
    assert tail_end is not None

    back_paw_segment = local_beam_segment_for_point(calibration, back_paw.x, back_paw.y)
    if back_paw_segment is None or not (
        RESEARCH_MIN_DISTANCE_CM <= back_paw_segment.distance_cm <= RESEARCH_MAX_DISTANCE_CM
    ):
        return None

    boundary_y = tick_line_y_at_x(calibration, body_center.x)
    if boundary_y is None or body_center.y >= boundary_y:
        return None

    tail_segment = local_beam_segment_for_point(calibration, tail_start.x, tail_start.y)
    if tail_segment is None:
        return None

    tail_x = tail_end.x - tail_start.x
    tail_y = tail_end.y - tail_start.y
    if math.hypot(tail_x, tail_y) <= 1e-9:
        return None

    along_beam = (tail_x * tail_segment.tangent_x) + (tail_y * tail_segment.tangent_y)
    toward_upper_video = (
        (tail_x * tail_segment.upper_normal_x) + (tail_y * tail_segment.upper_normal_y)
    )
    angle_degrees = math.degrees(math.atan2(toward_upper_video, abs(along_beam)))

    return TailAngleMeasurement(
        angle_degrees=angle_degrees,
        back_paw_distance_cm=back_paw_segment.distance_cm,
        tail_start_x=tail_start.x,
        tail_start_y=tail_start.y,
        tail_end_x=tail_end.x,
        tail_end_y=tail_end.y,
        beam_tangent_x=tail_segment.tangent_x,
        beam_tangent_y=tail_segment.tangent_y,
    )


def tail_angle_frame_records(
    video: TrialVideo,
    tracking: DLCTracking,
    calibration: BeamCalibration,
    fps: float,
) -> list[TailAngleFrameRecord]:
    """Measure every valid research-angle frame in one tracked video.

    This deliberately evaluates the whole tracking CSV rather than a legacy
    scoring start/stop range, so measurements can resume after fall recovery.
    """
    frame_rate = fps if fps > 0 else 15.0
    records: list[TailAngleFrameRecord] = []
    for frame, prediction in sorted(tracking.frames.items()):
        measurement = calculate_tail_angle(prediction, calibration)
        if measurement is None:
            continue
        records.append(
            TailAngleFrameRecord(
                relative_video=video.relative_path,
                dataset=video.dataset,
                day=video.day,
                cage=video.cage_number,
                animal=video.rat_id,
                group=video.group,
                trial=video.trial,
                frame=frame,
                time_seconds=frame / frame_rate,
                back_paw_position_cm=measurement.back_paw_distance_cm,
                signed_tail_angle_degrees=measurement.angle_degrees,
            )
        )
    return records


def _high_confidence(point: DLCPoint | None, likelihood_cutoff: float) -> DLCPoint | None:
    if point is None or point.likelihood < likelihood_cutoff:
        return None
    return point
