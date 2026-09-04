"""Research timing for ordered 10 cm back-paw crossings on the beam.

This module is deliberately independent of Tkinter, OpenCV, Excel, and legacy
fall scoring. It derives interval times from the full saved DLC tracking stream
and a confirmed beam calibration, so recovery posture contributes real elapsed
time rather than truncating a trial at the first fall boundary crossing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .ticks import BeamCalibration, tick_line_y_at_x

if TYPE_CHECKING:
    from .dataset import TrialVideo
    from .detection import DLCTracking


INTERVAL_STARTS_CM = tuple(range(0, 90, 10))
INTERVAL_ENDS_CM = tuple(start + 10 for start in INTERVAL_STARTS_CM)
INTERVAL_BOUNDARIES_CM = tuple(range(0, 91, 10))


@dataclass(frozen=True)
class TickIntervalRecord:
    """One valid elapsed-time measurement between adjacent beam ticks."""

    relative_video: str
    dataset: str
    day: str
    cage: str
    animal: str
    group: str
    trial: int
    interval_start_cm: int
    interval_end_cm: int
    start_frame: int
    end_frame: int
    elapsed_seconds: float


def tick_interval_records(
    video: TrialVideo,
    tracking: DLCTracking,
    calibration: BeamCalibration,
    fps: float,
) -> list[TickIntervalRecord]:
    """Return valid 0-10 through 80-90 cm back-paw interval times.

    A sequence begins at the first reliable on-beam back-paw point at or beyond
    0 cm. Each next tick uses the first reliable on-beam point at or beyond that
    tick, which handles fast movement when no image captures the exact line
    crossing. A missing/low-confidence required point or missing CSV frame after
    the sequence starts terminates the remaining sequence: no interpolation or
    gap-bridging is performed.

    A reliable below-boundary posture is a pause rather than a hard stop. The
    next reliable recovered point can satisfy the next tick, and its actual
    timestamp retains the elapsed recovery time in the interval duration.
    """
    frames_by_tick = ordered_tick_crossing_frames(tracking, calibration)
    frame_rate = fps if math.isfinite(fps) and fps > 0 else 15.0
    records: list[TickIntervalRecord] = []
    for start_cm, end_cm in zip(INTERVAL_STARTS_CM, INTERVAL_ENDS_CM):
        start_frame = frames_by_tick.get(start_cm)
        end_frame = frames_by_tick.get(end_cm)
        if start_frame is None or end_frame is None or end_frame < start_frame:
            continue
        records.append(
            TickIntervalRecord(
                relative_video=video.relative_path,
                dataset=video.dataset,
                day=video.day,
                cage=video.cage_number,
                animal=video.rat_id,
                group=video.group,
                trial=video.trial,
                interval_start_cm=start_cm,
                interval_end_cm=end_cm,
                start_frame=start_frame,
                end_frame=end_frame,
                elapsed_seconds=(end_frame - start_frame) / frame_rate,
            )
        )
    return records


def ordered_tick_crossing_frames(
    tracking: DLCTracking,
    calibration: BeamCalibration,
) -> dict[int, int]:
    """Find ordered reliable forward tick timestamps from 0 through 90 cm.

    The returned mapping contains only the uninterrupted prefix of successfully
    observed crossings. A low-confidence/missing back-paw frame after 0 cm is
    a hard stop, preserving the requested no-gap-bridging rule.
    """
    if not _has_required_ticks(calibration):
        return {}

    cutoff = tracking.likelihood_cutoff
    crossings: dict[int, int] = {}
    next_index = 0
    started = False
    previous_frame: int | None = None

    for frame, prediction in sorted(tracking.frames.items()):
        back_paw = prediction.visible_back_paw
        body_center = prediction.body_center
        if started and previous_frame is not None and frame != previous_frame + 1:
            break
        previous_frame = frame
        if (
            back_paw is None
            or back_paw.likelihood < cutoff
            or body_center is None
            or body_center.likelihood < cutoff
            or not math.isfinite(back_paw.x)
            or not math.isfinite(back_paw.y)
            or not math.isfinite(body_center.x)
            or not math.isfinite(body_center.y)
        ):
            if started:
                break
            continue

        distance_cm = project_back_paw_distance_cm(calibration, back_paw.x, back_paw.y)
        if distance_cm is None:
            if started:
                break
            continue

        boundary_y = tick_line_y_at_x(calibration, body_center.x)
        if boundary_y is None:
            if started:
                break
            continue
        # A confident recovery/fall posture does not erase already observed
        # crossings or pause the wall-clock timer. It simply cannot establish
        # a new tick timestamp until the mouse is back above the boundary.
        if body_center.y >= boundary_y:
            continue

        if not started:
            if distance_cm < INTERVAL_BOUNDARIES_CM[0]:
                continue
            started = True

        # A fast paw may pass more than one 10 cm line between video frames.
        # Record each skipped crossing at this first reliable post-tick frame.
        while (
            next_index < len(INTERVAL_BOUNDARIES_CM)
            and distance_cm >= INTERVAL_BOUNDARIES_CM[next_index]
        ):
            crossings[INTERVAL_BOUNDARIES_CM[next_index]] = frame
            next_index += 1
        if next_index == len(INTERVAL_BOUNDARIES_CM):
            break

    return crossings


def project_back_paw_distance_cm(
    calibration: BeamCalibration,
    x: float,
    y: float,
) -> float | None:
    """Project an image point continuously along the nearest calibrated beam segment.

    Unlike tail-angle geometry, this projection also supports a vertical beam:
    timing has no dependence on an upper-video normal. The projection remains
    unclamped along its chosen segment so an observed paw can be recognized as
    already beyond a tick even when it lands just past the segment endpoint.
    """
    ticks = sorted(calibration.ticks, key=lambda tick: tick.distance_cm)
    best_distance: float | None = None
    best_error = float("inf")
    for start, end in zip(ticks, ticks[1:]):
        span_cm = end.distance_cm - start.distance_cm
        delta_x = float(end.x - start.x)
        delta_y = float(end.y - start.y)
        length_squared = (delta_x * delta_x) + (delta_y * delta_y)
        if span_cm <= 0 or length_squared <= 0:
            continue

        fraction = (((x - start.x) * delta_x) + ((y - start.y) * delta_y)) / length_squared
        nearest_fraction = max(0.0, min(1.0, fraction))
        projected_x = start.x + (nearest_fraction * delta_x)
        projected_y = start.y + (nearest_fraction * delta_y)
        error = ((x - projected_x) ** 2) + ((y - projected_y) ** 2)
        if error < best_error:
            best_error = error
            best_distance = start.distance_cm + (fraction * span_cm)
    return best_distance


def _has_required_ticks(calibration: BeamCalibration) -> bool:
    distances = {tick.distance_cm for tick in calibration.ticks}
    return set(INTERVAL_BOUNDARIES_CM).issubset(distances)
