"""Pure frame-by-frame rules for scoring a DLC-tracked beam trial.

This module deliberately contains no Tkinter, OpenCV, or file I/O. A caller loads
a high-confidence :class:`~rbtcv.detection.DLCTracking` once, analyzes it once, and
then asks the resulting timeline for the state at a video frame during playback.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .detection import DLCPoint, DLCTracking
from .scoring import BEAM_LENGTH_CM
from .ticks import BeamCalibration, BeamTick, interval_midpoint_distance_from_point, tick_line_y_at_x


WAITING = "waiting"
RUNNING = "running"
FELL = "fell"
REACHED = "reached"
FALL_BOUNDARY_MARGIN_PX = 15.0
# DLC points can sit a couple of pixels short of a tick even when the paw has
# visibly reached the platform. Require two consecutive high-confidence frames
# inside this small endpoint zone so a one-frame wobble cannot finish a trial.
ENDPOINT_TOLERANCE_PX = 3.0
ENDPOINT_CONFIRMATION_FRAMES = 2


@dataclass(frozen=True)
class TrackingFrameState:
    """Automatic trial state after processing one DLC prediction frame."""

    state: str
    start_frame: int | None
    end_frame: int | None
    farthest_distance_cm: int
    back_paw_x: float | None
    back_paw_y: float | None
    body_center_x: float | None
    body_center_y: float | None


@dataclass(frozen=True)
class TrackingTimeline:
    """Cached automatic-scoring state for a complete tracking CSV."""

    states: Mapping[int, TrackingFrameState]
    frame_numbers: tuple[int, ...]
    start_frame: int | None
    end_frame: int | None
    final_state: str
    farthest_distance_cm: int
    ended_at_video_end: bool = False

    def state_at(self, frame_number: int) -> TrackingFrameState | None:
        """Return the latest known state at or before a video frame.

        A tracking CSV normally contains every video frame. Returning the prior
        cached state also keeps the GUI informative if an unusual CSV omits one.
        """
        index = bisect_right(self.frame_numbers, frame_number) - 1
        if index < 0:
            return None
        return self.states[self.frame_numbers[index]]


def analyze_tracking_timeline(tracking: DLCTracking, calibration: BeamCalibration) -> TrackingTimeline:
    """Analyze one high-confidence DLC timeline against confirmed beam ticks.

    The visible *back* paw must reach or pass tick 0 in the direction of tick 120
    before timing begins. This intentionally includes the first reliable frame
    after an uncaptured or low-confidence crossing. After that start, a body center
    below the local tick-center line is a fall and takes precedence over a
    same-frame endpoint crossing. Otherwise, the visible back paw crossing tick
    120 ends a successful trial. If the trial reaches the final prediction frame
    after a valid start without either terminal event, the video end itself is
    treated as the completion time.
    """
    start_tick, end_tick = _required_endpoint_ticks(calibration)
    direction = _beam_direction(start_tick, end_tick)
    cutoff = tracking.likelihood_cutoff

    states: dict[int, TrackingFrameState] = {}
    frame_numbers = tuple(sorted(tracking.frames))
    state = WAITING
    start_frame: int | None = None
    end_frame: int | None = None
    farthest_distance_cm = 0
    previous_back_paw: DLCPoint | None = None
    endpoint_frames = 0
    ended_at_video_end = False

    for frame_number in frame_numbers:
        prediction = tracking.frames[frame_number]
        # These are exact, named DLC labels. ``DLCTracking`` has already removed
        # low-confidence points, and the extra check makes the rule safe for any
        # manually created tracking object too.
        back_paw = _high_confidence(prediction.visible_back_paw, cutoff)
        body_center = _high_confidence(prediction.body_center, cutoff)

        if state == WAITING and (
            _crossed_vertical_tick(previous_back_paw, back_paw, start_tick.x, direction)
            or _at_or_past_vertical_tick(back_paw, start_tick.x, direction)
        ):
            state = RUNNING
            start_frame = frame_number

        tick_line_y: float | None = None
        if state == RUNNING:
            if body_center is not None:
                tick_line_y = tick_line_y_at_x(calibration, body_center.x)
                if tick_line_y is not None:
                    distance_cm = interval_midpoint_distance_from_point(
                        calibration,
                        int(round(body_center.x)),
                        int(round(tick_line_y)),
                    )
                    farthest_distance_cm = max(farthest_distance_cm, distance_cm)
            # A fall wins over the endpoint if both events happen in the same frame.
            if (
                tick_line_y is not None
                and body_center is not None
                and body_center.y > tick_line_y + FALL_BOUNDARY_MARGIN_PX
            ):
                state = FELL
                end_frame = frame_number
            else:
                endpoint_frames = _endpoint_frame_count(
                    endpoint_frames,
                    back_paw,
                    end_tick.x,
                    direction,
                )
            if (
                state == RUNNING
                and (
                    _crossed_vertical_tick(previous_back_paw, back_paw, end_tick.x, direction)
                    or endpoint_frames >= ENDPOINT_CONFIRMATION_FRAMES
                )
            ):
                state = REACHED
                end_frame = frame_number
                farthest_distance_cm = BEAM_LENGTH_CM

        # Some recordings end before a completed beam traversal is visible. Once
        # the 0 cm start has been established, use the final video/tracking frame
        # as the requested completion time when no fall or 120 cm crossing was
        # observed. DLC emits one row for every video frame, so this is the video
        # end rather than an arbitrary missing-prediction gap.
        if state == RUNNING and frame_number == frame_numbers[-1]:
            state = REACHED
            end_frame = frame_number
            ended_at_video_end = True

        states[frame_number] = TrackingFrameState(
            state=state,
            start_frame=start_frame,
            end_frame=end_frame,
            farthest_distance_cm=farthest_distance_cm,
            back_paw_x=back_paw.x if back_paw is not None else None,
            back_paw_y=back_paw.y if back_paw is not None else None,
            body_center_x=body_center.x if body_center is not None else None,
            body_center_y=body_center.y if body_center is not None else None,
        )

        # Require a real prior frame observation for a directional crossing; a
        # missing/low-confidence back-paw point resets that continuity.
        previous_back_paw = back_paw

    return TrackingTimeline(
        states=MappingProxyType(states),
        frame_numbers=frame_numbers,
        start_frame=start_frame,
        end_frame=end_frame,
        final_state=state,
        farthest_distance_cm=farthest_distance_cm,
        ended_at_video_end=ended_at_video_end,
    )


def _required_endpoint_ticks(calibration: BeamCalibration) -> tuple[BeamTick, BeamTick]:
    ticks_by_distance = {tick.distance_cm: tick for tick in calibration.ticks}
    start_tick = ticks_by_distance.get(0)
    end_tick = ticks_by_distance.get(BEAM_LENGTH_CM)
    if start_tick is None or end_tick is None:
        raise ValueError("Tracking rules require exact confirmed 0 cm and 120 cm tick marks.")
    return start_tick, end_tick


def _beam_direction(start_tick: BeamTick, end_tick: BeamTick) -> int:
    if end_tick.x > start_tick.x:
        return 1
    if end_tick.x < start_tick.x:
        return -1
    raise ValueError("The confirmed 0 cm and 120 cm tick marks must have different x positions.")


def _high_confidence(point: DLCPoint | None, cutoff: float) -> DLCPoint | None:
    if point is None or point.likelihood < cutoff:
        return None
    return point


def _crossed_vertical_tick(
    previous: DLCPoint | None,
    current: DLCPoint | None,
    tick_x: float,
    direction: int,
) -> bool:
    if previous is None or current is None:
        return False
    previous_position = (previous.x - tick_x) * direction
    current_position = (current.x - tick_x) * direction
    return previous_position < 0 <= current_position

def _at_or_past_vertical_tick(current: DLCPoint | None, tick_x: float, direction: int) -> bool:
    """Fallback when the only low-confidence frame is the 0 cm crossing itself."""
    return current is not None and ((current.x - tick_x) * direction) >= 0


def _endpoint_frame_count(
    previous_count: int,
    current: DLCPoint | None,
    tick_x: float,
    direction: int,
) -> int:
    if current is None:
        return 0
    position = (current.x - tick_x) * direction
    return previous_count + 1 if position >= -ENDPOINT_TOLERANCE_PX else 0
