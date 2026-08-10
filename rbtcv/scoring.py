from __future__ import annotations

from dataclasses import dataclass


OUTCOME_REACHED = "reached"
OUTCOME_FELL = "fell"
OUTCOMES = {OUTCOME_REACHED, OUTCOME_FELL}

# Rotating beam scoring rules used throughout the GUI.
FALL_MAX_SECONDS = 60.0
BEAM_LENGTH_CM = 120
BEAM_TICK_STEP_CM = 10
FALL_DISTANCE_STEP_CM = 5
BEAM_TICK_MARKS_CM = tuple(range(0, BEAM_LENGTH_CM + BEAM_TICK_STEP_CM, BEAM_TICK_STEP_CM))
DISTANCE_MARKS_CM = tuple(range(0, BEAM_LENGTH_CM + FALL_DISTANCE_STEP_CM, FALL_DISTANCE_STEP_CM))


@dataclass(frozen=True)
class PawMark:
    frame: int
    x: int
    y: int


def normalize_outcome(value: str) -> str:
    outcome = value.strip().lower()
    if outcome in OUTCOMES:
        return outcome
    return OUTCOME_REACHED


def normalize_distance_cm(distance_cm: int | float | str) -> int:
    try:
        value = int(float(distance_cm))
    except (TypeError, ValueError):
        value = 0

    # Distances should stay on the beam and use 5 cm scoring steps.
    value = max(0, min(BEAM_LENGTH_CM, value))
    return int(round(value / FALL_DISTANCE_STEP_CM) * FALL_DISTANCE_STEP_CM)


def distance_for_outcome(outcome: str, selected_distance_cm: int) -> int:
    # Reaching the platform always counts as the full beam length.
    if normalize_outcome(outcome) == OUTCOME_REACHED:
        return BEAM_LENGTH_CM

    # Falls keep the selected or estimated fall distance.
    return normalize_distance_cm(selected_distance_cm)


def raw_crossing_time_seconds(start: PawMark | None, stop: PawMark | None, fps: float) -> float | None:
    if start is None or stop is None:
        return None
    if fps == 0:
        return None
    return (stop.frame - start.frame) / fps


def scored_crossing_time_seconds(outcome: str, raw_crossing_time: float | None) -> float | None:
    # Any fall gets the maximum time, even if the observed fall happened sooner.
    if normalize_outcome(outcome) == OUTCOME_FELL:
        return FALL_MAX_SECONDS

    if raw_crossing_time is None:
        return None

    return raw_crossing_time


def max_time_applied(outcome: str) -> str:
    if normalize_outcome(outcome) == OUTCOME_FELL:
        return "yes"
    return "no"



def result_text(outcome: str, raw_crossing_time: float | None, scored_time: float | None, distance_cm: int) -> str:
    if scored_time is None:
        return "Crossing time -"

    # Negative time means the stop/fall mark was placed before the start mark.
    if raw_crossing_time is not None and raw_crossing_time < 0:
        return "Crossing time invalid"

    if normalize_outcome(outcome) == OUTCOME_FELL:
        return f"Scored time {FALL_MAX_SECONDS:.2f} sec | fall distance {distance_cm} cm"

    return f"Crossing time {scored_time:.2f} sec | distance {BEAM_LENGTH_CM} cm"

def distance_status_text(outcome: str, distance_cm: int | float | str) -> str:
    """Legacy GUI label text backed by the current 5 cm scoring rules."""
    distance = distance_for_outcome(outcome, normalize_distance_cm(distance_cm))
    return "Reached platform: 120 cm" if normalize_outcome(outcome) == OUTCOME_REACHED else f"Fall distance: {distance} cm"
