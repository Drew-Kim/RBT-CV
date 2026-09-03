from __future__ import annotations

import unittest

import numpy as np

from rbtcv.detection import DLCFramePrediction, DLCPoint, _line_to_frame_bounds, draw_tracking_overlay
from rbtcv.research_angle import calculate_tail_angle
from rbtcv.ticks import BeamCalibration, BeamTick, local_beam_segment_for_point


def calibration(*, tilted: bool = False) -> BeamCalibration:
    return BeamCalibration(
        key="research-angle",
        dataset="synthetic",
        day="D0",
        cage="1",
        subject="1",
        source_video="synthetic.avi",
        source_trial=1,
        frame_numbers=(0,),
        ticks=tuple(
            BeamTick(distance, 100 + distance, 100 + ((distance // 10) * 3 if tilted else 0))
            for distance in range(0, 121, 10)
        ),
        confirmed_at="test",
    )


def prediction(
    *,
    back_paw_x: float = 150.0,
    body_center_y: float = 90.0,
    tail_start: tuple[float, float] = (150.0, 80.0),
    tail_end: tuple[float, float] = (170.0, 80.0),
    front_likelihood: float = 0.99,
) -> DLCFramePrediction:
    return DLCFramePrediction(
        frame=0,
        points=(
            DLCPoint("back_paw", "paw", back_paw_x, 100.0, 0.99),
            DLCPoint("front_paw", "paw", 160.0, 90.0, front_likelihood),
            DLCPoint("body_center", "body", back_paw_x, body_center_y, 0.99),
            DLCPoint("tail_S", "tail", *tail_start, 0.99),
            DLCPoint("tail_M", "tail", 160.0, 80.0, 0.99),
            DLCPoint("tail_E", "tail", *tail_end, 0.99),
        ),
    )


class TailAngleTests(unittest.TestCase):
    def test_horizontal_beam_sign_and_parallel_directions(self) -> None:
        beam = calibration()
        upper = calculate_tail_angle(prediction(tail_end=(170.0, 60.0)), beam)
        lower = calculate_tail_angle(prediction(tail_end=(170.0, 100.0)), beam)
        forward = calculate_tail_angle(prediction(tail_end=(170.0, 80.0)), beam)
        reverse = calculate_tail_angle(prediction(tail_end=(130.0, 80.0)), beam)

        self.assertAlmostEqual(upper.angle_degrees if upper else 0.0, 45.0, places=5)
        self.assertAlmostEqual(lower.angle_degrees if lower else 0.0, -45.0, places=5)
        self.assertAlmostEqual(forward.angle_degrees if forward else 99.0, 0.0, places=5)
        self.assertAlmostEqual(reverse.angle_degrees if reverse else 99.0, 0.0, places=5)

    def test_tilted_beam_uses_local_tangent_and_upper_normal(self) -> None:
        beam = calibration(tilted=True)
        segment = local_beam_segment_for_point(beam, 150.0, 80.0)
        self.assertIsNotNone(segment)
        assert segment is not None
        tail_start = (150.0, 80.0)
        tail_end = (
            tail_start[0] + (20.0 * segment.upper_normal_x),
            tail_start[1] + (20.0 * segment.upper_normal_y),
        )

        result = calculate_tail_angle(prediction(tail_start=tail_start, tail_end=tail_end), beam)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.angle_degrees, 90.0, places=5)

    def test_nearest_segment_and_continuous_back_paw_position(self) -> None:
        beam = calibration()
        segment = local_beam_segment_for_point(beam, 155.0, 110.0)
        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertEqual((segment.start.distance_cm, segment.end.distance_cm), (50, 60))
        self.assertAlmostEqual(segment.distance_cm, 55.0, places=5)

    def test_research_gate_requires_all_six_points_and_valid_bounds(self) -> None:
        beam = calibration()
        self.assertIsNotNone(calculate_tail_angle(prediction(back_paw_x=100.0), beam))
        self.assertIsNotNone(calculate_tail_angle(prediction(back_paw_x=190.0), beam))
        self.assertIsNone(calculate_tail_angle(prediction(back_paw_x=99.0), beam))
        self.assertIsNone(calculate_tail_angle(prediction(back_paw_x=195.0), beam))
        self.assertIsNone(calculate_tail_angle(prediction(body_center_y=100.0), beam))
        self.assertIsNone(calculate_tail_angle(prediction(front_likelihood=0.59), beam))
        self.assertIsNone(calculate_tail_angle(prediction(tail_end=(150.0, 80.0)), beam))

    def test_overlay_accepts_absent_or_valid_angle(self) -> None:
        frame = np.zeros((180, 280, 3), dtype=np.uint8)
        tracked = prediction()
        no_angle = draw_tracking_overlay(frame, tracked)
        angle = calculate_tail_angle(tracked, calibration())
        self.assertIsNotNone(angle)
        with_angle = draw_tracking_overlay(frame, tracked, tail_angle=angle)
        self.assertFalse(np.array_equal(no_angle, with_angle))

    def test_tail_axis_extends_to_the_visible_frame_edges(self) -> None:
        diagonal = _line_to_frame_bounds((10.0, 10.0), (20.0, 20.0), 100, 80)
        horizontal = _line_to_frame_bounds((10.0, 30.0), (20.0, 30.0), 100, 80)

        self.assertEqual(diagonal, ((0.0, 0.0), (79.0, 79.0)))
        self.assertEqual(horizontal, ((0.0, 30.0), (99.0, 30.0)))
