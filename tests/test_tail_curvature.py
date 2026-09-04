from __future__ import annotations

import math
from pathlib import Path
import unittest

from rbtcv.dataset import TrialVideo
from rbtcv.detection import DLCFramePrediction, DLCPoint, DLCTracking
from rbtcv.tail_curvature import calculate_tail_curvature, tail_curvature_frame_records
from rbtcv.ticks import BeamCalibration, BeamTick


def calibration() -> BeamCalibration:
    """A horizontal 0--120 cm beam whose tick-center line is y=100."""
    return BeamCalibration(
        key="tail-curvature",
        dataset="Synthetic",
        day="BL",
        cage="1",
        subject="1",
        source_video="synthetic.avi",
        source_trial=1,
        frame_numbers=(0,),
        ticks=tuple(
            BeamTick(distance, 100 + distance, 100) for distance in range(0, 121, 10)
        ),
        confirmed_at="test",
    )


def curvature_prediction(
    *,
    frame: int = 0,
    back_paw_x: float = 150.0,
    body_center_y: float = 90.0,
    tail_start: tuple[float, float] = (140.0, 80.0),
    tail_middle: tuple[float, float] = (160.0, 80.0),
    tail_end: tuple[float, float] = (180.0, 80.0),
    back_likelihood: float = 0.99,
    body_likelihood: float = 0.99,
    tail_start_likelihood: float = 0.99,
    tail_middle_likelihood: float = 0.99,
    tail_end_likelihood: float = 0.99,
    include_back: bool = True,
    include_body: bool = True,
    include_tail_start: bool = True,
    include_tail_middle: bool = True,
    include_tail_end: bool = True,
) -> DLCFramePrediction:
    """Build a curvature frame without a front paw (which is not required)."""
    points: list[DLCPoint] = []
    if include_back:
        points.append(DLCPoint("back_paw", "paw", back_paw_x, 100.0, back_likelihood))
    if include_body:
        points.append(DLCPoint("body_center", "body", back_paw_x, body_center_y, body_likelihood))
    if include_tail_start:
        points.append(DLCPoint("tail_S", "tail", *tail_start, tail_start_likelihood))
    if include_tail_middle:
        points.append(DLCPoint("tail_M", "tail", *tail_middle, tail_middle_likelihood))
    if include_tail_end:
        points.append(DLCPoint("tail_E", "tail", *tail_end, tail_end_likelihood))
    return DLCFramePrediction(frame=frame, points=tuple(points))


def trial_video() -> TrialVideo:
    return TrialVideo(
        dataset="Synthetic",
        day="D3",
        group="C7",
        subject="3",
        trial=2,
        date="2026-01-01",
        clock="120000",
        path=Path("synthetic.avi"),
        relative_path="data/Synthetic/D3/C7_3_T2_2026-01-01-120000-0000.avi",
    )


class TailCurvatureTests(unittest.TestCase):
    def test_reports_zero_straight_ninety_right_angle_and_180_reversal(self) -> None:
        beam = calibration()
        straight = calculate_tail_curvature(curvature_prediction(), beam)
        right_angle = calculate_tail_curvature(
            curvature_prediction(tail_end=(160.0, 60.0)), beam
        )
        reversal = calculate_tail_curvature(
            curvature_prediction(tail_end=(140.0, 80.0)), beam
        )

        self.assertIsNotNone(straight)
        self.assertIsNotNone(right_angle)
        self.assertIsNotNone(reversal)
        assert straight is not None
        assert right_angle is not None
        assert reversal is not None
        self.assertAlmostEqual(straight.curvature_degrees, 0.0, places=6)
        self.assertAlmostEqual(right_angle.curvature_degrees, 90.0, places=6)
        self.assertAlmostEqual(reversal.curvature_degrees, 180.0, places=6)

    def test_curvature_is_rotation_and_mirror_invariant(self) -> None:
        beam = calibration()
        # The original, a 90-degree bend at tail_M.
        original = curvature_prediction(tail_end=(160.0, 60.0))
        # Rotate the same three-point geometry by 90 degrees about tail_M.
        rotated = curvature_prediction(
            tail_start=(160.0, 60.0),
            tail_middle=(160.0, 80.0),
            tail_end=(180.0, 80.0),
        )
        # Reflect it about the vertical line through tail_M.
        mirrored = curvature_prediction(
            tail_start=(180.0, 80.0),
            tail_middle=(160.0, 80.0),
            tail_end=(160.0, 60.0),
        )

        values = [
            calculate_tail_curvature(prediction, beam)
            for prediction in (original, rotated, mirrored)
        ]
        self.assertTrue(all(value is not None for value in values))
        self.assertEqual(len(values), 3)
        for value in values:
            assert value is not None
            self.assertAlmostEqual(value.curvature_degrees, 90.0, places=6)

    def test_back_paw_gate_is_0_to_90_cm_inclusive_and_front_paw_is_not_needed(self) -> None:
        beam = calibration()
        at_zero = calculate_tail_curvature(curvature_prediction(back_paw_x=100.0), beam)
        at_ninety = calculate_tail_curvature(curvature_prediction(back_paw_x=190.0), beam)
        before_zero = calculate_tail_curvature(curvature_prediction(back_paw_x=99.0), beam)
        after_ninety = calculate_tail_curvature(curvature_prediction(back_paw_x=191.0), beam)

        self.assertIsNotNone(at_zero)
        self.assertIsNotNone(at_ninety)
        self.assertIsNone(before_zero)
        self.assertIsNone(after_ninety)

    def test_body_center_must_be_strictly_above_the_fall_boundary(self) -> None:
        beam = calibration()
        self.assertIsNotNone(calculate_tail_curvature(curvature_prediction(body_center_y=99.9), beam))
        self.assertIsNone(calculate_tail_curvature(curvature_prediction(body_center_y=100.0), beam))
        self.assertIsNone(calculate_tail_curvature(curvature_prediction(body_center_y=100.1), beam))

    def test_rejects_missing_or_low_confidence_required_landmarks(self) -> None:
        beam = calibration()
        cases = {
            "low-confidence back paw": curvature_prediction(back_likelihood=0.59),
            "low-confidence body center": curvature_prediction(body_likelihood=0.59),
            "low-confidence tail start": curvature_prediction(tail_start_likelihood=0.59),
            "low-confidence tail middle": curvature_prediction(tail_middle_likelihood=0.59),
            "low-confidence tail end": curvature_prediction(tail_end_likelihood=0.59),
            "missing back paw": curvature_prediction(include_back=False),
            "missing body center": curvature_prediction(include_body=False),
            "missing tail start": curvature_prediction(include_tail_start=False),
            "missing tail middle": curvature_prediction(include_tail_middle=False),
            "missing tail end": curvature_prediction(include_tail_end=False),
        }

        for description, prediction in cases.items():
            with self.subTest(description=description):
                self.assertIsNone(calculate_tail_curvature(prediction, beam))

    def test_rejects_nonfinite_or_degenerate_tail_geometry(self) -> None:
        beam = calibration()
        cases = {
            "nonfinite tail start": curvature_prediction(tail_start=(math.nan, 80.0)),
            "nonfinite tail middle": curvature_prediction(tail_middle=(160.0, math.nan)),
            "nonfinite tail end": curvature_prediction(tail_end=(math.inf, 80.0)),
            "start equals middle": curvature_prediction(tail_start=(160.0, 80.0)),
            "middle equals end": curvature_prediction(tail_end=(160.0, 80.0)),
        }

        for description, prediction in cases.items():
            with self.subTest(description=description):
                self.assertIsNone(calculate_tail_curvature(prediction, beam))

    def test_frame_records_are_sorted_and_use_the_supplied_fps(self) -> None:
        tracking = DLCTracking(
            Path("synthetic.csv"),
            {
                6: curvature_prediction(frame=6),
                0: curvature_prediction(frame=0),
                3: curvature_prediction(frame=3),
            },
        )

        records = tail_curvature_frame_records(trial_video(), tracking, calibration(), fps=3.0)

        self.assertEqual([record.frame for record in records], [0, 3, 6])
        self.assertEqual([record.time_seconds for record in records], [0.0, 1.0, 2.0])
        self.assertEqual([record.back_paw_position_cm for record in records], [50.0, 50.0, 50.0])
        self.assertEqual([record.tail_curvature_degrees for record in records], [0.0, 0.0, 0.0])
        self.assertEqual(records[0].relative_video, trial_video().relative_path)
        self.assertEqual(records[0].day, "D3")
        self.assertEqual(records[0].cage, "7")
        self.assertEqual(records[0].animal, "3")
        self.assertEqual(records[0].trial, 2)


if __name__ == "__main__":
    unittest.main()
