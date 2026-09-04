from __future__ import annotations

import math
from pathlib import Path
import unittest

from rbtcv.back_paw_body_distance import (
    back_paw_body_distance_frame_records,
    calculate_back_paw_body_distance,
)
from rbtcv.dataset import TrialVideo
from rbtcv.detection import DLCFramePrediction, DLCPoint, DLCTracking
from rbtcv.ticks import BeamCalibration, BeamTick, local_beam_segment_for_point


def horizontal_calibration() -> BeamCalibration:
    """A horizontal beam with a local scale of two image pixels per cm."""
    return BeamCalibration(
        key="back-paw-body-horizontal",
        dataset="Synthetic",
        day="BL",
        cage="1",
        subject="1",
        source_video="synthetic.avi",
        source_trial=1,
        frame_numbers=(0,),
        ticks=tuple(BeamTick(distance, 100 + (2 * distance), 100) for distance in range(0, 121, 10)),
        confirmed_at="test",
    )


def calibration_with_ticks(*ticks: BeamTick) -> BeamCalibration:
    return BeamCalibration(
        key="back-paw-body-custom",
        dataset="Synthetic",
        day="BL",
        cage="1",
        subject="1",
        source_video="synthetic.avi",
        source_trial=1,
        frame_numbers=(0,),
        ticks=ticks,
        confirmed_at="test",
    )


def body_distance_prediction(
    *,
    frame: int = 0,
    back: tuple[float, float] = (160.0, 100.0),
    body: tuple[float, float] = (166.0, 92.0),
    back_likelihood: float = 0.99,
    body_likelihood: float = 0.99,
    include_back: bool = True,
    include_body: bool = True,
) -> DLCFramePrediction:
    """Build a frame deliberately containing no tail or front-paw points."""
    points: list[DLCPoint] = []
    if include_back:
        points.append(DLCPoint("back_paw", "paw", *back, back_likelihood))
    if include_body:
        points.append(DLCPoint("body_center", "body", *body, body_likelihood))
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


class BackPawBodyDistanceTests(unittest.TestCase):
    def test_uses_local_calibrated_scale_for_euclidean_distance(self) -> None:
        # Back paw is at 30 cm. The vector to the body centre is (6, -8) px:
        # 10 px / (2 px per cm) = 5 cm.
        result = calculate_back_paw_body_distance(
            body_distance_prediction(), horizontal_calibration()
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.back_paw_position_cm, 30.0, places=6)
        self.assertAlmostEqual(result.distance_cm, 5.0, places=6)

    def test_uses_nearest_local_segment_at_the_landmark_midpoint(self) -> None:
        # This 0 -> 90 cm beam runs leftward and downward. Its 30--40 cm
        # segment has a deliberately different local scale from its neighbours.
        calibration = calibration_with_ticks(
            BeamTick(0, 300, 100),
            BeamTick(10, 280, 110),
            BeamTick(20, 260, 120),
            BeamTick(30, 240, 130),
            BeamTick(40, 200, 150),
            BeamTick(50, 180, 160),
            BeamTick(60, 160, 170),
            BeamTick(70, 140, 180),
            BeamTick(80, 120, 190),
            BeamTick(90, 100, 200),
            BeamTick(100, 80, 210),
        )
        # Back paw projects to 35 cm on the tilted 30--40 segment. Body is
        # above its fall line; their midpoint stays nearest the same segment.
        tracked = body_distance_prediction(back=(220.0, 140.0), body=(226.0, 132.0))
        local = local_beam_segment_for_point(calibration, 223.0, 136.0)
        result = calculate_back_paw_body_distance(tracked, calibration)

        self.assertIsNotNone(local)
        assert local is not None
        self.assertEqual((local.start.distance_cm, local.end.distance_cm), (30, 40))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.back_paw_position_cm, 35.0, places=6)
        self.assertAlmostEqual(result.distance_cm, math.sqrt(5.0), places=6)

    def test_back_paw_0_to_90_cm_gate_is_inclusive(self) -> None:
        beam = horizontal_calibration()

        at_zero = calculate_back_paw_body_distance(
            body_distance_prediction(back=(100.0, 100.0), body=(104.0, 90.0)), beam
        )
        at_ninety = calculate_back_paw_body_distance(
            body_distance_prediction(back=(280.0, 100.0), body=(284.0, 90.0)), beam
        )
        before_zero = calculate_back_paw_body_distance(
            body_distance_prediction(back=(99.0, 100.0), body=(103.0, 90.0)), beam
        )
        after_ninety = calculate_back_paw_body_distance(
            body_distance_prediction(back=(281.0, 100.0), body=(285.0, 90.0)), beam
        )

        self.assertIsNotNone(at_zero)
        self.assertIsNotNone(at_ninety)
        self.assertIsNone(before_zero)
        self.assertIsNone(after_ninety)

    def test_body_center_must_be_strictly_above_the_fall_boundary(self) -> None:
        beam = horizontal_calibration()
        self.assertIsNotNone(
            calculate_back_paw_body_distance(body_distance_prediction(body=(166.0, 99.9)), beam)
        )
        self.assertIsNone(
            calculate_back_paw_body_distance(body_distance_prediction(body=(166.0, 100.0)), beam)
        )
        self.assertIsNone(
            calculate_back_paw_body_distance(body_distance_prediction(body=(166.0, 100.1)), beam)
        )

    def test_rejects_low_confidence_missing_and_nonfinite_required_landmarks(self) -> None:
        beam = horizontal_calibration()
        cases = {
            "low-confidence back paw": body_distance_prediction(back_likelihood=0.59),
            "low-confidence body center": body_distance_prediction(body_likelihood=0.59),
            "missing back paw": body_distance_prediction(include_back=False),
            "missing body center": body_distance_prediction(include_body=False),
            "NaN back-paw x": body_distance_prediction(back=(math.nan, 100.0)),
            "infinite body-center y": body_distance_prediction(body=(166.0, math.inf)),
            "NaN body-center likelihood": body_distance_prediction(body_likelihood=math.nan),
        }

        for description, tracked in cases.items():
            with self.subTest(description=description):
                self.assertIsNone(calculate_back_paw_body_distance(tracked, beam))

    def test_rejects_missing_or_degenerate_calibration(self) -> None:
        tracked = body_distance_prediction()
        one_tick = calibration_with_ticks(BeamTick(0, 100, 100))
        duplicate_ticks = calibration_with_ticks(
            BeamTick(0, 100, 100),
            BeamTick(10, 100, 100),
        )

        self.assertIsNone(calculate_back_paw_body_distance(tracked, None))
        self.assertIsNone(calculate_back_paw_body_distance(tracked, one_tick))
        self.assertIsNone(calculate_back_paw_body_distance(tracked, duplicate_ticks))

    def test_frame_records_are_sorted_and_use_the_supplied_fps_rate(self) -> None:
        tracking = DLCTracking(
            Path("synthetic.csv"),
            {
                6: body_distance_prediction(frame=6),
                0: body_distance_prediction(frame=0),
                3: body_distance_prediction(frame=3),
            },
        )

        records = back_paw_body_distance_frame_records(
            trial_video(), tracking, horizontal_calibration(), fps=3.0
        )

        self.assertEqual([record.frame for record in records], [0, 3, 6])
        self.assertEqual([record.time_seconds for record in records], [0.0, 1.0, 2.0])
        self.assertEqual([record.back_paw_position_cm for record in records], [30.0, 30.0, 30.0])
        self.assertEqual([record.back_paw_body_distance_cm for record in records], [5.0, 5.0, 5.0])
        self.assertEqual(records[0].relative_video, trial_video().relative_path)
        self.assertEqual(records[0].day, "D3")
        self.assertEqual(records[0].cage, "7")
        self.assertEqual(records[0].animal, "3")
        self.assertEqual(records[0].trial, 2)


if __name__ == "__main__":
    unittest.main()
