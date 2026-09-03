from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from rbtcv.detection import (
    DISPLAY_LIKELIHOOD_CUTOFF,
    SCORING_LIKELIHOOD_CUTOFF,
    DLCFramePrediction,
    DLCPoint,
    DLCTracking,
    DLCPredictionStore,
)
from rbtcv.scoring import FALL_MAX_SECONDS, PawMark, raw_crossing_time_seconds, scored_crossing_time_seconds
from rbtcv.ticks import BeamCalibration, BeamTick
from rbtcv.tracking_rules import FELL, REACHED, analyze_tracking_timeline


def calibration() -> BeamCalibration:
    return BeamCalibration(
        key="synthetic",
        dataset="synthetic",
        day="D0",
        cage="1",
        subject="1",
        source_video="synthetic.avi",
        source_trial=1,
        frame_numbers=(0,),
        ticks=(
            BeamTick(0, 100, 50),
            BeamTick(60, 200, 50),
            BeamTick(120, 300, 50),
        ),
        confirmed_at="test",
    )


def prediction(frame_number: int, back_paw_x: float, body_center_y: float) -> DLCFramePrediction:
    return DLCFramePrediction(
        frame=frame_number,
        points=(
            DLCPoint("visible_back_paw", "paw", back_paw_x, 50.0, 0.99),
            DLCPoint("body_center", "body", back_paw_x, body_center_y, 0.99),
        ),
    )


class DLCPredictionStoreTests(unittest.TestCase):
    def test_parser_keeps_expected_points_at_display_and_scoring_cutoffs(self) -> None:
        rows = [
            ["scorer", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC"],
            ["bodyparts", "visible_back_paw", "visible_back_paw", "visible_back_paw", "visible_front_paw", "visible_front_paw", "visible_front_paw", "tail_end", "tail_end", "tail_end", "body_center", "body_center", "body_center"],
            ["coords", "x", "y", "likelihood", "x", "y", "likelihood", "x", "y", "likelihood", "x", "y", "likelihood"],
            ["0", "10", "20", "0.95", "30", "40", "0.20", "50", "60", "0.90", "70", "80", "0.99"],
        ]
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sample_dlc.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            display = DLCPredictionStore(likelihood_cutoff=DISPLAY_LIKELIHOOD_CUTOFF).load(csv_path)
            scoring = display.filtered(SCORING_LIKELIHOOD_CUTOFF)

        display_frame = display.points_for_frame(0)
        scoring_frame = scoring.points_for_frame(0)

        self.assertIsNotNone(display_frame)
        self.assertEqual(display_frame.paw_count, 2)
        self.assertIsNotNone(scoring_frame)
        self.assertEqual(scoring_frame.paw_count, 1)
        self.assertEqual(scoring_frame.tail_count, 1)
        self.assertEqual(len(scoring_frame.body_points), 1)

    def test_parser_recognizes_six_landmark_model_tail_points(self) -> None:
        bodyparts = ["back_paw", "front_paw", "tail_S", "tail_M", "tail_E", "body_center"]
        rows = [
            ["scorer", *("DLC" for _ in range(18))],
            ["bodyparts", *(part for part in bodyparts for _ in range(3))],
            ["coords", *(coord for _ in bodyparts for coord in ("x", "y", "likelihood"))],
            ["0", *(value for index in range(6) for value in (str(index * 10), str(index * 10 + 1), "0.99"))],
        ]
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "six_point_dlc.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)
            frame = DLCPredictionStore().load(csv_path).points_for_frame(0)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.paw_count, 2)
        self.assertEqual(frame.tail_count, 3)
        self.assertIsNotNone(frame.tail_start)
        self.assertIsNotNone(frame.tail_middle)
        self.assertIsNotNone(frame.tail_end)


class TrackingRulesTests(unittest.TestCase):
    def tracking(self, frames: dict[int, DLCFramePrediction]) -> DLCTracking:
        return DLCTracking(Path("synthetic.csv"), frames)

    def test_completed_trial_starts_at_zero_and_ends_at_120(self) -> None:
        timeline = analyze_tracking_timeline(
            self.tracking(
                {
                    0: prediction(0, 90.0, 40.0),
                    1: prediction(1, 100.0, 40.0),
                    2: prediction(2, 200.0, 40.0),
                    3: prediction(3, 300.0, 40.0),
                }
            ),
            calibration(),
        )

        self.assertEqual(timeline.final_state, REACHED)
        self.assertEqual(timeline.start_frame, 1)
        self.assertEqual(timeline.end_frame, 3)
        self.assertEqual(timeline.farthest_distance_cm, 120)

    def test_first_reliable_back_paw_already_past_zero_starts_timing(self) -> None:
        """A missed exact crossing cannot leave a trial permanently waiting."""
        timeline = analyze_tracking_timeline(
            self.tracking(
                {
                    # The first reliable back-paw observation is already beyond
                    # the 0 cm tick, as happens after an occluded crossing.
                    0: prediction(0, 108.0, 40.0),
                    1: prediction(1, 200.0, 40.0),
                    2: prediction(2, 300.0, 40.0),
                }
            ),
            calibration(),
        )

        self.assertEqual(timeline.start_frame, 0)
        self.assertEqual(timeline.end_frame, 2)
        self.assertEqual(timeline.final_state, REACHED)

    def test_stable_near_endpoint_finishes_a_trial(self) -> None:
        timeline = analyze_tracking_timeline(
            self.tracking(
                {
                    0: prediction(0, 90.0, 40.0),
                    1: prediction(1, 100.0, 40.0),
                    2: prediction(2, 297.5, 40.0),
                    3: prediction(3, 298.0, 40.0),
                }
            ),
            calibration(),
        )

        self.assertEqual(timeline.final_state, REACHED)
        self.assertEqual(timeline.end_frame, 3)
        self.assertFalse(timeline.ended_at_video_end)

    def test_running_trial_uses_final_video_frame_as_completion(self) -> None:
        timeline = analyze_tracking_timeline(
            self.tracking(
                {
                    0: prediction(0, 90.0, 40.0),
                    1: prediction(1, 100.0, 40.0),
                    2: prediction(2, 180.0, 40.0),
                }
            ),
            calibration(),
        )

        self.assertEqual(timeline.start_frame, 1)
        self.assertEqual(timeline.end_frame, 2)
        self.assertEqual(timeline.final_state, REACHED)
        self.assertTrue(timeline.ended_at_video_end)
        self.assertEqual(timeline.farthest_distance_cm, 30)

    def test_fall_records_distance_and_the_penalty_time(self) -> None:
        timeline = analyze_tracking_timeline(
            self.tracking(
                {
                    0: prediction(0, 90.0, 40.0),
                    1: prediction(1, 100.0, 40.0),
                    2: prediction(2, 200.0, 70.0),
                }
            ),
            calibration(),
        )

        self.assertEqual(timeline.final_state, FELL)
        self.assertEqual(timeline.start_frame, 1)
        self.assertEqual(timeline.end_frame, 2)
        self.assertEqual(timeline.farthest_distance_cm, 60)
        self.assertEqual(
            scored_crossing_time_seconds(
                FELL,
                raw_crossing_time_seconds(PawMark(1, 100, 50), PawMark(2, 200, 70), 15.0),
            ),
            FALL_MAX_SECONDS,
        )

    def test_15_pixel_margin_allows_recovery_before_declaring_a_fall(self) -> None:
        def final_state(body_center_y: float) -> str:
            return analyze_tracking_timeline(
                self.tracking(
                    {
                        0: prediction(0, 90.0, 40.0),
                        1: prediction(1, 100.0, 40.0),
                        2: prediction(2, 200.0, body_center_y),
                        3: prediction(3, 300.0, 40.0),
                    }
                ),
                calibration(),
            ).final_state

        self.assertEqual(final_state(64.0), REACHED)
        self.assertEqual(final_state(65.0), REACHED)
        self.assertEqual(final_state(66.0), FELL)


if __name__ == "__main__":
    unittest.main()
