from __future__ import annotations

import unittest
from pathlib import Path

from rbtcv.dataset import TrialVideo
from rbtcv.detection import DLCFramePrediction, DLCPoint, DLCTracking
from rbtcv.tick_intervals import (
    ordered_tick_crossing_frames,
    project_back_paw_distance_cm,
    tick_interval_records,
)
from rbtcv.tick_interval_plot import (
    TickIntervalPlotStore,
    TrialIntervalSeries,
    summarize_day_by_day_interval_within_trial_variation,
    summarize_interval_within_trial_variation,
)
from rbtcv.ticks import BeamCalibration, BeamTick


def horizontal_calibration() -> BeamCalibration:
    """A 0--90 cm beam with one 10 cm tick every 20 image pixels."""
    return BeamCalibration(
        key="synthetic",
        dataset="Synthetic",
        day="BL",
        cage="1",
        subject="1",
        source_video="synthetic.avi",
        source_trial=1,
        frame_numbers=(0,),
        ticks=tuple(BeamTick(distance, 100 + (2 * distance), 50) for distance in range(0, 91, 10)),
        confirmed_at="test",
    )


def trial_video() -> TrialVideo:
    return TrialVideo(
        dataset="Synthetic",
        day="BL",
        group="C1",
        subject="1",
        trial=1,
        date="2026-01-01",
        clock="120000",
        path=Path("synthetic.avi"),
        relative_path="data/Synthetic/BL/C1_1_T1_2026-01-01-120000-0000.avi",
    )


def prediction(
    frame: int,
    distance_cm: float,
    *,
    body_y: float = 40.0,
    paw_likelihood: float = 0.99,
    body_likelihood: float = 0.99,
) -> DLCFramePrediction:
    # The horizontal fixture maps 1 cm to two x pixels.
    x = 100.0 + (2.0 * distance_cm)
    return DLCFramePrediction(
        frame=frame,
        points=(
            DLCPoint("back_paw", "paw", x, 50.0, paw_likelihood),
            DLCPoint("body_center", "body", x, body_y, body_likelihood),
        ),
    )


def tracking(frames: dict[int, DLCFramePrediction]) -> DLCTracking:
    return DLCTracking(Path("synthetic.csv"), frames)


class TickIntervalTimingTests(unittest.TestCase):
    def test_orders_all_0_to_90_crossings_and_exports_adjacent_intervals(self) -> None:
        frames = {0: prediction(0, -2.0)}
        # Each point is just past its own tick. The first reliable 0 cm point is
        # frame 1, then every interval has one frame of elapsed video time.
        frames.update(
            {
                frame: prediction(frame, float((frame - 1) * 10) + 0.5)
                for frame in range(1, 11)
            }
        )

        crossings = ordered_tick_crossing_frames(tracking(frames), horizontal_calibration())
        self.assertEqual(crossings, {distance: (distance // 10) + 1 for distance in range(0, 91, 10)})

        records = tick_interval_records(trial_video(), tracking(frames), horizontal_calibration(), fps=20.0)
        self.assertEqual([(record.interval_start_cm, record.interval_end_cm) for record in records], [
            (0, 10),
            (10, 20),
            (20, 30),
            (30, 40),
            (40, 50),
            (50, 60),
            (60, 70),
            (70, 80),
            (80, 90),
        ])
        self.assertTrue(all(record.elapsed_seconds == 0.05 for record in records))

    def test_first_reliable_point_at_or_past_multiple_ticks_uses_that_frame(self) -> None:
        frames = {
            0: prediction(0, -3.0),
            # No frame was captured exactly at 0 or 10 cm. This is the first
            # reliable point at/past both crossings.
            1: prediction(1, 14.0),
            2: prediction(2, 22.0),
        }

        crossings = ordered_tick_crossing_frames(tracking(frames), horizontal_calibration())
        self.assertEqual(crossings, {0: 1, 10: 1, 20: 2})
        records = tick_interval_records(trial_video(), tracking(frames), horizontal_calibration(), fps=10.0)
        self.assertEqual([(record.interval_start_cm, record.interval_end_cm, record.elapsed_seconds) for record in records], [
            (0, 10, 0.0),
            (10, 20, 0.1),
        ])

    def test_recovery_pause_delays_next_timestamp_and_retains_wall_clock_time(self) -> None:
        frames = {
            0: prediction(0, -1.0),
            1: prediction(1, 0.0),
            2: prediction(2, 10.0),
            # At and below the fall boundary: reliable but cannot establish the
            # 20 cm crossing. These frames still count in elapsed real time.
            3: prediction(3, 20.0, body_y=50.0),
            4: prediction(4, 20.0, body_y=60.0),
            5: prediction(5, 20.0, body_y=40.0),
        }

        crossings = ordered_tick_crossing_frames(tracking(frames), horizontal_calibration())
        self.assertEqual(crossings, {0: 1, 10: 2, 20: 5})
        records = tick_interval_records(trial_video(), tracking(frames), horizontal_calibration(), fps=10.0)
        self.assertEqual([record.elapsed_seconds for record in records], [0.1, 0.3])

    def test_low_confidence_after_start_breaks_the_sequence(self) -> None:
        frames = {
            0: prediction(0, -1.0),
            1: prediction(1, 0.0),
            2: prediction(2, 10.0),
            3: prediction(3, 15.0, paw_likelihood=0.59),
            4: prediction(4, 30.0),
        }

        crossings = ordered_tick_crossing_frames(tracking(frames), horizontal_calibration())
        self.assertEqual(crossings, {0: 1, 10: 2})
        records = tick_interval_records(trial_video(), tracking(frames), horizontal_calibration(), fps=10.0)
        self.assertEqual([(record.interval_start_cm, record.interval_end_cm) for record in records], [(0, 10)])

    def test_missing_csv_frame_after_start_breaks_the_sequence(self) -> None:
        frames = {
            0: prediction(0, -1.0),
            1: prediction(1, 0.0),
            2: prediction(2, 10.0),
            # Frame 3 is absent from the saved DLC CSV; frame 4 must not bridge it.
            4: prediction(4, 30.0),
        }

        crossings = ordered_tick_crossing_frames(tracking(frames), horizontal_calibration())
        self.assertEqual(crossings, {0: 1, 10: 2})

    def test_tilted_mirrored_beam_projection_is_continuous(self) -> None:
        # The 0 -> 90 cm direction runs leftward and downward in the video.
        # Projection must follow calibrated distance order, not increasing x.
        calibration = BeamCalibration(
            key="tilted-mirror",
            dataset="Synthetic",
            day="BL",
            cage="1",
            subject="1",
            source_video="synthetic.avi",
            source_trial=1,
            frame_numbers=(0,),
            ticks=tuple(
                BeamTick(distance, int(300 - (2 * distance)), int(100 + (distance / 2)))
                for distance in range(0, 91, 10)
            ),
            confirmed_at="test",
        )

        # 35 cm on the unrounded line is (230, 117.5), half way through the
        # 30--40 cm calibrated segment.
        self.assertAlmostEqual(project_back_paw_distance_cm(calibration, 230.0, 117.5), 35.0, places=6)


class TickIntervalWithinTrialVariationTests(unittest.TestCase):
    def test_compares_raw_interval_time_sd_with_equal_mouse_weighting(self) -> None:
        trials = (
            TrialIntervalSeries("BL", "1", "1", "SHAM", 1, (1.0, 1.0, 1.0) + (None,) * 6),
            TrialIntervalSeries("BL", "1", "1", "SHAM", 2, (1.0, 3.0, 5.0) + (None,) * 6),
            TrialIntervalSeries("BL", "1", "2", "SHAM", 1, (1.0, 4.0, 7.0) + (None,) * 6),
            TrialIntervalSeries("BL", "1", "2", "SHAM", 2, (1.0, 4.0, 7.0) + (None,) * 6),
        )

        trial_values, mice, groups = summarize_interval_within_trial_variation(trials)

        self.assertEqual([item.interval_count for item in trial_values], [3, 3, 3, 3])
        self.assertEqual([item.standard_deviation_seconds for item in trial_values], [0.0, 2.0, 3.0, 3.0])
        means_by_mouse = {item.animal: item.mean_standard_deviation_seconds for item in mice}
        self.assertEqual(means_by_mouse, {"1": 1.0, "2": 3.0})
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].condition, "SHAM")
        self.assertEqual(groups[0].mean_standard_deviation_seconds, 2.0)
        self.assertAlmostEqual(groups[0].standard_deviation_seconds or 0.0, 2**0.5)

        day_summary = summarize_day_by_day_interval_within_trial_variation(trials)
        self.assertEqual(day_summary[0].means, {"BL": 2.0})
        self.assertAlmostEqual(day_summary[0].standard_deviations["BL"] or 0.0, 2**0.5)

        svg = TickIntervalPlotStore._daily_within_trial_variation_svg(
            "Dataset", "BL", trial_values, groups
        )
        self.assertIn("Each dot is one trial", svg)
        self.assertIn("Trial T1 | SD 0.000 s across 3 interval(s)", svg)
        self.assertIn('fill-opacity="0.18"', svg)

        day_svg = TickIntervalPlotStore._day_within_trial_variation_bar_svg(
            "Dataset", day_summary
        )
        self.assertIn("Day-by-day tick-interval within-trial timing variation", day_svg)
        self.assertIn("2.000 s", day_svg)
        self.assertIn("Lower SD means more even timing", day_svg)
        self.assertIn('fill-opacity="0.24"', day_svg)
        self.assertIn('#6B7280', day_svg)
        self.assertIn('#374151', day_svg)
        self.assertGreater(
            day_svg.index('stroke="#374151"'),
            day_svg.index('fill="#1976D2" cursor="help"'),
        )


if __name__ == "__main__":
    unittest.main()
