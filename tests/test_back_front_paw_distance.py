from __future__ import annotations

import math
from pathlib import Path
import unittest

from rbtcv.back_front_paw_distance import (
    back_front_paw_distance_frame_records,
    calculate_back_front_paw_distance,
)
from rbtcv.back_front_paw_distance_plot import (
    BackFrontPawDistancePlotStore,
    TrialPawDistanceSeries,
    _zoomed_nonnegative_axis_bounds,
    summarize_back_front_paw_distance_trial_consistency,
    summarize_back_front_paw_distance_within_trial_variation,
    summarize_day_by_day_back_front_paw_distance_within_trial_variation,
    summarize_group_back_front_paw_distances,
)
from rbtcv.dataset import TrialVideo
from rbtcv.detection import DLCFramePrediction, DLCPoint, DLCTracking
from rbtcv.ticks import BeamCalibration, BeamTick, local_beam_segment_for_point


def horizontal_calibration() -> BeamCalibration:
    """A horizontal beam with a local scale of two image pixels per cm."""
    return BeamCalibration(
        key="back-front-horizontal",
        dataset="Synthetic",
        day="BL",
        cage="1",
        subject="1",
        source_video="synthetic.avi",
        source_trial=1,
        frame_numbers=(0,),
        ticks=tuple(BeamTick(distance, 100 + (2 * distance), 50) for distance in range(0, 121, 10)),
        confirmed_at="test",
    )


def calibration_with_ticks(*ticks: BeamTick) -> BeamCalibration:
    return BeamCalibration(
        key="back-front-custom",
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


def paws_prediction(
    *,
    frame: int = 0,
    back: tuple[float, float] = (160.0, 50.0),
    front: tuple[float, float] = (166.0, 58.0),
    back_likelihood: float = 0.99,
    front_likelihood: float = 0.99,
    include_back: bool = True,
    include_front: bool = True,
) -> DLCFramePrediction:
    """Make a prediction intentionally containing no tail/body landmarks."""
    points: list[DLCPoint] = []
    if include_back:
        points.append(DLCPoint("back_paw", "paw", *back, back_likelihood))
    if include_front:
        points.append(DLCPoint("front_paw", "paw", *front, front_likelihood))
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


class BackFrontPawDistanceTests(unittest.TestCase):
    def test_uses_local_calibrated_scale_for_euclidean_paw_separation(self) -> None:
        # The back paw is at 30 cm. The paw-to-paw vector is (6, 8) image
        # pixels, i.e. 10 px or 5 cm at the local 2 px/cm beam scale.
        result = calculate_back_front_paw_distance(paws_prediction(), horizontal_calibration())

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.back_paw_position_cm, 30.0, places=6)
        self.assertAlmostEqual(result.distance_cm, 5.0, places=6)

    def test_uses_nearest_local_segment_on_a_tilted_mirrored_beam(self) -> None:
        # 0 -> 90 cm runs leftward and downward. The 30--40 cm segment has a
        # different local scale from its neighbors, so selecting by image x or
        # using a global scale would give the wrong result.
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
        # Back paw is exactly 35 cm along the tilted 30--40 cm segment. Its
        # midpoint with the front paw remains closest to that same segment.
        tracked = paws_prediction(back=(220.0, 140.0), front=(226.0, 148.0))
        local = local_beam_segment_for_point(calibration, 223.0, 144.0)
        result = calculate_back_front_paw_distance(tracked, calibration)

        self.assertIsNotNone(local)
        assert local is not None
        self.assertEqual((local.start.distance_cm, local.end.distance_cm), (30, 40))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.back_paw_position_cm, 35.0, places=6)
        self.assertAlmostEqual(result.distance_cm, math.sqrt(5.0), places=6)

    def test_back_paw_0_to_90_cm_gate_is_inclusive(self) -> None:
        beam = horizontal_calibration()

        at_zero = calculate_back_front_paw_distance(
            paws_prediction(back=(100.0, 50.0), front=(104.0, 50.0)), beam
        )
        at_ninety = calculate_back_front_paw_distance(
            paws_prediction(back=(280.0, 50.0), front=(284.0, 50.0)), beam
        )
        before_zero = calculate_back_front_paw_distance(
            paws_prediction(back=(99.0, 50.0), front=(103.0, 50.0)), beam
        )
        after_ninety = calculate_back_front_paw_distance(
            paws_prediction(back=(281.0, 50.0), front=(285.0, 50.0)), beam
        )

        self.assertIsNotNone(at_zero)
        self.assertIsNotNone(at_ninety)
        self.assertIsNone(before_zero)
        self.assertIsNone(after_ninety)

    def test_rejects_low_confidence_missing_and_nonfinite_paws(self) -> None:
        beam = horizontal_calibration()
        cases = {
            "low-confidence back paw": paws_prediction(back_likelihood=0.59),
            "low-confidence front paw": paws_prediction(front_likelihood=0.59),
            "missing back paw": paws_prediction(include_back=False),
            "missing front paw": paws_prediction(include_front=False),
            "NaN back-paw x": paws_prediction(back=(math.nan, 50.0)),
            "NaN front-paw y": paws_prediction(front=(166.0, math.nan)),
        }

        for description, tracked in cases.items():
            with self.subTest(description=description):
                self.assertIsNone(calculate_back_front_paw_distance(tracked, beam))

    def test_rejects_missing_or_degenerate_calibration(self) -> None:
        tracked = paws_prediction()
        one_tick = calibration_with_ticks(BeamTick(0, 100, 50))
        duplicate_ticks = calibration_with_ticks(
            BeamTick(0, 100, 50),
            BeamTick(10, 100, 50),
        )

        self.assertIsNone(calculate_back_front_paw_distance(tracked, None))
        self.assertIsNone(calculate_back_front_paw_distance(tracked, one_tick))
        self.assertIsNone(calculate_back_front_paw_distance(tracked, duplicate_ticks))

    def test_frame_records_are_sorted_and_use_the_supplied_three_fps_rate(self) -> None:
        # The source dict is deliberately unordered. No tail/body point is
        # supplied: both paw landmarks are sufficient for this metric.
        tracking = DLCTracking(
            Path("synthetic.csv"),
            {
                6: paws_prediction(frame=6),
                0: paws_prediction(frame=0),
                3: paws_prediction(frame=3),
            },
        )

        records = back_front_paw_distance_frame_records(
            trial_video(), tracking, horizontal_calibration(), fps=3.0
        )

        self.assertEqual([record.frame for record in records], [0, 3, 6])
        self.assertEqual([record.time_seconds for record in records], [0.0, 1.0, 2.0])
        self.assertEqual([record.back_paw_position_cm for record in records], [30.0, 30.0, 30.0])
        self.assertEqual([record.back_front_paw_distance_cm for record in records], [5.0, 5.0, 5.0])
        self.assertEqual(records[0].relative_video, trial_video().relative_path)
        self.assertEqual(records[0].day, "D3")
        self.assertEqual(records[0].cage, "7")
        self.assertEqual(records[0].animal, "3")
        self.assertEqual(records[0].trial, 2)

    def test_group_mean_and_variation_weight_each_mouse_equally(self) -> None:
        """Frame count cannot give a long trial more influence than one mouse."""
        values = (
            TrialPawDistanceSeries("BL", "1", "1", "SHAM", 1, (1.0,) + (None,) * 8),
            TrialPawDistanceSeries("BL", "1", "1", "SHAM", 2, (3.0,) + (None,) * 8),
            TrialPawDistanceSeries("BL", "1", "2", "SHAM", 1, (10.0,) + (None,) * 8),
            TrialPawDistanceSeries("BL", "1", "2", "SHAM", 2, (14.0,) + (None,) * 8),
        )

        mice, groups = summarize_group_back_front_paw_distances(values)
        self.assertEqual([mouse.bin_means[0] for mouse in mice], [2.0, 12.0])
        group = groups[0]
        self.assertEqual(group.mouse_counts[0], 2)
        self.assertEqual(group.means[0], 7.0)
        self.assertAlmostEqual(group.standard_deviations[0] or 0.0, math.sqrt(50.0))

        consistency_mice, consistency_groups = summarize_back_front_paw_distance_trial_consistency(values)
        self.assertAlmostEqual(consistency_mice[0].standard_deviations[0] or 0.0, math.sqrt(2.0))
        self.assertAlmostEqual(consistency_mice[1].standard_deviations[0] or 0.0, math.sqrt(8.0))
        self.assertAlmostEqual(consistency_groups[0].means[0] or 0.0, (math.sqrt(2.0) + math.sqrt(8.0)) / 2)
        self.assertAlmostEqual(consistency_groups[0].standard_deviations[0] or 0.0, 1.0)

    def test_within_trial_variation_controls_for_back_paw_position(self) -> None:
        """Raw-frame SDs are calculated within bins, then weighted by mouse."""
        raw = lambda values: (tuple(values),) + ((),) * 8
        values = (
            # Mouse 1 has one lower- and one higher-variation run.
            TrialPawDistanceSeries("BL", "1", "1", "SHAM", 1, (2.0,) + (None,) * 8, raw((1.0, 3.0))),
            TrialPawDistanceSeries("BL", "1", "1", "SHAM", 2, (3.0,) + (None,) * 8, raw((1.0, 5.0))),
            # Mouse 2 contributes equally even though its runs have the same SD.
            TrialPawDistanceSeries("BL", "1", "2", "SHAM", 1, (4.0,) + (None,) * 8, raw((1.0, 7.0))),
            TrialPawDistanceSeries("BL", "1", "2", "SHAM", 2, (5.0,) + (None,) * 8, raw((1.0, 7.0))),
        )

        variations, mice, groups = summarize_back_front_paw_distance_within_trial_variation(values)
        self.assertEqual([item.frame_counts[0] for item in variations], [2, 2, 2, 2])
        self.assertAlmostEqual(variations[0].standard_deviations[0] or 0.0, math.sqrt(2.0))
        self.assertAlmostEqual(variations[1].standard_deviations[0] or 0.0, math.sqrt(8.0))
        expected_mouse_one = (math.sqrt(2.0) + math.sqrt(8.0)) / 2.0
        expected_mouse_two = math.sqrt(18.0)
        self.assertAlmostEqual(mice[0].bin_means[0] or 0.0, expected_mouse_one)
        self.assertAlmostEqual(mice[1].bin_means[0] or 0.0, expected_mouse_two)
        self.assertAlmostEqual(groups[0].means[0] or 0.0, (expected_mouse_one + expected_mouse_two) / 2.0)

        day_summary = summarize_day_by_day_back_front_paw_distance_within_trial_variation(values)
        self.assertEqual(day_summary[0].mouse_counts["BL"], 2)
        self.assertAlmostEqual(
            day_summary[0].means["BL"],
            (expected_mouse_one + expected_mouse_two) / 2.0,
        )
        svg = BackFrontPawDistancePlotStore._day_bar_svg(
            "Dataset",
            day_summary,
            title="Day-by-day back/front paw-distance within-trial variation",
            calculation_note="position-controlled raw-frame sample SD",
            comparison_note="Lower SD means more consistent posture and gait.",
            empty_note="empty",
            y_label="Within-trial paw-distance SD (cm; lower = more consistent)",
        )
        self.assertIn("Day-by-day back/front paw-distance within-trial variation", svg)
        self.assertIn('fill-opacity="0.24"', svg)
        self.assertIn("#6B7280", svg)
        self.assertIn("#374151", svg)

    def test_plot_axis_zooms_to_observations_and_keeps_nonnegative_sd(self) -> None:
        self.assertEqual(_zoomed_nonnegative_axis_bounds((2.14, 6.45)), (2.0, 7.0))
        self.assertEqual(_zoomed_nonnegative_axis_bounds((-0.03, 1.62)), (0.0, 2.0))


if __name__ == "__main__":
    unittest.main()
