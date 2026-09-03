from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook, load_workbook

from rbtcv.annotations import TrialAnnotation
from rbtcv.condition_map import ConditionMapStore
from rbtcv.dataset import TrialVideo
from rbtcv.detection import DLCFramePrediction, DLCPoint, DLCTracking
from rbtcv.research_angle import TailAngleFrameRecord
from rbtcv.results_workbook import (
    ANGLE_CONSISTENCY_SHEET,
    LEGACY_ANGLE_HEADERS,
    ResultsWorkbook,
)
from rbtcv.tail_angle_consistency_plot import ConsistencySeries, TailAngleConsistencyPlotStore
from rbtcv.tail_angle_group_plot import (
    TailAngleGroupPlotStore,
    TrialAngleSeries,
    summarize_day_by_day_consistency,
    summarize_day_by_day_group_angles,
    summarize_group_angles,
)
from rbtcv.tail_position import RAW_FILENAME, TailPositionStore
from rbtcv.ticks import BeamCalibration, BeamTick


def annotation(
    *,
    cage: str,
    subject: str,
    trial: int,
    crossing_time: float,
    distance_cm: int,
    day: str = "BL",
) -> TrialAnnotation:
    return TrialAnnotation(
        relative_video=f"data/Dataset/{day}/{cage}_{subject}_T{trial}.avi",
        dataset="Dataset",
        day=day,
        group=cage,
        subject=subject,
        trial=trial,
        fps=15.0,
        start_frame=10,
        start_time=10 / 15,
        start_x=100,
        start_y=50,
        stop_frame=40,
        stop_time=40 / 15,
        stop_x=300,
        stop_y=50,
        crossing_time=crossing_time,
        outcome="reached",
        distance_cm=distance_cm,
        max_time_applied="no",
        saved_at="2026-08-15T12:00:00",
    )


class ResultsWorkbookTests(unittest.TestCase):
    def test_batch_upsert_keeps_tables_ordered_and_rebuilds_speed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ResultsWorkbook(Path(folder))
            path = store.save_many(
                (
                    annotation(cage="C2", subject="1", trial=1, crossing_time=12, distance_cm=120),
                    annotation(cage="C1", subject="2", trial=2, crossing_time=10, distance_cm=100),
                    annotation(cage="C1", subject="1", trial=1, crossing_time=8, distance_cm=120),
                )
            )["Dataset"]

            workbook = load_workbook(path)
            sheet = workbook["Forelimb"]
            self.assertEqual(
                [(sheet.cell(row=row, column=1).value, sheet.cell(row=row, column=2).value) for row in range(4, 7)],
                [(1, 1), (1, 2), (2, 1)],
            )
            self.assertEqual(sheet["A30"].value, "SPEED (cm/s)")
            self.assertEqual(sheet["C33"].value, 15.0)  # C1_1: 120 cm / 8 sec
            self.assertEqual(sheet["D34"].value, 10.0)  # C1_2: 100 cm / 10 sec

            updated = annotation(
                cage="C1",
                subject="2",
                trial=2,
                crossing_time=20,
                distance_cm=100,
            )
            store.save(updated)

            workbook = load_workbook(path)
            sheet = workbook["Forelimb"]
            self.assertEqual(sheet["D34"].value, 5.0)
            self.assertEqual(
                sum(
                    sheet.cell(row=row, column=1).value == "SPEED (cm/s)"
                    for row in range(1, sheet.max_row + 1)
                ),
                1,
            )

    def test_forelimb_days_are_chronological_left_to_right_after_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ResultsWorkbook(Path(folder))
            path = store.save_many(
                (
                    annotation(cage="C1", subject="1", trial=1, crossing_time=8, distance_cm=120, day="D8"),
                    annotation(cage="C1", subject="1", trial=1, crossing_time=10, distance_cm=100, day="BL"),
                    annotation(cage="C1", subject="1", trial=2, crossing_time=20, distance_cm=110, day="D3"),
                )
            )["Dataset"]

            sheet = load_workbook(path)["Forelimb"]
            self.assertEqual([sheet.cell(row=2, column=column).value for column in (3, 6, 9)], ["BL", "D3", "D8"])
            self.assertEqual([sheet.cell(row=3, column=column).value for column in range(3, 12)], ["T1", "T2", "T3"] * 3)
            self.assertEqual(sheet.cell(row=4, column=3).value, 10)
            self.assertEqual(sheet.cell(row=4, column=7).value, 20)
            self.assertEqual(sheet.cell(row=4, column=9).value, 8)
            self.assertEqual([sheet.cell(row=2, column=column).value for column in (28, 31, 34)], ["BL", "D3", "D8"])
            self.assertEqual(sheet.cell(row=4, column=28).value, 100)
            self.assertEqual(sheet.cell(row=4, column=32).value, 110)
            self.assertEqual(sheet.cell(row=4, column=34).value, 120)

    def test_zero_distance_fall_records_zero_speed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ResultsWorkbook(Path(folder))
            path = store.save(
                annotation(
                    cage="C1",
                    subject="1",
                    trial=1,
                    crossing_time=60,
                    distance_cm=0,
                )
            )

            workbook = load_workbook(path)
            self.assertEqual(workbook["Forelimb"]["C33"].value, 0)

    def test_speed_table_moves_below_a_long_time_table(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ResultsWorkbook(Path(folder))
            store.save_many(
                annotation(
                    cage=f"C{cage}",
                    subject="1",
                    trial=1,
                    crossing_time=10,
                    distance_cm=120,
                )
                for cage in range(1, 29)
            )

            workbook = load_workbook(store.path_for_dataset("Dataset"))
            sheet = workbook["Forelimb"]
            self.assertEqual(sheet["A34"].value, "SPEED (cm/s)")

    def test_tail_angle_sheet_is_wide_and_reanalysis_replaces_one_video(self) -> None:
        first_video = video(1)
        second_video = video(2)
        first_row = TailAngleFrameRecord(
            relative_video=first_video.relative_path,
            dataset="Dataset",
            day="D3",
            cage="1",
            animal="1",
            group="C1",
            trial=1,
            frame=20,
            time_seconds=20 / 15,
            back_paw_position_cm=24.5,
            signed_tail_angle_degrees=12.25,
        )
        second_row = TailAngleFrameRecord(
            relative_video=second_video.relative_path,
            dataset="Dataset",
            day="D3",
            cage="1",
            animal="1",
            group="C1",
            trial=2,
            frame=30,
            time_seconds=2.0,
            back_paw_position_cm=35.56789,
            signed_tail_angle_degrees=-8.56789,
        )
        replacement = TailAngleFrameRecord(
            relative_video=first_video.relative_path,
            dataset="Dataset",
            day="D3",
            cage="1",
            animal="1",
            group="C1",
            trial=1,
            frame=21,
            time_seconds=1.4,
            back_paw_position_cm=25.12349,
            signed_tail_angle_degrees=4.56789,
        )

        with tempfile.TemporaryDirectory() as folder:
            store = ResultsWorkbook(Path(folder))
            store.save_tail_angle_measurements((first_video, second_video), (first_row, second_row))
            store.save_tail_angle_measurements((first_video,), (replacement,))

            workbook = load_workbook(store.path_for_dataset("Dataset"))
            sheet = workbook["Frame Angles"]
            self.assertEqual(sheet.freeze_panes, "B2")
            self.assertEqual(sheet.auto_filter.ref, "A1:AK3")
            headers = [cell.value for cell in sheet[1]]
            self.assertEqual(headers[:6], ["Source video", "Day", "Cage", "Animal", "Group", "Trial"])
            self.assertNotIn("Dataset", headers)
            rows = list(sheet.iter_rows(min_row=2, values_only=True))
            self.assertEqual(len(rows), 2)
            by_source = {row[0]: row for row in rows}
            first_values = by_source[first_video.relative_path]
            second_values = by_source[second_video.relative_path]
            frame_21 = headers.index("Frame 21")
            frame_30 = headers.index("Frame 30")
            group_column = headers.index("Group")
            self.assertEqual(first_values[group_column], "Unassigned")
            self.assertEqual(first_values[frame_21], "+4.568 deg; 25.123 cm")
            self.assertEqual(second_values[frame_30], "-8.568 deg; 35.568 cm")

    def test_tail_angle_group_column_uses_saved_sham_stroke_label(self) -> None:
        trial_video = video(1)
        record = TailAngleFrameRecord(
            relative_video=trial_video.relative_path,
            dataset="Dataset",
            day="D3",
            cage="1",
            animal="1",
            group="C1",
            trial=1,
            frame=0,
            time_seconds=0.0,
            back_paw_position_cm=5.0,
            signed_tail_angle_degrees=12.0,
        )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            conditions = ConditionMapStore(root)
            conditions.update_many("Dataset", [("1", "1")], "SHAM")
            store = ResultsWorkbook(root)
            store.save_tail_angle_measurements((trial_video,), (record,))

            workbook = load_workbook(store.path_for_dataset("Dataset"))
            sheet = workbook["Frame Angles"]
            group_column = [cell.value for cell in sheet[1]].index("Group") + 1
            self.assertEqual(sheet.cell(row=2, column=group_column).value, "SHAM")
            workbook.close()

            conditions.update_many("Dataset", [("1", "1")], "STROKE")
            store.refresh_tail_angle_consistency("Dataset")
            workbook = load_workbook(store.path_for_dataset("Dataset"))
            self.assertEqual(workbook["Frame Angles"].cell(row=2, column=group_column).value, "STROKE")
            workbook.close()

    def test_tail_angle_sheet_migrates_existing_long_format(self) -> None:
        retained_video = video(2)
        replacement_video = video(1)
        legacy_record = TailAngleFrameRecord(
            relative_video=retained_video.relative_path,
            dataset="Dataset",
            day="D3",
            cage="1",
            animal="1",
            group="C1",
            trial=2,
            frame=30,
            time_seconds=2.0,
            back_paw_position_cm=35.5,
            signed_tail_angle_degrees=-8.5,
        )
        replacement = TailAngleFrameRecord(
            relative_video=replacement_video.relative_path,
            dataset="Dataset",
            day="D3",
            cage="1",
            animal="1",
            group="C1",
            trial=1,
            frame=21,
            time_seconds=1.4,
            back_paw_position_cm=25.0,
            signed_tail_angle_degrees=4.5,
        )

        with tempfile.TemporaryDirectory() as folder:
            store = ResultsWorkbook(Path(folder))
            path = store.path_for_dataset("Dataset")
            path.parent.mkdir(parents=True)
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Frame Angles"
            sheet.append(LEGACY_ANGLE_HEADERS)
            sheet.append(
                (
                    legacy_record.dataset,
                    legacy_record.day,
                    legacy_record.cage,
                    legacy_record.animal,
                    legacy_record.group,
                    legacy_record.trial,
                    legacy_record.frame,
                    legacy_record.time_seconds,
                    legacy_record.back_paw_position_cm,
                    legacy_record.signed_tail_angle_degrees,
                    legacy_record.relative_video,
                )
            )
            workbook.save(path)

            store.save_tail_angle_measurements((replacement_video,), (replacement,))

            migrated = load_workbook(path)["Frame Angles"]
            headers = [cell.value for cell in migrated[1]]
            rows = {row[0]: row for row in migrated.iter_rows(min_row=2, values_only=True)}
            retained = rows[retained_video.relative_path]
            self.assertEqual(retained[headers.index("Frame 30")], "-8.500 deg; 35.500 cm")

    def test_tail_angle_sheet_migrates_paired_frame_columns(self) -> None:
        retained_video = video(2)
        replacement_video = video(1)
        replacement = TailAngleFrameRecord(
            relative_video=replacement_video.relative_path,
            dataset="Dataset",
            day="D3",
            cage="1",
            animal="1",
            group="C1",
            trial=1,
            frame=21,
            time_seconds=1.4,
            back_paw_position_cm=25.0,
            signed_tail_angle_degrees=4.5,
        )

        with tempfile.TemporaryDirectory() as folder:
            store = ResultsWorkbook(Path(folder))
            path = store.path_for_dataset("Dataset")
            path.parent.mkdir(parents=True)
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Frame Angles"
            sheet.append(
                (
                    "Source video",
                    "Day",
                    "Cage",
                    "Animal",
                    "Group",
                    "Trial",
                    "Frame 30 angle (deg)",
                    "Frame 30 back paw (cm)",
                )
            )
            sheet.append((retained_video.relative_path, "D3", "1", "1", "C1", 2, -8.5, 35.5))
            workbook.save(path)

            store.save_tail_angle_measurements((replacement_video,), (replacement,))

            migrated = load_workbook(path)["Frame Angles"]
            headers = [cell.value for cell in migrated[1]]
            rows = {row[0]: row for row in migrated.iter_rows(min_row=2, values_only=True)}
            retained = rows[retained_video.relative_path]
            self.assertEqual(retained[headers.index("Frame 30")], "-8.500 deg; 35.500 cm")

    def test_tail_angle_consistency_reports_sample_sd_across_trials_per_mouse(self) -> None:
        """Each trial contributes one mean per position bin, never one mean per frame."""
        videos = (video(1), video(2), video(3))
        records = tuple(
            TailAngleFrameRecord(
                relative_video=trial_video.relative_path,
                dataset="Dataset",
                day="D3",
                cage="1",
                animal="1",
                group="C1",
                trial=trial_video.trial,
                frame=15,
                time_seconds=1.0,
                back_paw_position_cm=5.0,
                signed_tail_angle_degrees=angle,
            )
            for trial_video, angle in zip(videos, (10.0, 14.0, 16.0))
        )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ConditionMapStore(root).update_many("Dataset", [("1", "1")], "SHAM")
            store = ResultsWorkbook(root)
            store.save_tail_angle_measurements(videos, records)

            sheet = load_workbook(store.path_for_dataset("Dataset"))[ANGLE_CONSISTENCY_SHEET]
            self.assertEqual(sheet.freeze_panes, "A2")
            headers = [cell.value for cell in sheet[1]]
            values = next(sheet.iter_rows(min_row=2, values_only=True))
            self.assertEqual(values[headers.index("Condition")], "SHAM")
            self.assertEqual(values[headers.index("0-10 cm trial n")], 3)
            self.assertEqual(values[headers.index("0-10 cm SD (deg)")], 3.055)
            self.assertEqual(values[headers.index("Overall 0-90 cm trial n")], 3)
            self.assertEqual(values[headers.index("Overall 0-90 cm SD (deg)")], 3.055)
            plot = root / "Dataset Results" / "tail_angle_trial_consistency_D3.svg"
            self.assertTrue(plot.exists())
            text = plot.read_text(encoding="utf-8")
            self.assertIn("Tail-angle trial consistency", text)
            self.assertIn("Cage 1 Mouse 1", text)
            self.assertTrue(
                (root / "Dataset Results" / "tail_angle_group_comparison_D3.svg").exists()
            )
            self.assertTrue(
                (root / "Dataset Results" / "Day_tail_angle_group_comparison.svg").exists()
            )
            self.assertTrue(
                (root / "Dataset Results" / "Day_tail_angle_trial_consistency.svg").exists()
            )


def video(trial: int) -> TrialVideo:
    return TrialVideo(
        dataset="Dataset",
        day="D3",
        group="C1",
        subject="1",
        trial=trial,
        date="2026-08-15",
        clock=f"1200{trial:02d}",
        path=Path(f"video_{trial}.avi"),
        relative_path=f"data/Dataset/D3/C1_1_T{trial}.avi",
    )


class TailAngleConsistencyPlotTests(unittest.TestCase):
    def test_group_error_band_and_five_degree_axis_ticks_are_rendered(self) -> None:
        series = (
            ConsistencySeries(
                day="D3",
                cage="1",
                animal="1",
                condition="SHAM",
                standard_deviations=(10.0,) * 9,
                trial_counts=(3,) * 9,
                overall_standard_deviation=10.0,
                overall_trial_count=3,
            ),
            ConsistencySeries(
                day="D3",
                cage="1",
                animal="2",
                condition="SHAM",
                standard_deviations=(20.0,) * 9,
                trial_counts=(3,) * 9,
                overall_standard_deviation=20.0,
                overall_trial_count=3,
            ),
        )

        svg = TailAngleConsistencyPlotStore._svg("Dataset", "D3", series)

        self.assertIn('fill-opacity="0.18"', svg)
        self.assertIn("shaded bands show variation across mice", svg)
        self.assertIn("Sham | Cage 1 Mouse 1 | Trial-to-trial consistency SD", svg)
        self.assertIn('stroke-width="8" stroke-opacity="0" pointer-events="stroke"', svg)
        for tick in range(0, 36, 5):
            self.assertIn(f">{tick}</text>", svg)


class TailAngleGroupPlotTests(unittest.TestCase):
    def test_group_summary_averages_trials_within_mouse_before_group_sd(self) -> None:
        trials = (
            TrialAngleSeries("D3", "1", "1", "STROKE", 1, (10.0,) * 9),
            TrialAngleSeries("D3", "1", "1", "STROKE", 2, (20.0,) * 9),
            TrialAngleSeries("D3", "1", "2", "STROKE", 1, (30.0,) * 9),
            TrialAngleSeries("D3", "1", "2", "STROKE", 2, (50.0,) * 9),
        )

        mice, groups = summarize_group_angles(trials)

        self.assertEqual([mouse.bin_means[0] for mouse in mice], [15.0, 40.0])
        summary = groups[0]
        self.assertEqual(summary.means[0], 27.5)
        self.assertAlmostEqual(summary.standard_deviations[0], 17.6776695297)
        self.assertEqual(summary.mouse_counts[0], 2)

        svg = TailAngleGroupPlotStore._svg("Dataset", "D3", trials, mice, groups)
        self.assertIn('fill-opacity="0.18"', svg)
        self.assertIn("Each mouse is averaged across its trials", svg)
        self.assertIn("Stroke | Cage 1 Mouse 1 | Trial T1", svg)
        self.assertIn('stroke-width="8" stroke-opacity="0" pointer-events="stroke"', svg)

    def test_day_by_day_summary_compares_mouse_mean_signed_angles(self) -> None:
        trials = (
            TrialAngleSeries("BL", "1", "1", "SHAM", 1, (10.0,) * 9),
            TrialAngleSeries("BL", "1", "1", "SHAM", 2, (20.0,) * 9),
            TrialAngleSeries("BL", "1", "2", "SHAM", 1, (30.0,) * 9),
            TrialAngleSeries("BL", "1", "2", "SHAM", 2, (50.0,) * 9),
            TrialAngleSeries("D3", "1", "1", "SHAM", 1, (30.0,) * 9),
            TrialAngleSeries("D3", "1", "1", "SHAM", 2, (40.0,) * 9),
            TrialAngleSeries("D3", "1", "2", "SHAM", 1, (50.0,) * 9),
            TrialAngleSeries("D3", "1", "2", "SHAM", 2, (70.0,) * 9),
            TrialAngleSeries("BL", "2", "1", "STROKE", 1, (-5.0,) * 9),
            TrialAngleSeries("BL", "2", "1", "STROKE", 2, (-1.0,) * 9),
            TrialAngleSeries("D3", "2", "1", "STROKE", 1, (5.0,) * 9),
            TrialAngleSeries("D3", "2", "1", "STROKE", 2, (7.0,) * 9),
        )

        summaries = {item.condition: item for item in summarize_day_by_day_group_angles(trials)}

        sham = summaries["SHAM"]
        self.assertAlmostEqual(sham.means["BL"], 27.5)
        self.assertAlmostEqual(sham.means["D3"], 47.5)
        self.assertAlmostEqual(sham.standard_deviations["BL"], 17.6776695297)
        self.assertEqual(sham.mouse_counts, {"BL": 2, "D3": 2})
        self.assertIsNone(summaries["STROKE"].standard_deviations["BL"])

        svg = TailAngleGroupPlotStore._day_by_day_svg("Dataset", summaries.values())
        self.assertIn("Day-by-day signed tail-angle comparison", svg)
        self.assertIn("Experimental day", svg)
        self.assertIn("Signed tail angle", svg)
        self.assertIn('fill-opacity="0.18"', svg)

    def test_day_by_day_consistency_summarizes_within_mouse_trial_sd(self) -> None:
        trials = (
            TrialAngleSeries("BL", "1", "1", "SHAM", 1, (10.0,) * 9),
            TrialAngleSeries("BL", "1", "1", "SHAM", 2, (20.0,) * 9),
            TrialAngleSeries("BL", "1", "2", "SHAM", 1, (30.0,) * 9),
            TrialAngleSeries("BL", "1", "2", "SHAM", 2, (50.0,) * 9),
            TrialAngleSeries("D3", "1", "1", "SHAM", 1, (30.0,) * 9),
            TrialAngleSeries("D3", "1", "1", "SHAM", 2, (40.0,) * 9),
            TrialAngleSeries("D3", "1", "2", "SHAM", 1, (50.0,) * 9),
            TrialAngleSeries("D3", "1", "2", "SHAM", 2, (70.0,) * 9),
            TrialAngleSeries("BL", "2", "1", "STROKE", 1, (-5.0,) * 9),
            TrialAngleSeries("BL", "2", "1", "STROKE", 2, (-1.0,) * 9),
        )

        summaries = {item.condition: item for item in summarize_day_by_day_consistency(trials)}

        sham = summaries["SHAM"]
        self.assertAlmostEqual(sham.means["BL"], 10.6066017178)
        self.assertAlmostEqual(sham.means["D3"], 10.6066017178)
        self.assertAlmostEqual(sham.standard_deviations["BL"], 5.0)
        self.assertEqual(sham.mouse_counts, {"BL": 2, "D3": 2})
        self.assertIsNone(summaries["STROKE"].standard_deviations["BL"])

        svg = TailAngleGroupPlotStore._day_by_day_consistency_svg("Dataset", summaries.values())
        self.assertIn("Day-by-day tail-angle trial consistency", svg)
        self.assertIn("Lower values mean more consistent trials", svg)
        self.assertIn("Trial-to-trial tail-angle SD", svg)
        self.assertIn('fill-opacity="0.18"', svg)


def calibration() -> BeamCalibration:
    return BeamCalibration(
        key="Dataset|D3|1_1",
        dataset="Dataset",
        day="D3",
        cage="1",
        subject="1",
        source_video="source.avi",
        source_trial=1,
        frame_numbers=(0,),
        ticks=(BeamTick(0, 100, 50), BeamTick(120, 300, 50)),
        confirmed_at="test",
    )


def tracking(tail_y: float) -> DLCTracking:
    points = {
        frame: DLCFramePrediction(
            frame=frame,
            points=(DLCPoint("tail_end", "tail", 120 + frame, tail_y, 0.99),),
        )
        for frame in range(3)
    }
    return DLCTracking(csv_path=Path("tracking.csv"), frames=points)


class TailPositionStoreTests(unittest.TestCase):
    def test_batch_record_replaces_each_trial_and_refreshes_labels(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = TailPositionStore(Path(folder))
            records = (
                (video(1), tracking(40), calibration(), 0, 2),
                (video(2), tracking(45), calibration(), 0, 2),
            )
            plots = store.record_trials(records, refresh_plots=True)

            plot = plots[("Dataset", "D3")]
            self.assertIsNotNone(plot)
            self.assertIn("C1_1 (2/3 trials)", plot.read_text(encoding="utf-8"))

            csv_path = store.result_dir("Dataset") / RAW_FILENAME
            with csv_path.open(newline="", encoding="utf-8") as handle:
                first_rows = list(csv.DictReader(handle))
            self.assertEqual(len(first_rows), 6)

            store.record_trials(
                ((video(1), tracking(35), calibration(), 0, 2),),
                refresh_plots=True,
            )
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            first_trial_offsets = {
                row["tail_offset_px"] for row in rows if row["trial"] == "1"
            }
            self.assertEqual(first_trial_offsets, {"15.00000"})

    def test_reanalysis_without_tail_points_removes_stale_plot(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = TailPositionStore(Path(folder))
            initial = store.record_trials(
                ((video(1), tracking(40), calibration(), 0, 2),),
                refresh_plots=True,
            )
            plot_path = initial[("Dataset", "D3")]
            self.assertIsNotNone(plot_path)
            self.assertTrue(plot_path.exists())

            empty_tracking = DLCTracking(Path("empty.csv"), {})
            refreshed = store.record_trials(
                ((video(1), empty_tracking, calibration(), 0, 2),),
                refresh_plots=True,
            )
            self.assertIsNone(refreshed[("Dataset", "D3")])
            self.assertFalse(plot_path.exists())

            csv_path = store.result_dir("Dataset") / RAW_FILENAME
            with csv_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()
