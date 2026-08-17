from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from rbtcv.annotations import TrialAnnotation
from rbtcv.dataset import TrialVideo
from rbtcv.detection import DLCFramePrediction, DLCPoint, DLCTracking
from rbtcv.results_workbook import ResultsWorkbook
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
