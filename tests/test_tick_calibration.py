from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from rbtcv.dataset import TrialVideo
from rbtcv.ticks import DLCTickDetector, calibration_key
from rbtcv.scoring import BEAM_TICK_MARKS_CM


def video(trial: int) -> TrialVideo:
    return TrialVideo(
        dataset="Dataset",
        day="D3",
        group="C1",
        subject="2",
        trial=trial,
        date="2026-08-15",
        clock="120000",
        path=Path(f"C1_2_T{trial}.avi"),
        relative_path=f"data/Dataset/D3/C1_2_T{trial}.avi",
    )


def csv_rows() -> list[list[str]]:
    bodyparts = ["bodyparts"]
    coordinates = ["coords"]
    scorer = ["scorer"]
    for distance in BEAM_TICK_MARKS_CM:
        bodyparts.extend((f"tick_{distance}",) * 3)
        coordinates.extend(("x", "y", "likelihood"))
        scorer.extend(("DLC",) * 3)

    rows = [scorer, bodyparts, coordinates]
    for frame in range(10):
        valid = frame in {2, 7}
        offset = -1 if frame == 2 else 1
        row = [str(frame)]
        for distance in BEAM_TICK_MARKS_CM:
            x = 100 + (2 * distance) + offset
            if not valid and distance == 70:
                x = 100 + (2 * 60)  # overlapping neighbor: reject this frame
            row.extend((str(x), "50", "0.95"))
        rows.append(row)
    return rows


class TickCalibrationTests(unittest.TestCase):
    def test_detector_uses_available_clear_frames_and_median_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ticks.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(csv_rows())

            result = DLCTickDetector().detect_from_csv(path)

        self.assertEqual(len(result.ticks), len(BEAM_TICK_MARKS_CM))
        self.assertEqual(result.frame_numbers, (2, 7))
        self.assertEqual(result.ticks[0].x, 100)
        self.assertEqual(result.ticks[7].x, 240)
        self.assertIn("2 clear non-overlapping", result.message)

    def test_detector_keeps_only_non_overlapping_ticks_as_a_draft(self) -> None:
        rows = csv_rows()
        tick_60_x_column = 1 + (BEAM_TICK_MARKS_CM.index(60) * 3)
        tick_70_x_column = 1 + (BEAM_TICK_MARKS_CM.index(70) * 3)
        for row in rows[3:]:
            row[tick_70_x_column] = row[tick_60_x_column]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "overlapping_ticks.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows(rows)

            result = DLCTickDetector().detect_from_csv(path)

        distances = {tick.distance_cm for tick in result.ticks}
        self.assertEqual(len(result.ticks), len(BEAM_TICK_MARKS_CM) - 2)
        self.assertNotIn(60, distances)
        self.assertNotIn(70, distances)
        self.assertIn("unconfirmed draft", result.message)

    def test_t2_and_t3_can_use_trial_specific_calibrations(self) -> None:
        self.assertEqual(
            calibration_key(video(1)),
            "Dataset|D3|1_2",
        )
        self.assertEqual(
            calibration_key(video(2), trial_specific=True),
            "Dataset|D3|1_2|T2",
        )


if __name__ == "__main__":
    unittest.main()
