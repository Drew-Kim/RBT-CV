from __future__ import annotations

import unittest
from pathlib import Path

from rbtcv.dataset import DatasetIndex, TrialVideo


def video(day: str, cage: str, rat: str) -> TrialVideo:
    return TrialVideo(
        dataset="Dataset",
        day=day,
        group=f"C{cage}",
        subject=rat,
        trial=1,
        date="2026-08-30",
        clock="120000",
        path=Path(f"{day}_{cage}_{rat}.avi"),
        relative_path=f"data/Dataset/{day}/C{cage}_{rat}_T1.avi",
    )


class DatasetEligibilityTests(unittest.TestCase):
    def test_only_subjects_with_both_bl_and_d30_remain_eligible(self) -> None:
        videos = [
            video("BL", "1", "1"),
            video("D3", "1", "1"),
            video("D30", "1", "1"),
            video("BL", "1", "2"),
            video("D30", "1", "3"),
            video("BL", "1", "4"),
            video("D3", "1", "4"),
        ]
        eligible = DatasetIndex._find_baseline_and_d30_subjects(videos)
        self.assertEqual(eligible, {("Dataset", "1", "1")})

        index = object.__new__(DatasetIndex)
        index.evaluated_subject_keys = eligible
        filtered = index._filter_evaluated_videos(videos)
        self.assertEqual([(item.day, item.rat_id) for item in filtered], [("BL", "1"), ("D3", "1"), ("D30", "1")])

