from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from rbtcv.app import (
    BROADER_TICK_SCAN_FRAMES,
    INITIAL_TICK_SCAN_FRAMES,
    RBTReviewApp,
    next_guided_tick_distance,
    next_missing_guided_tick_distance,
    tick_confirmation_requirements,
    tick_review_videos,
)
from rbtcv.dataset import TrialVideo
from rbtcv.ticks import BeamCalibration, BeamTick, TickDetectionResult


def video(cage: str, animal: str, trial: int) -> TrialVideo:
    return TrialVideo(
        dataset="Dataset",
        day="D3",
        group=f"C{cage}",
        subject=animal,
        trial=trial,
        date="2026-09-01",
        clock="120000",
        path=Path(f"C{cage}_{animal}_T{trial}.avi"),
        relative_path=f"data/Dataset/D3/C{cage}_{animal}_T{trial}.avi",
    )


class TickConfirmationRequirementTests(unittest.TestCase):
    def test_missing_day_calibrations_are_reported_as_shared_t1_actions(self) -> None:
        requirements = tick_confirmation_requirements(
            [video("6", "2", trial) for trial in (1, 2, 3)]
            + [video("1", "1", trial) for trial in (1, 2, 3)],
            {},
        )

        self.assertEqual(
            requirements,
            [
                "Cage 1 Mouse 1: T1 (covers T1-T3)",
                "Cage 6 Mouse 2: T1 (covers T1-T3)",
            ],
        )

    def test_failed_initial_tick_scan_retries_with_broader_sample(self) -> None:
        scans: list[int] = []
        completed: list[tuple[bool, str]] = []
        status_messages: list[str] = []
        detections = iter(
            (
                TickDetectionResult((), (), "Initial scan had overlapping ticks."),
                TickDetectionResult((), (), "Broader scan had overlapping ticks."),
            )
        )
        app = SimpleNamespace(
            _runtime=lambda: (Path("python"), Path("runner"), Path("tracking"), Path("ticks")),
            dlc_tick_detector=SimpleNamespace(detect_for_video=lambda _video: next(detections)),
            _save_calibration=lambda _video, _detection: False,
            _save_tick_draft=lambda _video, _detection: False,
            tick_status_var=SimpleNamespace(set=status_messages.append),
        )

        def run_dlc(args, done, _message) -> None:
            scans.append(int(args[args.index("--early-frames") + 1]))
            done(0, "")

        app._run_dlc = run_dlc
        RBTReviewApp._detect_ticks_with_fallback(
            app,
            video("1", "1", 1),
            lambda succeeded, message: completed.append((succeeded, message)),
            "Detecting trial ticks",
        )

        self.assertEqual(scans, [INITIAL_TICK_SCAN_FRAMES, BROADER_TICK_SCAN_FRAMES])
        self.assertEqual(completed, [(False, "Broader scan had overlapping ticks.")])
        self.assertIn("retrying", status_messages[0])

    def test_guided_manual_ticks_advance_in_ten_centimeter_steps(self) -> None:
        self.assertEqual(next_guided_tick_distance(0), 10)
        self.assertEqual(next_guided_tick_distance(110), 120)
        self.assertIsNone(next_guided_tick_distance(120))

    def test_partial_tick_draft_skips_ticks_that_were_retained(self) -> None:
        calibration = BeamCalibration(
            key="Dataset|D3|1_1",
            dataset="Dataset",
            day="D3",
            cage="1",
            subject="1",
            source_video="video.avi",
            source_trial=1,
            frame_numbers=(),
            ticks=(BeamTick(0, 0, 0), BeamTick(20, 20, 0)),
            confirmed_at="",
        )

        self.assertEqual(next_missing_guided_tick_distance(calibration, 0), 10)
        self.assertEqual(next_missing_guided_tick_distance(calibration, 10), 30)

    def test_tick_checker_orders_trials_before_moving_to_next_subject(self) -> None:
        first_subject = "Dataset|D3|1_1"
        second_subject = "Dataset|D3|2_1"
        fake_dataset = SimpleNamespace(
            subjects_for_day=lambda _day: [(first_subject, "Cage 1"), (second_subject, "Cage 2")],
            trials_for_subject=lambda key: (
                {3: video("1", "1", 3), 1: video("1", "1", 1), 2: video("1", "1", 2)}
                if key == first_subject
                else {2: video("2", "1", 2), 1: video("2", "1", 1)}
            ),
        )

        ordered = tick_review_videos(fake_dataset, "D3")

        self.assertEqual(
            [(item.cage_number, item.rat_id, item.trial) for item in ordered],
            [("1", "1", 1), ("1", "1", 2), ("1", "1", 3), ("2", "1", 1), ("2", "1", 2)],
        )
