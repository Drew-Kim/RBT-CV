"""Complete, resumable evaluation of SEONG RBT DATA."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbtcv.annotations import AnnotationStore, TrialAnnotation, now_stamp
from rbtcv.dataset import DatasetIndex, TrialVideo
from rbtcv.detection import DLCPredictionStore, SCORING_LIKELIHOOD_CUTOFF
from rbtcv.results_workbook import ResultsWorkbook
from rbtcv.scoring import (BEAM_LENGTH_CM, OUTCOME_FELL, OUTCOME_REACHED, PawMark,
    max_time_applied, raw_crossing_time_seconds, scored_crossing_time_seconds)
from rbtcv.tail_position import TailPositionStore
from rbtcv.ticks import (DEFAULT_TICK_DLC_OUTPUT_DIR, DLCTickDetector,
    TickCalibrationStore, calibration_from_detection, calibration_key, point_for_distance)
from rbtcv.tracking_rules import FELL, REACHED, analyze_tracking_timeline

import dlc_tracking as workflow

DATASET = "SEONG RBT DATA"
RESULT_DIR = ROOT / "outputs" / f"{DATASET} Results"
STATUS_PATH = RESULT_DIR / "evaluation_status.json"
MANIFEST_PATH = RESULT_DIR / "evaluation_manifest.csv"
TICK_FAILURES = RESULT_DIR / "tick_calibration_failures.csv"
REVIEW_FAILURES = RESULT_DIR / "scoring_review_needed.csv"
TRACKING_CONFIG = ROOT / "models" / "dlc_tracking" / "RBT_visible_front_back_tail-RBT_CV-2026-07-14" / "config.yaml"
TICK_CONFIG = ROOT / "models" / "dlc_tickmarks" / "RBT_tick_landmarks-RBT_CV-2026-06-07" / "config.yaml"
TRACKING_OUTPUT = ROOT / "outputs" / "dlc_predictions"
BATCH_SIZE = 25


def selected_videos() -> list[TrialVideo]:
    """The GUI rule: where a trial is duplicated, use the later timestamp."""
    index = DatasetIndex(ROOT / "data" / DATASET)
    chosen: dict[tuple[str, str, str, int], TrialVideo] = {}
    for video in index.videos:
        chosen[video.day, video.cage_number, video.rat_id, video.trial] = video
    return sorted(chosen.values(), key=lambda video: video.sort_key)


def tick_sources(videos: list[TrialVideo]) -> list[TrialVideo]:
    return [video for video in videos if video.trial == 1]


def chunks(items: list[TrialVideo]):
    for start in range(0, len(items), BATCH_SIZE):
        yield items[start:start + BATCH_SIZE]


def default_status(videos: list[TrialVideo]) -> dict:
    return {"dataset": DATASET, "selected_videos": [v.relative_path for v in videos],
            "ticks_completed": [], "tracking_completed": [], "saved_results": [],
            "tick_failures": [], "scoring_review": []}


def load_status(videos: list[TrialVideo], restart: bool) -> dict:
    if not restart and STATUS_PATH.exists():
        try:
            status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            if status.get("selected_videos") == [v.relative_path for v in videos]:
                return status
        except (OSError, json.JSONDecodeError):
            pass
    return default_status(videos)


def save_status(status: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(videos: list[TrialVideo]) -> None:
    rows = [{"day": v.day, "cage": v.cage_number, "animal": v.rat_id,
             "trial": v.trial, "selected_video": v.relative_path} for v in videos]
    write_rows(MANIFEST_PATH, rows, ("day", "cage", "animal", "trial", "selected_video"))


def require_models() -> None:
    missing = [str(path) for path in (TRACKING_CONFIG, TICK_CONFIG) if not path.exists()]
    if missing:
        raise RuntimeError("Missing model configuration: " + "; ".join(missing))


def ensure_excel_support() -> None:
    """Use the GUI environment's openpyxl package when DLC uses its own venv."""
    try:
        import openpyxl  # noqa: F401
    except ModuleNotFoundError:
        gui_site_packages = ROOT / ".venv" / "Lib" / "site-packages"
        if gui_site_packages.exists():
            sys.path.insert(0, str(gui_site_packages))
        try:
            import openpyxl  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError("Excel support is missing from both Python environments.") from exc


def calibrate(t1_videos: list[TrialVideo], status: dict) -> bool:
    require_models()
    import deeplabcut
    calibrations = TickCalibrationStore()
    detector = DLCTickDetector()
    completed = set(status["ticks_completed"])
    saved = calibrations.load_by_key()
    pending = [v for v in t1_videos if v.relative_path not in completed or calibration_key(v) not in saved]
    if not pending:
        print(f"TICK_STATUS\tcomplete\t{len(t1_videos)}/{len(t1_videos)}", flush=True)
        return True
    failures: list[dict[str, str]] = []
    for batch_index, batch in enumerate(chunks(pending), 1):
        try:
            clips = [workflow.make_tick_calibration_clip(v.path,
                DEFAULT_TICK_DLC_OUTPUT_DIR / "_calibration_clips", 10) for v in batch]
            workflow.clear_existing_dlc_outputs(DEFAULT_TICK_DLC_OUTPUT_DIR, clips)
            print(f"TICK_STATUS\trunning\tbatch {batch_index}\t{len(batch)} trials", flush=True)
            deeplabcut.analyze_videos(str(TICK_CONFIG), [str(c) for c in clips],
                videotype=".avi", shuffle=1, save_as_csv=True,
                destfolder=str(DEFAULT_TICK_DLC_OUTPUT_DIR))
        except Exception as exc:
            failures.extend({"video": v.relative_path, "reason": str(exc)} for v in batch)
            break
        for video in batch:
            detection = detector.detect_for_video(video)
            if {tick.distance_cm for tick in detection.ticks} == set(range(0, 121, 10)):
                calibrations.save(calibration_from_detection(video, detection, now_stamp()))
                completed.add(video.relative_path)
            else:
                failures.append({"video": video.relative_path, "reason": detection.message})
        status["ticks_completed"] = sorted(completed)
        save_status(status)
    status["tick_failures"] = failures
    write_rows(TICK_FAILURES, failures, ("video", "reason"))
    save_status(status)
    if failures:
        print(f"TICK_STATUS\tfailed\t{len(failures)} calibration(s); tracking was not started", flush=True)
        return False
    print(f"TICK_STATUS\tcomplete\t{len(completed)}/{len(t1_videos)}", flush=True)
    return True


def automatic_annotation(video: TrialVideo, calibrations, predictions):
    calibration = calibrations.get(calibration_key(video))
    csv_path = predictions.find_for_video(video)
    if calibration is None or csv_path is None:
        return None, None, "Missing confirmed calibration or new tracking CSV."
    try:
        tracking = predictions.load(csv_path).filtered(SCORING_LIKELIHOOD_CUTOFF)
        timeline = analyze_tracking_timeline(tracking, calibration)
    except (OSError, ValueError) as exc:
        return None, None, str(exc)
    if timeline.final_state not in {FELL, REACHED} or timeline.start_frame is None or timeline.end_frame is None:
        return None, tracking, "No automatic start-and-terminal event was detected."
    outcome = OUTCOME_FELL if timeline.final_state == FELL else OUTCOME_REACHED
    distance = timeline.farthest_distance_cm if outcome == OUTCOME_FELL else BEAM_LENGTH_CM
    start_tick, stop_tick = point_for_distance(calibration, 0), point_for_distance(calibration, distance)
    if start_tick is None or stop_tick is None:
        return None, tracking, "A required scoring tick is missing."
    capture = cv2.VideoCapture(str(video.path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
    capture.release()
    start = PawMark(timeline.start_frame, start_tick.x, start_tick.y)
    stop = PawMark(timeline.end_frame, stop_tick.x, stop_tick.y)
    return TrialAnnotation(relative_video=video.relative_path, dataset=video.dataset, day=video.day,
        group=video.group, subject=video.subject, trial=video.trial, fps=fps,
        start_frame=start.frame, start_time=start.frame / fps, start_x=start.x, start_y=start.y,
        stop_frame=stop.frame, stop_time=stop.frame / fps, stop_x=stop.x, stop_y=stop.y,
        crossing_time=scored_crossing_time_seconds(outcome, raw_crossing_time_seconds(start, stop, fps)),
        outcome=outcome, distance_cm=distance, max_time_applied=max_time_applied(outcome), saved_at=now_stamp()), tracking, None


def analyze_and_save(videos: list[TrialVideo], status: dict) -> bool:
    require_models()
    ensure_excel_support()
    import deeplabcut
    calibrations = TickCalibrationStore().load_by_key()
    missing = [v.relative_path for v in videos if calibration_key(v) not in calibrations]
    if missing:
        print(f"TRACKING_STATUS\tblocked\t{len(missing)} missing tick calibrations", flush=True)
        return False
    predictions = DLCPredictionStore()
    annotations, workbook, tails = AnnotationStore(), ResultsWorkbook(), TailPositionStore()
    completed, saved = set(status["tracking_completed"]), set(status["saved_results"])
    pending = [v for v in videos if v.relative_path not in completed]
    for batch_index, batch in enumerate(chunks(pending), 1):
        try:
            workflow.clear_existing_dlc_outputs(TRACKING_OUTPUT, [v.path for v in batch])
            print(f"TRACKING_STATUS\trunning\tbatch {batch_index}\t{len(batch)} trials", flush=True)
            deeplabcut.analyze_videos(str(TRACKING_CONFIG), [str(v.path) for v in batch],
                videotype=".avi", shuffle=1, save_as_csv=True, destfolder=str(TRACKING_OUTPUT))
        except Exception as exc:
            print(f"TRACKING_STATUS\tfailed\t{exc}", flush=True)
            return False
        review: list[dict[str, str]] = []
        for video in batch:
            annotation, tracking, reason = automatic_annotation(video, calibrations, predictions)
            if annotation is None:
                review.append({"video": video.relative_path, "reason": reason or "Unknown scoring error"})
                continue
            try:
                annotations.save(annotation)
                tails.record_trial(video, tracking, calibrations[calibration_key(video)],
                    annotation.start_frame, annotation.stop_frame)
                workbook.save(annotation)
            except Exception as exc:
                review.append({"video": video.relative_path, "reason": f"Could not save result: {exc}"})
                continue
            completed.add(video.relative_path)
            saved.add(video.relative_path)
        status["tracking_completed"] = sorted(completed)
        status["saved_results"] = sorted(saved)
        status["scoring_review"] = review
        save_status(status)
        if review:
            write_rows(REVIEW_FAILURES, review, ("video", "reason"))
            print(f"SAVE_STATUS\tmanual-review\t{len(review)} trial(s); batch stopped before continuing", flush=True)
            return False
        print(f"TRACKING_STATUS\tcomplete\t{len(completed)}/{len(videos)}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate all selected SEONG RBT DATA trials.")
    parser.add_argument("--stage", choices=("all", "ticks", "tracking"), default="all")
    parser.add_argument("--restart", action="store_true", help="run fresh tick and tracking predictions")
    args = parser.parse_args()
    videos = selected_videos()
    t1_videos = tick_sources(videos)
    write_manifest(videos)
    status = load_status(videos, args.restart)
    save_status(status)
    print(f"EVALUATION_STATUS\tselected\t{len(videos)} trials; {len(t1_videos)} T1 calibrations", flush=True)
    if args.stage in {"all", "ticks"} and not calibrate(t1_videos, status):
        return 2
    if args.stage == "ticks":
        return 0
    if args.stage in {"all", "tracking"} and not analyze_and_save(videos, status):
        return 3
    print(f"EVALUATION_STATUS\tcomplete\t{len(status['saved_results'])}/{len(videos)} scored trials", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())