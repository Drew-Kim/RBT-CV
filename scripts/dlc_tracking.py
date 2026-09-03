"""DeepLabCut commands used by the RBT-CV GUI.

The GUI launches this module only for fresh paw/tail/body-center tracking and
tick-calibration inference. Training and historical frame-extraction helpers
live outside the runtime path and are intentionally not kept here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbtcv.detection import DEFAULT_DLC_PREDICTIONS_DIR


DEFAULT_TICK_PREDICTIONS_DIR = ROOT / "outputs" / "dlc_tick_predictions"


def load_deeplabcut():
    try:
        import deeplabcut
    except ImportError:
        print("DeepLabCut is not installed. Activate .venv-dlc and try again.")
        return None
    return deeplabcut


def clear_existing_outputs(output_dir: Path, videos: list[Path]) -> int:
    """Remove stale derived predictions only for videos being reanalyzed."""
    removed = 0
    for video in dict.fromkeys(videos):
        for result in output_dir.glob(f"{video.stem}DLC_*"):
            if result.is_file():
                result.unlink()
                removed += 1
    return removed


def run_dlc_analysis(config: Path, videos: list[Path], output_dir: Path, shuffle: int) -> int:
    """Run fresh CSV inference for explicit videos and report GUI-friendly progress."""
    if not config.exists():
        print(f"Missing config file: {config}")
        return 1

    unique_videos = list(dict.fromkeys(videos))
    if not unique_videos:
        print("No videos were selected.")
        return 1

    missing = next((video for video in unique_videos if not video.exists()), None)
    if missing is not None:
        print(f"Missing video file: {missing}")
        return 1

    deeplabcut = load_deeplabcut()
    if deeplabcut is None:
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    removed = clear_existing_outputs(output_dir, unique_videos)
    if removed:
        print(f"Removed {removed} stale DLC output file(s); running fresh analysis.")

    total = len(unique_videos)
    for index, video in enumerate(unique_videos, start=1):
        print(f"RBT_PROGRESS\t{index}\t{total}\t{video}", flush=True)
        deeplabcut.analyze_videos(
            str(config),
            [str(video)],
            videotype=video.suffix.lower() or ".avi",
            shuffle=shuffle,
            save_as_csv=True,
            destfolder=str(output_dir),
        )

    print(f"Saved DLC CSV files for {total} video(s) to: {output_dir}")
    return 0


def analyze_exact_videos(args: argparse.Namespace) -> int:
    return run_dlc_analysis(args.config, args.video, args.output_dir, args.shuffle)


def make_tick_calibration_clip(video: Path, output_dir: Path, frame_count: int) -> Path:
    """Create a short calibration clip from frames spread across the full trial."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_path = output_dir / f"{video.stem}_tick_calibration.avi"

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create calibration clip: {clip_path}")

    total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_count == 1:
        frame_numbers = [0]
    else:
        frame_numbers = [
            round(index * (total_frames - 1) / (frame_count - 1))
            for index in range(frame_count)
        ]

    written = 0
    for frame_number in dict.fromkeys(frame_numbers):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if ok:
            writer.write(frame)
            written += 1

    capture.release()
    writer.release()
    if not written:
        raise RuntimeError(f"No frames could be read from: {video}")
    return clip_path


def analyze_tick_video(args: argparse.Namespace) -> int:
    if not args.video.exists():
        print(f"Missing video file: {args.video}")
        return 1
    if args.early_frames < 1:
        print("--early-frames must be at least 1.")
        return 1

    try:
        clip = make_tick_calibration_clip(
            args.video,
            args.output_dir / "_calibration_clips",
            args.early_frames,
        )
    except RuntimeError as exc:
        print(exc)
        return 1

    return run_dlc_analysis(args.config, [clip], args.output_dir, args.shuffle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RBT-CV DeepLabCut inference runner")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze-files", help="analyze selected trial videos")
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--video", type=Path, action="append", required=True)
    analyze.add_argument("--output-dir", type=Path, default=DEFAULT_DLC_PREDICTIONS_DIR)
    analyze.add_argument("--shuffle", type=int, default=1)
    analyze.set_defaults(handler=analyze_exact_videos)

    ticks = commands.add_parser("tick-analyze", help="analyze distributed tick-calibration frames")
    ticks.add_argument("--config", type=Path, required=True)
    ticks.add_argument("--video", type=Path, required=True)
    ticks.add_argument("--output-dir", type=Path, default=DEFAULT_TICK_PREDICTIONS_DIR)
    ticks.add_argument("--shuffle", type=int, default=1)
    ticks.add_argument(
        "--early-frames",
        type=int,
        default=10,
        help="number of candidate frames distributed across the full video",
    )
    ticks.set_defaults(handler=analyze_tick_video)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
