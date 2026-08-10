from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbtcv.detection import TRACKING_BODYPARTS


DEFAULT_SOURCE_CONFIG = ROOT / "models" / "dlc_tracking" / "RBT_visible_front_back_tail-RBT_CV-2026-07-14" / "config.yaml"
DEFAULT_PROJECT_DIR = ROOT / "models" / "dlc_tracking" / "RBT_visible_front_back_tail_platform_end-RBT_CV-2026-07-20"


def write_manifest(path: Path, videos: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_video", "frames", "sampling", "purpose"))
        writer.writeheader()
        for video in videos:
            writer.writerow({
                "source_video": str(video),
                "frames": 7,
                "sampling": "75-88%: 2; 88-96%: 3; 96-100%: 2",
                "purpose": "platform_end_supplement",
            })


def destination_paths(config_path: Path, videos: list[Path]) -> list[str]:
    videos_dir = config_path.parent / "videos"
    return [str((videos_dir / video.name).resolve()) for video in videos]


def extract_phase(deeplabcut, auxiliaryfunctions, config_path: Path, videos: list[Path], frames: int, start: float, stop: float) -> None:
    config = auxiliaryfunctions.read_config(str(config_path))
    config["numframes2pick"] = frames
    config["start"] = start
    config["stop"] = stop
    auxiliaryfunctions.write_config(str(config_path), config)
    result = deeplabcut.extract_frames(
        str(config_path),
        mode="automatic",
        algo="kmeans",
        crop=False,
        cluster_step=1,
        userfeedback=False,
        videos_list=destination_paths(config_path, videos),
    )
    if result and any(result):
        raise RuntimeError("DeepLabCut platform-end frame extraction did not finish.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a separate platform-end frame-labeling project")
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "platform_end_mouse_tracking_batch.csv")
    args = parser.parse_args()

    if not args.source_config.exists():
        print(f"Missing source config: {args.source_config}")
        return 1
    if args.project_dir.exists():
        print(f"Destination already exists; leaving it unchanged: {args.project_dir}")
        return 1

    try:
        import deeplabcut
        from deeplabcut.utils import auxiliaryfunctions
    except ImportError:
        print("DeepLabCut is not installed. Use .venv-dlc.")
        return 1

    source_config = auxiliaryfunctions.read_config(str(args.source_config))
    source_videos = [Path(path) for path in source_config.get("video_sets", {})]
    if len(source_videos) != 66:
        print(f"Expected 66 source videos, found {len(source_videos)}. No project was created.")
        return 1
    missing = [path for path in source_videos if not path.exists()]
    if missing:
        print(f"Source video is missing: {missing[0]}")
        return 1
    if len({video.name for video in source_videos}) != len(source_videos):
        print("Source video filenames are not unique; no project was created.")
        return 1

    args.project_dir.mkdir(parents=True)
    (args.project_dir / "videos").mkdir()
    config_path = args.project_dir / "config.yaml"
    try:
        shutil.copy2(args.source_config, config_path)
        config = auxiliaryfunctions.read_config(str(config_path))
        config["Task"] = "RBT_visible_front_back_tail_platform_end"
        config["project_path"] = str(args.project_dir.resolve())
        config["video_sets"] = {}
        config["bodyparts"] = list(TRACKING_BODYPARTS)
        config["TrainingFraction"] = [0.95]
        config["numframes2pick"] = 7
        config["start"] = 0.0
        config["stop"] = 1.0
        auxiliaryfunctions.write_config(str(config_path), config)

        deeplabcut.add_new_videos(str(config_path), [str(video) for video in source_videos], copy_videos=True)
        write_manifest(args.manifest, source_videos)
        extract_phase(deeplabcut, auxiliaryfunctions, config_path, source_videos, 2, 0.75, 0.88)
        extract_phase(deeplabcut, auxiliaryfunctions, config_path, source_videos, 3, 0.88, 0.96)
        extract_phase(deeplabcut, auxiliaryfunctions, config_path, source_videos, 2, 0.96, 1.00)
    except Exception:
        shutil.rmtree(args.project_dir, ignore_errors=True)
        raise
    finally:
        if config_path.exists():
            config = auxiliaryfunctions.read_config(str(config_path))
            config["bodyparts"] = list(TRACKING_BODYPARTS)
            config["TrainingFraction"] = [0.95]
            config["numframes2pick"] = 7
            config["start"] = 0.0
            config["stop"] = 1.0
            auxiliaryfunctions.write_config(str(config_path), config)

    print(f"Created separate platform-end project: {config_path}")
    print(f"Registered {len(source_videos)} videos and extracted {len(source_videos) * 7} platform-end frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())