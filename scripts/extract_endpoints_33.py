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
from extract_platform_end_frames import extract_phase

SOURCE_CONFIG = ROOT / "models" / "dlc_tracking" / "RBT_visible_front_back_tail-RBT_CV-2026-07-14" / "config.yaml"
PROJECT_DIR = ROOT / "models" / "dlc_tracking" / "RBT_visible_front_back_tail_endpoints_33-RBT_CV-2026-07-20"
MANIFEST = ROOT / "outputs" / "endpoints_33_mouse_tracking_batch.csv"


def select_evenly(videos: list[Path], count: int) -> list[Path]:
    return [videos[round(i * (len(videos) - 1) / (count - 1))] for i in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a 33-video platform-end supplement")
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if not args.source_config.exists() or args.project_dir.exists():
        print("Source config is missing or destination project already exists.")
        return 1
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
    source = auxiliaryfunctions.read_config(str(args.source_config))
    all_videos = [Path(path) for path in source.get("video_sets", {})]
    if len(all_videos) != 66 or any(not path.exists() for path in all_videos):
        print("Expected 66 available source videos.")
        return 1
    videos = select_evenly(all_videos, 33)
    args.project_dir.mkdir(parents=True)
    (args.project_dir / "videos").mkdir()
    config_path = args.project_dir / "config.yaml"
    try:
        shutil.copy2(args.source_config, config_path)
        config = auxiliaryfunctions.read_config(str(config_path))
        config["Task"] = "RBT_visible_front_back_tail_endpoints_33"
        config["project_path"] = str(args.project_dir.resolve())
        config["video_sets"] = {}
        config["bodyparts"] = list(TRACKING_BODYPARTS)
        config["TrainingFraction"] = [0.95]
        auxiliaryfunctions.write_config(str(config_path), config)
        deeplabcut.add_new_videos(str(config_path), [str(video) for video in videos], copy_videos=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("source_video", "frames", "sampling"))
            writer.writeheader()
            writer.writerows({"source_video": str(video), "frames": 3, "sampling": "80-90%: 1; 90-97%: 1; 97-100%: 1"} for video in videos)
        extract_phase(deeplabcut, auxiliaryfunctions, config_path, videos, 1, 0.80, 0.90)
        extract_phase(deeplabcut, auxiliaryfunctions, config_path, videos, 1, 0.90, 0.97)
        extract_phase(deeplabcut, auxiliaryfunctions, config_path, videos, 1, 0.97, 1.00)
    except Exception:
        shutil.rmtree(args.project_dir, ignore_errors=True)
        raise
    finally:
        if config_path.exists():
            config = auxiliaryfunctions.read_config(str(config_path))
            config["bodyparts"] = list(TRACKING_BODYPARTS)
            config["TrainingFraction"] = [0.95]
            auxiliaryfunctions.write_config(str(config_path), config)
    print(f"Created {args.project_dir.name}: {len(videos)} videos, {len(videos) * 3} endpoint frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())