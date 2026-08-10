from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbtcv.detection import TRACKING_BODYPARTS
from extract_platform_end_frames import DEFAULT_PROJECT_DIR, DEFAULT_SOURCE_CONFIG, extract_phase, write_manifest


def image_count(project_dir: Path, video: Path) -> int:
    return len(list((project_dir / "labeled-data" / video.stem).glob("*.png")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume one platform-end frame extraction batch")
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--batch-start", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "platform_end_mouse_tracking_batch.csv")
    args = parser.parse_args()
    config_path = args.project_dir / "config.yaml"
    if not args.source_config.exists() or not config_path.exists() or args.batch_start < 0 or args.batch_size < 1:
        print("Missing config or invalid batch arguments.")
        return 1

    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions

    source = auxiliaryfunctions.read_config(str(args.source_config))
    videos = [Path(path) for path in source.get("video_sets", {})]
    if len(videos) != 66:
        print(f"Expected 66 source videos, found {len(videos)}.")
        return 1
    batch = videos[args.batch_start : args.batch_start + args.batch_size]
    if not batch:
        print("The requested batch is empty.")
        return 1
    invalid = [(video.name, image_count(args.project_dir, video)) for video in batch if image_count(args.project_dir, video) not in {0, 2, 5, 7}]
    if invalid:
        print(f"Unexpected frame count: {invalid[0]}")
        return 1

    write_manifest(args.manifest, videos)
    phases = ((0, 2, 0.75, 0.88), (2, 3, 0.88, 0.96), (5, 2, 0.96, 1.00))
    try:
        for expected, frames, start, stop in phases:
            pending = [video for video in batch if image_count(args.project_dir, video) == expected]
            if pending:
                extract_phase(deeplabcut, auxiliaryfunctions, config_path, pending, frames, start, stop)
    finally:
        config = auxiliaryfunctions.read_config(str(config_path))
        config["bodyparts"] = list(TRACKING_BODYPARTS)
        config["TrainingFraction"] = [0.95]
        config["numframes2pick"] = 7
        config["start"] = 0.0
        config["stop"] = 1.0
        auxiliaryfunctions.write_config(str(config_path), config)
    counts = [image_count(args.project_dir, video) for video in batch]
    print(f"Completed batch {args.batch_start}-{args.batch_start + len(batch) - 1}; frame_counts={sorted(set(counts))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())