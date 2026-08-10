from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbtcv.dataset import DatasetIndex, TrialVideo
from rbtcv.detection import TRACKING_BODYPARTS


NORMAL_DATASET = "RBT DATA_B"
FALL_DATASET = "SEONG RBT DATA"
FALL_DAY = "D3"
FALL_SUBJECTS = {("C1", "5"), ("C3", "2"), ("C3", "3"), ("C4", "1")}


def evenly_spaced(videos: list[TrialVideo], count: int) -> list[TrialVideo]:
    if len(videos) <= count:
        return videos
    if count == 1:
        return [videos[len(videos) // 2]]
    indexes = [round(index * (len(videos) - 1) / (count - 1)) for index in range(count)]
    return [videos[index] for index in dict.fromkeys(indexes)]


def selected_videos() -> tuple[list[TrialVideo], list[TrialVideo]]:
    dataset = DatasetIndex()
    normal: list[TrialVideo] = []
    for day in dataset.days:
        for trial in (1, 2, 3):
            rbt = [video for video in dataset.videos if video.dataset == NORMAL_DATASET and video.day == day and video.trial == trial]
            seong = [
                video
                for video in dataset.videos
                if video.dataset == FALL_DATASET
                and video.day == day
                and video.trial == trial

                and not (video.day.upper() == FALL_DAY and (video.group, video.subject) in FALL_SUBJECTS)
            ]
            if rbt:
                normal.extend(evenly_spaced(rbt, 2))
            if seong:
                normal.extend(evenly_spaced(seong, 1))

    falls = [
        video
        for video in dataset.videos
        if video.dataset == FALL_DATASET
        and video.day.upper() == FALL_DAY
        and (video.group, video.subject) in FALL_SUBJECTS
    ]
    if len(normal) != 54 or len(falls) != 12:
        raise RuntimeError(f"Expected 54 normal and 12 fall videos; found {len(normal)} normal and {len(falls)} fall.")
    return normal, falls


def write_manifest(path: Path, normal: list[TrialVideo], falls: list[TrialVideo]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "day", "trial", "purpose", "source_video", "frames", "sampling"))
        writer.writeheader()
        for video in normal:
            writer.writerow({
                "dataset": video.dataset,
                "day": video.day,
                "trial": video.trial,
                "purpose": "normal",
                "source_video": video.relative_path,
                "frames": 8,
                "sampling": "0-25%: 3; 25-75%: 4; 75-90%: 1",
            })
        for video in falls:
            writer.writerow({
                "dataset": video.dataset,
                "day": video.day,
                "trial": video.trial,
                "purpose": "known_D3_fall",
                "source_video": video.relative_path,
                "frames": 15,
                "sampling": "0-25%: 3; 25-55%: 2; 55-75%: 3; 75-90%: 5; 90-100%: 2",
            })


def clear_project(config_path: Path, auxiliaryfunctions, *, discard_labels: bool) -> None:
    project = config_path.parent.resolve()
    labeled_data = project / "labeled-data"
    videos_dir = project / "videos"
    for folder in labeled_data.iterdir() if labeled_data.exists() else ():
        if not folder.is_dir():
            continue
        labels = [file for file in folder.iterdir() if file.name.startswith("CollectedData_")]
        if labels and not discard_labels:
            raise RuntimeError(
                f"Refusing to clear {folder.name}: it contains saved labels. "
                "Re-run with --discard-labels to remove them deliberately."
            )
    for folder in labeled_data.iterdir() if labeled_data.exists() else ():
        if folder.is_dir():
            shutil.rmtree(folder)
    for video in videos_dir.iterdir() if videos_dir.exists() else ():
        if video.is_file():
            video.unlink()

    config = auxiliaryfunctions.read_config(str(config_path))
    config["video_sets"] = {}
    config["bodyparts"] = list(TRACKING_BODYPARTS)
    config["TrainingFraction"] = [0.95]
    config["numframes2pick"] = 8
    config["start"] = 0.0
    config["stop"] = 1.0
    auxiliaryfunctions.write_config(str(config_path), config)


def project_paths(config_path: Path, videos: list[TrialVideo]) -> list[str]:
    copied_videos = config_path.parent.resolve() / "videos"
    return [str((copied_videos / video.path.name).resolve()) for video in videos]


def extract_phase(deeplabcut, auxiliaryfunctions, config_path: Path, videos: list[TrialVideo], frames: int, start: float, stop: float) -> None:
    if not videos:
        return
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
        videos_list=project_paths(config_path, videos),
    )
    if result and any(result):
        raise RuntimeError("DeepLabCut frame extraction did not finish.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild a focused mouse paw/tail/body labeling dataset")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--reset-project", action="store_true")
    parser.add_argument("--discard-labels", action="store_true", help="Permit reset to remove saved frame annotations.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "focused_mouse_tracking_batch.csv")
    args = parser.parse_args()

    if not args.config.exists() or args.batch_start < 0 or args.batch_size < 1:
        print("Provide an existing config and a non-negative batch start with a positive batch size.")
        return 1
    if args.reset_project and args.batch_start != 0:
        print("--reset-project can only be used for the first batch.")
        return 1

    normal, falls = selected_videos()
    write_manifest(args.manifest, normal, falls)
    records = [("normal", video) for video in normal] + [("fall", video) for video in falls]
    batch = records[args.batch_start : args.batch_start + args.batch_size]
    if not batch:
        print("The requested batch is empty.")
        return 1

    try:
        import deeplabcut
        from deeplabcut.utils import auxiliaryfunctions
    except ImportError:
        print("DeepLabCut is not installed. Use .venv-dlc.")
        return 1

    if args.reset_project:
        try:
            clear_project(args.config, auxiliaryfunctions, discard_labels=args.discard_labels)
        except RuntimeError as exc:
            print(exc)
            return 1
        print("Cleared the prior mouse-video registration, extracted frames, and permitted annotations.")

    normal_batch = [video for purpose, video in batch if purpose == "normal"]
    fall_batch = [video for purpose, video in batch if purpose == "fall"]
    config = auxiliaryfunctions.read_config(str(args.config))
    registered = {Path(path).name for path in config.get("video_sets", {})}
    new_videos = [video for _, video in batch if video.path.name not in registered]
    if new_videos:
        deeplabcut.add_new_videos(str(args.config), [str(video.path) for video in new_videos], copy_videos=True)

    try:
        extract_phase(deeplabcut, auxiliaryfunctions, args.config, normal_batch, 3, 0.00, 0.25)
        extract_phase(deeplabcut, auxiliaryfunctions, args.config, normal_batch, 4, 0.25, 0.75)
        extract_phase(deeplabcut, auxiliaryfunctions, args.config, normal_batch, 1, 0.75, 0.90)
        extract_phase(deeplabcut, auxiliaryfunctions, args.config, fall_batch, 3, 0.00, 0.25)
        extract_phase(deeplabcut, auxiliaryfunctions, args.config, fall_batch, 2, 0.25, 0.55)
        extract_phase(deeplabcut, auxiliaryfunctions, args.config, fall_batch, 3, 0.55, 0.75)
        extract_phase(deeplabcut, auxiliaryfunctions, args.config, fall_batch, 5, 0.75, 0.90)
        extract_phase(deeplabcut, auxiliaryfunctions, args.config, fall_batch, 2, 0.90, 1.00)
    except RuntimeError as exc:
        print(exc)
        return 1
    finally:
        config = auxiliaryfunctions.read_config(str(args.config))
        config["bodyparts"] = list(TRACKING_BODYPARTS)
        config["TrainingFraction"] = [0.95]
        config["numframes2pick"] = 8
        config["start"] = 0.0
        config["stop"] = 1.0
        auxiliaryfunctions.write_config(str(args.config), config)

    print(f"Completed batch {args.batch_start}-{args.batch_start + len(batch) - 1}: {len(normal_batch)} normal, {len(fall_batch)} fall videos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())