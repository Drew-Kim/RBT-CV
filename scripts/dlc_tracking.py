from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbtcv.dataset import DatasetIndex, TrialVideo
from rbtcv.detection import (
    DEFAULT_DLC_PREDICTIONS_DIR,
    DISPLAY_LIKELIHOOD_CUTOFF,
    SCORING_LIKELIHOOD_CUTOFF,
    DLCFramePrediction,
    DLCPoint,
    DLCTracking,
    DLCPredictionStore,
    TRACKING_BODYPARTS,
)
from rbtcv.ticks import BeamCalibration, BeamTick
from rbtcv.tracking_rules import FELL, REACHED, analyze_tracking_timeline

DEFAULT_PROJECT_DIR = ROOT / "models" / "dlc_tracking"
DEFAULT_SCORER = "RBT_CV"


def load_deeplabcut():
    try:
        import deeplabcut
    except ImportError:
        print("DeepLabCut is not installed. Activate .venv-dlc and try again.")
        return None
    return deeplabcut


def select_videos(day: str | None, trial: int | None, limit: int | None) -> list[TrialVideo]:
    videos = list(DatasetIndex().videos)
    if day and day.lower() != "all":
        videos = [video for video in videos if video.day.upper() == day.upper()]
    if trial:
        videos = [video for video in videos if video.trial == trial]
    return videos[:limit] if limit else videos


def edit_config(deeplabcut, config_path: str) -> None:
    try:
        edit = deeplabcut.auxiliaryfunctions.edit_config
    except AttributeError:
        from deeplabcut.utils import auxiliaryfunctions
        edit = auxiliaryfunctions.edit_config
    edit(config_path, {"bodyparts": list(TRACKING_BODYPARTS), "TrainingFraction": [0.95], "dotsize": 5, "pcutoff": 0.6})

def create_project(args) -> int:
    deeplabcut = load_deeplabcut()
    if deeplabcut is None:
        return 1
    videos = select_videos(args.day, args.trial, args.limit)
    if not videos:
        print("No matching videos found.")
        return 1
    args.working_directory.mkdir(parents=True, exist_ok=True)
    config = deeplabcut.create_new_project(
        args.project_name,
        args.scorer,
        [str(video.path) for video in videos],
        working_directory=str(args.working_directory),
        copy_videos=not args.use_symlinks,
        multianimal=False,
    )
    edit_config(deeplabcut, config)
    print(f"Created project: {config}")
    print("Label: visible_back_paw, visible_front_paw, tail_end, and body_center (the middle of the torso).")
    return 0


def train_model(args) -> int:
    deeplabcut = load_deeplabcut()
    if deeplabcut is None or not args.config.exists():
        print(f"Missing config file: {args.config}")
        return 1
    config = str(args.config)
    if not args.skip_convert:
        deeplabcut.convertcsv2h5(config, scorer=args.scorer)
    if not args.skip_check_labels:
        deeplabcut.check_labels(config)
    if not args.skip_train:
        deeplabcut.create_training_dataset(config)
        deeplabcut.train_network(config, shuffle=args.shuffle, maxiters=args.maxiters, displayiters=args.displayiters, saveiters=args.saveiters)
    if not args.skip_evaluate:
        deeplabcut.evaluate_network(config, Shuffles=[args.shuffle], plotting=True)
    print("Training complete.")
    return 0


def clear_existing_dlc_outputs(output_dir: Path, videos: list[Path]) -> int:
    """Remove only stale derived DLC outputs for the videos about to be reanalyzed."""
    removed = 0
    for video in dict.fromkeys(videos):
        for result in output_dir.glob(f"{video.stem}DLC_*"):
            if result.is_file():
                result.unlink()
                removed += 1
    return removed

def run_dlc_analysis(
    config: Path,
    videos: list[Path],
    output_dir: Path,
    shuffle: int,
) -> int:
    """Validate video paths once and run one DLC analysis per file type."""
    if not config.exists():
        print(f"Missing config file: {config}")
        return 1

    unique_videos = list(dict.fromkeys(videos))
    if not unique_videos:
        print("No matching videos found.")
        return 1

    missing = next((video for video in unique_videos if not video.exists()), None)
    if missing is not None:
        print(f"Missing video file: {missing}")
        return 1

    deeplabcut = load_deeplabcut()
    if deeplabcut is None:
        return 1


    output_dir.mkdir(parents=True, exist_ok=True)
    removed = clear_existing_dlc_outputs(output_dir, unique_videos)
    if removed:
        print(f"Removed {removed} stale DLC output file(s); running fresh analysis.")

    total = len(unique_videos)
    for index, video in enumerate(unique_videos, start=1):
        # A stable progress marker lets the GUI show the current animal/trial.
        print(f"RBT_PROGRESS\t{index}\t{total}\t{video}", flush=True)
        deeplabcut.analyze_videos(
            str(config),
            [str(video)],
            videotype=video.suffix.lower() or ".avi",
            shuffle=shuffle,
            save_as_csv=True,
            destfolder=str(output_dir),
        )

    print(f"Saved DLC CSV files for {len(unique_videos)} video(s) to: {output_dir}")
    return 0


def analyze_videos(args) -> int:
    videos = select_videos(args.day, args.trial, args.limit)
    return run_dlc_analysis(
        args.config,
        [video.path for video in videos],
        args.output_dir,
        args.shuffle,
    )


def analyze_one_video(args) -> int:
    return run_dlc_analysis(args.config, [args.video], args.output_dir, args.shuffle)


def analyze_exact_videos(args) -> int:
    return run_dlc_analysis(args.config, args.video, args.output_dir, args.shuffle)

def make_tick_calibration_clip(video: Path, output_dir: Path, frame_limit: int) -> Path:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    clip_path = output_dir / f"{video.stem}_tick_calibration.avi"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create calibration clip: {clip_path}")

    total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    # Spread samples across the full trial: the mouse can hide different tick
    # marks at different points, while the beam itself stays fixed.
    if frame_limit == 1:
        frame_numbers = [0]
    else:
        frame_numbers = [round(index * (total_frames - 1) / (frame_limit - 1)) for index in range(frame_limit)]

    written = 0
    for frame_number in dict.fromkeys(frame_numbers):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        writer.write(frame)
        written += 1
    capture.release()
    writer.release()
    if not written:
        raise RuntimeError(f"No frames could be read from: {video}")
    return clip_path


def analyze_tick_video(args) -> int:
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

    # Remove only this temporary clip's stale results so the GUI always loads
    # predictions from the analysis started by the current button click.
    for previous_result in args.output_dir.glob(f"{clip.stem}DLC_*"):
        if previous_result.is_file():
            previous_result.unlink()

    return run_dlc_analysis(args.config, [clip], args.output_dir, args.shuffle)

def plot_tick_report(args) -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        print(f"Missing plotting dependency: {exc}")
        return 1

    if not args.csv.exists():
        print(f"Missing DLC CSV file: {args.csv}")
        return 1

    data = pd.read_csv(args.csv, header=[0, 1, 2], index_col=0)
    bodyparts = data.columns.get_level_values(1)
    labels = [label for label in dict.fromkeys(bodyparts) if label.startswith("tick_")]
    if not labels:
        print("No tick labels were found in this CSV file.")
        return 1

    frames = pd.to_numeric(data.index, errors="coerce")
    report_dir = args.output_dir or args.csv.parent / "tick-reports" / args.csv.stem
    report_dir.mkdir(parents=True, exist_ok=True)
    title = args.csv.stem.split("DLC_")[0].rstrip("_- ")

    confidence = {}
    positions = {"x": {}, "y": {}}
    for label in labels:
        label_data = data.xs(label, axis=1, level=1)
        for coordinate in ("x", "y", "likelihood"):
            values = pd.to_numeric(label_data.xs(coordinate, axis=1, level=1).iloc[:, 0], errors="coerce")
            if coordinate == "likelihood":
                confidence[label] = values
            else:
                positions[coordinate][label] = values

    fig, axis = plt.subplots(figsize=(12, 6))
    for label in labels:
        axis.plot(frames, confidence[label], label=label, linewidth=1)
    axis.axhline(args.pcutoff, color="black", linestyle="--", linewidth=1, label=f"Display cutoff ({args.pcutoff:.2f})")
    axis.set_title(f"Tick prediction confidence - {title}")
    axis.set_xlabel("Video frame index")
    axis.set_ylabel("DeepLabCut confidence (likelihood, 0 to 1)")
    axis.set_ylim(0, 1.05)
    axis.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Tick landmark")
    fig.tight_layout()
    fig.savefig(report_dir / "tick_confidence_over_time.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label in labels:
        axes[0].plot(frames, positions["x"][label], label=label, linewidth=1)
        axes[1].plot(frames, positions["y"][label], label=label, linewidth=1)
    axes[0].set_title(f"Tick position stability - {title}")
    axes[0].set_ylabel("Horizontal position (pixels)")
    axes[1].set_xlabel("Video frame index")
    axes[1].set_ylabel("Vertical position (pixels)")
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Tick landmark")
    fig.tight_layout()
    fig.savefig(report_dir / "tick_positions_over_time.png", dpi=180)
    plt.close(fig)

    averages = [confidence[label].mean() for label in labels]
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(labels, averages, color="#2f7f9d")
    axis.axhline(args.pcutoff, color="black", linestyle="--", linewidth=1, label=f"Display cutoff ({args.pcutoff:.2f})")
    axis.set_title(f"Average tick prediction confidence - {title}")
    axis.set_xlabel("Tick landmark")
    axis.set_ylabel("Average DeepLabCut confidence (likelihood, 0 to 1)")
    axis.set_ylim(0, 1.05)
    axis.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "tick_average_confidence.png", dpi=180)
    plt.close(fig)

    print(f"Saved tick report graphs to: {report_dir}")
    return 0

def test_csv_parser(_args) -> int:
    rows = [
        ["scorer", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC", "DLC"],
        ["bodyparts", "visible_back_paw", "visible_back_paw", "visible_back_paw", "visible_front_paw", "visible_front_paw", "visible_front_paw", "tail_end", "tail_end", "tail_end", "body_center", "body_center", "body_center"],
        ["coords", "x", "y", "likelihood", "x", "y", "likelihood", "x", "y", "likelihood", "x", "y", "likelihood"],
        ["0", "10", "20", "0.95", "30", "40", "0.20", "50", "60", "0.90", "70", "80", "0.99"],
    ]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sample_dlc.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(rows)
        display_tracking = DLCPredictionStore(likelihood_cutoff=DISPLAY_LIKELIHOOD_CUTOFF).load(path)
        display_frame = display_tracking.points_for_frame(0)
        scoring_frame = display_tracking.filtered(SCORING_LIKELIHOOD_CUTOFF).points_for_frame(0)
    if display_frame is None or display_frame.paw_count != 2:
        print("FAIL: display cutoff did not keep the expected visible paw points.")
        return 1
    frame = scoring_frame
    if frame is None or frame.paw_count != 1 or frame.tail_count != 1 or len(frame.body_points) != 1:
        print("FAIL: DLC CSV parser did not keep the expected visible points.")
        return 1
    if "body_center" not in {point.name for point in frame.points}:
        print("FAIL: DLC CSV parser did not keep body_center.")
        return 1
    print("PASS: parser keeps visible paws, tail_end, and body_center.")

    calibration = BeamCalibration(
        key="synthetic",
        dataset="synthetic",
        day="D0",
        cage="1",
        subject="1",
        source_video="synthetic.avi",
        source_trial=1,
        frame_numbers=(0,),
        ticks=(
            BeamTick(0, 100, 50),
            BeamTick(60, 200, 50),
            BeamTick(120, 300, 50),
        ),
        confirmed_at="test",
    )

    def prediction(frame_number: int, back_paw_x: float, body_center_y: float) -> DLCFramePrediction:
        return DLCFramePrediction(
            frame=frame_number,
            points=(
                DLCPoint("visible_back_paw", "paw", back_paw_x, 50.0, 0.99),
                DLCPoint("body_center", "body", back_paw_x, body_center_y, 0.99),
            ),
        )

    reached_tracking = DLCTracking(
        csv_path=Path("synthetic_reached.csv"),
        frames={
            0: prediction(0, 90.0, 40.0),
            1: prediction(1, 100.0, 40.0),
            2: prediction(2, 200.0, 40.0),
            3: prediction(3, 300.0, 40.0),
        },
    )
    reached_timeline = analyze_tracking_timeline(reached_tracking, calibration)
    if (
        reached_timeline.final_state != REACHED
        or reached_timeline.start_frame != 1
        or reached_timeline.end_frame != 3
        or reached_timeline.farthest_distance_cm != 120
    ):
        print("FAIL: automatic scoring did not record the expected completed 120 cm trial.")
        return 1
    print("PASS: automatic scoring starts at tick 0 and stops at tick 120 for a completed trial.")

    fell_tracking = DLCTracking(
        csv_path=Path("synthetic_fell.csv"),
        frames={
            0: prediction(0, 90.0, 40.0),
            1: prediction(1, 100.0, 40.0),
            2: prediction(2, 200.0, 70.0),
        },
    )
    fell_timeline = analyze_tracking_timeline(fell_tracking, calibration)
    if (
        fell_timeline.final_state != FELL
        or fell_timeline.start_frame != 1
        or fell_timeline.end_frame != 2
        or fell_timeline.farthest_distance_cm != 60
    ):
        print("FAIL: automatic scoring did not record the expected fall and farthest distance.")
        return 1
    print("PASS: automatic scoring records a fall below the tick line and its farthest distance.")

    def boundary_timeline(body_center_y: float):
        return analyze_tracking_timeline(
            DLCTracking(
                csv_path=Path("synthetic_boundary.csv"),
                frames={
                    0: prediction(0, 90.0, 40.0),
                    1: prediction(1, 100.0, 40.0),
                    2: prediction(2, 200.0, body_center_y),
                    3: prediction(3, 300.0, 40.0),
                },
            ),
            calibration,
        )

    # The 15-pixel margin protects a mouse that briefly hangs under the beam
    # and then recovers: only a body center more than 15 pixels below the line
    # is treated as a fall.
    if boundary_timeline(64.0).final_state != REACHED or boundary_timeline(65.0).final_state != REACHED:
        print("FAIL: the 15-pixel fall margin ended a recoverable trial too early.")
        return 1
    if boundary_timeline(66.0).final_state != FELL:
        print("FAIL: a body center more than 15 pixels below the beam was not scored as a fall.")
        return 1
    print("PASS: the 15-pixel fall margin allows recovery but records a confirmed fall.")
    return 0


def add_video_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--day", help="Optional day, such as D30. Use all for every day.")
    parser.add_argument("--trial", type=int, choices=(1, 2, 3), help="Optional trial number.")
    parser.add_argument("--limit", type=int, help="Optional maximum videos for a small test run.")


def add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--maxiters", type=int, default=50000)
    parser.add_argument("--displayiters", type=int, default=1000)
    parser.add_argument("--saveiters", type=int, default=10000)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-check-labels", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")


TICK_BATCH_DATASETS = ("RBT DATA_B", "SEONG RBT DATA")
TICK_BATCH_DAYS = ("BL", "D3", "D8", "D9", "D14", "D21", "D30")


def select_tick_batch(videos_per_group: int) -> list[TrialVideo]:
    dataset = DatasetIndex()
    selected: list[TrialVideo] = []
    for dataset_name in TICK_BATCH_DATASETS:
        for day in TICK_BATCH_DAYS:
            matches = [
                video
                for video in dataset.videos
                if video.dataset == dataset_name and video.day.upper() == day and video.trial == 1
            ]
            if matches:
                selected.extend(matches[:videos_per_group])
    if not selected:
        raise ValueError("No matching tick videos were found.")
    return selected


def write_tick_manifest(path: Path, videos: list[TrialVideo], frames_per_video: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "day", "trial", "source_video", "labeled_folder", "frames_to_label"))
        writer.writeheader()
        for video in videos:
            writer.writerow({
                "dataset": video.dataset,
                "day": video.day,
                "trial": video.trial,
                "source_video": video.relative_path,
                "labeled_folder": video.path.stem,
                "frames_to_label": frames_per_video,
            })


def prepare_tick_batch(args) -> int:
    try:
        videos = select_tick_batch(args.videos_per_group)
    except ValueError as exc:
        print(exc)
        return 1
    write_tick_manifest(args.manifest, videos, args.frames_per_video)
    print(f"Saved tick batch manifest: {args.manifest}")
    for video in videos:
        print(f"  {video.dataset} | {video.day} | {video.path.name}")
    if not args.extract:
        print("Review the manifest. Rerun with --extract after closing the DeepLabCut GUI.")
        return 0
    deeplabcut = load_deeplabcut()
    if deeplabcut is None or not args.config.exists():
        print(f"Missing config file: {args.config}")
        return 1
    from deeplabcut.utils import auxiliaryfunctions
    config = auxiliaryfunctions.read_config(str(args.config))
    if not 0 < args.early_fraction <= 1:
        print("--early-fraction must be greater than 0 and no more than 1.")
        return 1
    config["numframes2pick"] = args.frames_per_video
    # Tick calibration happens before the mouse begins its run.
    config["start"] = 0.0
    config["stop"] = args.early_fraction
    auxiliaryfunctions.write_config(str(args.config), config)
    labeled_data = args.config.parent / "labeled-data"
    videos_to_extract = [
        video for video in videos
        if not (labeled_data / video.path.stem).exists()
        or not list((labeled_data / video.path.stem).glob("*.png"))
    ]
    if videos_to_extract:
        # Register videos before extracting. DeepLabCut only extracts paths listed in config.yaml.
        deeplabcut.add_new_videos(
            str(args.config),
            [str(video.path) for video in videos_to_extract],
            copy_videos=True,
        )
        project_videos = args.config.parent.resolve() / "videos"
        project_paths = [str((project_videos / video.path.name).resolve()) for video in videos_to_extract]
        failed = deeplabcut.extract_frames(
            str(args.config),
            mode="automatic",
            algo="kmeans",
            crop=False,
            cluster_step=1,
            userfeedback=False,
            videos_list=project_paths,
        )
        if failed and any(failed):
            print("Tick frame extraction did not finish. Check the messages above.")
            return 1
        print(f"Extracted frames for {len(videos_to_extract)} new tick video(s).")
    else:
        print("All tick-batch folders already contain frames; nothing was overwritten.")
    print("Use the manifest to choose folders in Label Frames.")
    return 0

MOUSE_TRACKING_DATASET = "RBT DATA_B"
FALL_EXAMPLE_DATASET = "SEONG RBT DATA"


def evenly_spaced_videos(videos: list[TrialVideo], count: int) -> list[TrialVideo]:
    """Pick videos across the sorted group instead of clustering on early subjects."""
    if count < 1:
        raise ValueError("Video counts must be at least 1.")
    if len(videos) <= count:
        return videos
    if count == 1:
        return [videos[len(videos) // 2]]
    indexes = [round(index * (len(videos) - 1) / (count - 1)) for index in range(count)]
    return [videos[index] for index in dict.fromkeys(indexes)]


def select_mouse_batch(normal_per_group: int, fall_per_group: int) -> tuple[list[TrialVideo], list[TrialVideo]]:
    """Select balanced normal and fall-example videos across every day and trial."""
    dataset = DatasetIndex()
    normal: list[TrialVideo] = []
    fall_examples: list[TrialVideo] = []
    for day in dataset.days:
        for trial in (1, 2, 3):
            normal_matches = [video for video in dataset.videos if video.dataset == MOUSE_TRACKING_DATASET and video.day == day and video.trial == trial]
            fall_matches = [video for video in dataset.videos if video.dataset == FALL_EXAMPLE_DATASET and video.day == day and video.trial == trial]
            normal.extend(evenly_spaced_videos(normal_matches, normal_per_group) if normal_matches else [])
            fall_examples.extend(evenly_spaced_videos(fall_matches, fall_per_group) if fall_matches else [])
    if not normal:
        raise ValueError(f"No videos found for {MOUSE_TRACKING_DATASET}.")
    if not fall_examples:
        raise ValueError(f"No videos found for {FALL_EXAMPLE_DATASET}.")
    return normal, fall_examples


def fall_phase_counts(total_frames: int) -> tuple[int, int, int]:
    """Keep fall candidates spread across the trial, with most late in the video."""
    if total_frames < 3:
        raise ValueError("Fall-example videos need at least 3 frames per video.")
    early = max(1, round(total_frames * 0.25))
    late = max(2, round(total_frames * 0.375))
    middle = total_frames - early - late
    if middle < 1:
        middle = 1
        late = total_frames - early - middle
    return early, middle, late


def write_mouse_manifest(path: Path, normal_videos: list[TrialVideo], fall_videos: list[TrialVideo], normal_frames: int, fall_frames: int) -> None:
    early, middle, late = fall_phase_counts(fall_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "day", "trial", "purpose", "source_video", "labeled_folder", "frames_to_label", "sampling"))
        writer.writeheader()
        for video in normal_videos:
            writer.writerow({"dataset": video.dataset, "day": video.day, "trial": video.trial, "purpose": "normal_run", "source_video": video.relative_path, "labeled_folder": video.path.stem, "frames_to_label": normal_frames, "sampling": "full video; k-means diversity selection"})
        for video in fall_videos:
            writer.writerow({"dataset": video.dataset, "day": video.day, "trial": video.trial, "purpose": "fall_example", "source_video": video.relative_path, "labeled_folder": video.path.stem, "frames_to_label": fall_frames, "sampling": f"0-55%: {early}; 55-90%: {middle}; 90-100%: {late}"})


def project_video_paths(config_path: Path, videos: list[TrialVideo]) -> list[str]:
    project_videos = config_path.parent.resolve() / "videos"
    return [str((project_videos / video.path.name).resolve()) for video in videos]


def extract_mouse_phase(deeplabcut, auxiliaryfunctions, config_path: Path, videos: list[TrialVideo], frames: int, start: float, stop: float) -> None:
    if not videos or frames <= 0:
        return
    config = auxiliaryfunctions.read_config(str(config_path))
    config["numframes2pick"] = frames
    config["start"] = start
    config["stop"] = stop
    auxiliaryfunctions.write_config(str(config_path), config)
    failed = deeplabcut.extract_frames(str(config_path), mode="automatic", algo="kmeans", crop=False, cluster_step=1, userfeedback=False, videos_list=project_video_paths(config_path, videos))
    if failed and any(failed):
        raise RuntimeError("DeepLabCut frame extraction did not finish.")


def prepare_mouse_batch(args) -> int:
    try:
        normal_videos, fall_videos = select_mouse_batch(args.normal_videos_per_group, args.fall_videos_per_group)
        if args.fall_batch_start < 0 or args.fall_batch_size is not None and args.fall_batch_size < 1:
            raise ValueError("Fall batch start must be non-negative and batch size must be at least 1.")
        if args.fall_batch_size is not None:
            fall_videos = fall_videos[args.fall_batch_start : args.fall_batch_start + args.fall_batch_size]
        if not fall_videos:
            raise ValueError("The requested fall-example batch is empty.")
        fall_phase_counts(args.fall_frames_per_video)
    except ValueError as exc:
        print(exc)
        return 1
    write_mouse_manifest(args.manifest, normal_videos, fall_videos, args.normal_frames_per_video, args.fall_frames_per_video)
    total_frames = (len(normal_videos) * args.normal_frames_per_video) + (len(fall_videos) * args.fall_frames_per_video)
    print(f"Saved mouse-tracking manifest: {args.manifest}")
    print(f"Selected {len(normal_videos)} normal-run videos x {args.normal_frames_per_video} frames.")
    print(f"Selected {len(fall_videos)} fall-example videos x {args.fall_frames_per_video} frames.")
    print(f"Planned total: {total_frames} candidate frames. Training split: {args.training_fraction:.0%}/{1 - args.training_fraction:.0%}.")
    if not args.extract:
        print("Review the manifest, then rerun with --extract to add these frames to the DLC project.")
        return 0
    deeplabcut = load_deeplabcut()
    if deeplabcut is None or not args.config.exists():
        print(f"Missing config file: {args.config}")
        return 1
    if not 0 < args.training_fraction < 1:
        print("--training-fraction must be greater than 0 and less than 1.")
        return 1
    from deeplabcut.utils import auxiliaryfunctions
    if args.reset_project:
        try:
            reset_mouse_project(args.config, auxiliaryfunctions, args.training_fraction, args.normal_frames_per_video)
        except RuntimeError as exc:
            print(exc)
            return 1
        print("Reset the unlabelled mouse-DLC project registration.")
    labeled_data = args.config.parent / "labeled-data"
    normal_to_extract = [video for video in normal_videos if not list((labeled_data / video.path.stem).glob("*.png"))]
    fall_to_extract = [video for video in fall_videos if not list((labeled_data / video.path.stem).glob("*.png"))]
    videos_to_add = normal_to_extract + fall_to_extract
    if not videos_to_add:
        print("All selected mouse-tracking folders already contain frames; nothing was overwritten.")
        return 0
    config = auxiliaryfunctions.read_config(str(args.config))
    registered_names = {Path(path).name for path in config.get("video_sets", {})}
    new_videos = [video for video in videos_to_add if video.path.name not in registered_names]
    if new_videos:
        deeplabcut.add_new_videos(str(args.config), [str(video.path) for video in new_videos], copy_videos=True)
    early, middle, late = fall_phase_counts(args.fall_frames_per_video)
    try:
        extract_mouse_phase(deeplabcut, auxiliaryfunctions, args.config, normal_to_extract, args.normal_frames_per_video, 0.0, 1.0)
        extract_mouse_phase(deeplabcut, auxiliaryfunctions, args.config, fall_to_extract, early, 0.0, 0.55)
        extract_mouse_phase(deeplabcut, auxiliaryfunctions, args.config, fall_to_extract, middle, 0.55, 0.90)
        extract_mouse_phase(deeplabcut, auxiliaryfunctions, args.config, fall_to_extract, late, 0.90, 1.0)
    except RuntimeError as exc:
        print(exc)
        return 1
    finally:
        config = auxiliaryfunctions.read_config(str(args.config))
        config["bodyparts"] = list(TRACKING_BODYPARTS)
        config["TrainingFraction"] = [args.training_fraction]
        config["numframes2pick"] = args.normal_frames_per_video
        config["start"] = 0.0
        config["stop"] = 1.0
        auxiliaryfunctions.write_config(str(args.config), config)
    print(f"Extracted frames for {len(normal_to_extract)} normal-run and {len(fall_to_extract)} fall-example videos.")
    print("In Label Frames, prioritize true fall frames from the SEONG folders and leave obscured points blank.")
    return 0


def reset_mouse_project(config_path: Path, auxiliaryfunctions, training_fraction: float, normal_frames: int) -> None:
    """Remove an unlabelled mouse project registration without touching source data."""
    project_root = config_path.parent.resolve()
    videos_dir = project_root / "videos"
    labeled_data_dir = project_root / "labeled-data"
    for folder in labeled_data_dir.iterdir() if labeled_data_dir.exists() else ():
        if folder.is_dir() and any(folder.iterdir()):
            raise RuntimeError(f"Refusing to reset: {folder.name} contains label files.")

    for video in videos_dir.iterdir() if videos_dir.exists() else ():
        if video.is_file():
            video.unlink()
    for folder in labeled_data_dir.iterdir() if labeled_data_dir.exists() else ():
        if folder.is_dir():
            folder.rmdir()

    config = auxiliaryfunctions.read_config(str(config_path))
    config["video_sets"] = {}
    config["bodyparts"] = list(TRACKING_BODYPARTS)
    config["TrainingFraction"] = [training_fraction]
    config["numframes2pick"] = normal_frames
    config["start"] = 0.0
    config["stop"] = 1.0
    auxiliaryfunctions.write_config(str(config_path), config)
def main() -> int:
    parser = argparse.ArgumentParser(description="RBT DeepLabCut tracking workflow")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create a paw/tail DeepLabCut project")
    create.add_argument("--project-name", default="RBT_visible_front_back_tail")
    create.add_argument("--scorer", default=DEFAULT_SCORER)
    create.add_argument("--working-directory", type=Path, default=DEFAULT_PROJECT_DIR)
    create.add_argument("--use-symlinks", action="store_true")
    add_video_filters(create)
    create.set_defaults(day="D30", trial=1, handler=create_project)

    train = commands.add_parser("train", help="train and evaluate the model")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--scorer", default=DEFAULT_SCORER)
    add_training_options(train)
    train.set_defaults(handler=train_model)

    analyze = commands.add_parser("analyze", help="create CSV predictions for review videos")
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, default=DEFAULT_DLC_PREDICTIONS_DIR)
    analyze.add_argument("--shuffle", type=int, default=1)
    add_video_filters(analyze)
    analyze.set_defaults(handler=analyze_videos)

    analyze_one = commands.add_parser("analyze-video", help="create a prediction CSV for one exact review video")
    analyze_one.add_argument("--config", type=Path, required=True)
    analyze_one.add_argument("--video", type=Path, required=True)
    analyze_one.add_argument("--output-dir", type=Path, default=DEFAULT_DLC_PREDICTIONS_DIR)
    analyze_one.add_argument("--shuffle", type=int, default=1)
    analyze_one.set_defaults(handler=analyze_one_video)

    analyze_files = commands.add_parser("analyze-files", help="create prediction CSVs for explicit review videos")
    analyze_files.add_argument("--config", type=Path, required=True)
    analyze_files.add_argument("--video", type=Path, action="append", required=True)
    analyze_files.add_argument("--output-dir", type=Path, default=DEFAULT_DLC_PREDICTIONS_DIR)
    analyze_files.add_argument("--shuffle", type=int, default=1)
    analyze_files.set_defaults(handler=analyze_exact_videos)

    mouse_batch = commands.add_parser("mouse-batch", help="prepare a balanced paw/tail labeling batch with fall examples")
    mouse_batch.add_argument("--config", type=Path, required=True)
    mouse_batch.add_argument("--normal-videos-per-group", type=int, default=2)
    mouse_batch.add_argument("--fall-videos-per-group", type=int, default=2)
    mouse_batch.add_argument("--normal-frames-per-video", type=int, default=5)
    mouse_batch.add_argument("--fall-frames-per-video", type=int, default=8)
    mouse_batch.add_argument("--fall-batch-start", type=int, default=0, help="zero-based offset for a resumable fall-example batch")
    mouse_batch.add_argument("--fall-batch-size", type=int, help="optional number of fall-example videos to extract in this run")
    mouse_batch.add_argument("--training-fraction", type=float, default=0.95)
    mouse_batch.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "mouse_tracking_batch.csv")
    mouse_batch.add_argument("--extract", action="store_true", help="add selected videos and extract frames without overwriting existing folders")
    mouse_batch.add_argument("--reset-project", action="store_true", help="remove only empty registered mouse videos and label folders before extraction")
    mouse_batch.set_defaults(handler=prepare_mouse_batch)
    tick_batch = commands.add_parser("tick-batch", help="prepare an organized tick-labeling batch")
    tick_batch.add_argument("--config", type=Path, required=True)
    tick_batch.add_argument("--videos-per-group", type=int, default=3)
    tick_batch.add_argument("--frames-per-video", type=int, default=2)
    tick_batch.add_argument("--early-fraction", type=float, default=0.05, help="fraction of each video reserved for tick calibration frames")
    tick_batch.add_argument("--manifest", type=Path, default=ROOT / "outputs" / "tick_batch.csv")
    tick_batch.add_argument("--extract", action="store_true", help="extract frames after writing the manifest")
    tick_batch.set_defaults(handler=prepare_tick_batch)

    tick_analyze = commands.add_parser("tick-analyze", help="run the trained tick model on one video")
    tick_analyze.add_argument("--config", type=Path, required=True)
    tick_analyze.add_argument("--video", type=Path, required=True)
    tick_analyze.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "dlc_tick_predictions")
    tick_analyze.add_argument("--shuffle", type=int, default=1)
    tick_analyze.add_argument("--early-frames", type=int, default=10, help="number of video-wide candidate frames used for calibration")
    tick_analyze.set_defaults(handler=analyze_tick_video)
    report = commands.add_parser("tick-report", help="create tick graphs from a DeepLabCut CSV")
    report.add_argument("--csv", type=Path, required=True)
    report.add_argument("--output-dir", type=Path)
    report.add_argument("--pcutoff", type=float, default=0.2)
    report.set_defaults(handler=plot_tick_report)
    test = commands.add_parser("test", help="test the DLC CSV reader and automatic tracking rules without DeepLabCut")
    test.set_defaults(handler=test_csv_parser)

    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
