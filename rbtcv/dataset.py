from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
PREFERRED_DATASETS = ("RBT DATA_B", "SEONG RBT DATA", "RBT DATA")

VIDEO_RE = re.compile(
    r"^(?P<group>[A-Za-z]\d+)_(?P<subject>\d+)_T(?P<trial>\d+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})-(?P<clock>\d{6})-\d+\.avi$",
    re.IGNORECASE,
)

D30_DAY = "D30"


@dataclass(frozen=True)
class TrialVideo:
    dataset: str
    day: str
    group: str
    subject: str
    trial: int
    date: str
    clock: str
    path: Path
    relative_path: str

    @property

    def cage_number(self) -> str:
        match = re.search(r"\d+", self.group)
        if match:
            return match.group(0)
        return self.group

    @property

    def rat_id(self) -> str:
        return self.subject

    @property

    def survivor_key(self) -> tuple[str, str, str]:
        return (self.dataset, self.cage_number, self.rat_id)

    @property

    def subject_key(self) -> str:
        return f"{self.dataset}|{self.day}|{self.cage_number}_{self.rat_id}"

    @property

    def subject_label(self) -> str:
        return f"{self.dataset} | {self.day}  Cage {self.cage_number} Rat {self.rat_id} ({self.group}_{self.subject})"


    @property

    def sort_key(self) -> tuple[str, tuple[int, str], int, int, str, str, int]:
        return (
            self.dataset,
            natural_day_key(self.day),
            int_or_large(self.cage_number),
            int_or_large(self.rat_id),
            self.date,
            self.clock,
            self.trial,
        )


def dataset_has_videos(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.rglob("*.avi"))


def find_dataset_dirs(data_root: Path = DATA_ROOT) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    # Check expected dataset names first, then any other folder under data/.
    candidates = [data_root / name for name in PREFERRED_DATASETS]
    candidates.extend(path for path in sorted(data_root.iterdir()) if path.is_dir())

    dataset_dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue

        seen.add(resolved)
        if dataset_has_videos(candidate):
            dataset_dirs.append(candidate)

    if dataset_dirs:
        return dataset_dirs

    raise FileNotFoundError(f"No dataset folder with AVI files found under: {data_root}")



def parse_trial_video(path: Path, dataset_dir: Path, root: Path = ROOT) -> TrialVideo | None:
    match = VIDEO_RE.match(path.name)

    # Skip files that do not follow the RBT video naming pattern.
    if not match:
        return None

    parts = match.groupdict()
    return TrialVideo(
        dataset=dataset_dir.name,
        day=path.parent.name,
        group=parts["group"].upper(),
        subject=parts["subject"],
        trial=int(parts["trial"]),
        date=parts["date"],
        clock=parts["clock"],
        path=path,
        relative_path=str(path.relative_to(root)),
    )


class DatasetIndex:

    def __init__(self, dataset_dir: Path | None = None) -> None:
        self.source_dir = dataset_dir
        if dataset_dir is None:
            self.dataset_dirs = find_dataset_dirs()
        elif dataset_dir.name == DATA_ROOT.name:
            self.dataset_dirs = find_dataset_dirs(dataset_dir)
        else:
            self.dataset_dirs = [dataset_dir]


        self.all_videos = self._load_all_videos()
        self.survivor_keys = self._find_d30_survivors(self.all_videos)
        self.evaluated_subject_keys = self._find_baseline_and_d30_subjects(self.all_videos)
        self.videos = self._filter_evaluated_videos(self.all_videos)

    @property
    def dataset_dir(self) -> Path:
        """Compatibility alias for single-dataset GUI callers."""
        return self.source_dir or self.dataset_dirs[0]

    def subject_key_from_label(self, label: object) -> str | None:
        """Accept both the current (key, label) entries and older string labels."""
        if isinstance(label, tuple) and label:
            return str(label[0])
        text = str(label)
        for key, display in self.subjects_for_day(self.days[0] if self.days else ""):
            if text == display or text == key:
                return key
        return text if "|" in text else None
    @property

    def label(self) -> str:
        labels: list[str] = []
        for path in self.dataset_dirs:
            try:
                labels.append(str(path.relative_to(ROOT)))
            except ValueError:
                labels.append(str(path))
        return ", ".join(labels)

    def _load_all_videos(self) -> list[TrialVideo]:
        videos: list[TrialVideo] = []
        for dataset_dir in self.dataset_dirs:
            for path in sorted(dataset_dir.rglob("*.avi")):
                trial_video = parse_trial_video(path, dataset_dir)

                # Only include files that parse into cage, subject, trial, and day.
                if trial_video is not None:
                    videos.append(trial_video)

        return sorted(videos, key=lambda item: item.sort_key)

    def _find_d30_survivors(self, videos: list[TrialVideo]) -> set[tuple[str, str, str]]:
        survivors: set[tuple[str, str, str]] = set()
        for video in videos:
            # Any animal with a D30 video is counted as a survivor.
            if video.day.upper() == D30_DAY:
                survivors.add(video.survivor_key)
        return survivors

    @staticmethod
    def _find_baseline_and_d30_subjects(videos: list[TrialVideo]) -> set[tuple[str, str, str]]:
        """Return only subjects represented in both BL and D30 folders."""
        days_by_subject: dict[tuple[str, str, str], set[str]] = {}
        for video in videos:
            days_by_subject.setdefault(video.survivor_key, set()).add(video.day.upper())
        return {
            subject_key
            for subject_key, days in days_by_subject.items()
            if {"BL", D30_DAY}.issubset(days)
        }

    def _filter_evaluated_videos(self, videos: list[TrialVideo]) -> list[TrialVideo]:
        """Keep trials only for animals eligible for the longitudinal analysis."""
        return sorted(
            (
                video
                for video in videos
                if video.survivor_key in self.evaluated_subject_keys
            ),
            key=lambda item: item.sort_key,
        )

    @property

    def days(self) -> list[str]:
        return sorted({video.day for video in self.videos}, key=natural_day_key)

    def subjects_for_day(self, day: str) -> list[tuple[str, str]]:
        labels = {
            video.subject_key: video.subject_label
            for video in self.videos
            if video.day == day
        }
        return [(key, labels[key]) for key in sorted(labels, key=subject_key_sort)]

    def trials_for_subject(self, subject_key: str) -> dict[int, TrialVideo]:
        trials: dict[int, TrialVideo] = {}
        for video in self.videos:
            if video.subject_key == subject_key:
                trials[video.trial] = video
        return trials


def natural_day_key(day: str) -> tuple[int, str]:
    # Baseline should sort before D3, D9, D14, and later day folders.
    if day.upper() == "BL":
        return (0, day)

    match = re.search(r"\d+", day)
    if match:
        return (int(match.group()), day)

    return (9999, day)


def int_or_large(value: str) -> int:
    match = re.search(r"\d+", value)
    if match:
        return int(match.group(0))
    return 9999


def subject_key_sort(subject_key: str) -> tuple[str, tuple[int, str], int, int]:
    parts = subject_key.split("|")
    if len(parts) == 3:
        dataset, day, subject = parts
    else:
        dataset = ""
        day, subject = parts

    cage, rat_id = subject.rsplit("_", 1)
    return (dataset, natural_day_key(day), int_or_large(cage), int_or_large(rat_id))
