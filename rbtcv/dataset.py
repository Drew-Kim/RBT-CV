from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
PREFERRED_DATASETS = ("RBT DATA", "RBT DATA_B")

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
        return match.group(0) if match else self.group

    @property
    def rat_id(self) -> str:
        return self.subject

    @property
    def survivor_key(self) -> tuple[str, str]:
        return (self.cage_number, self.rat_id)

    @property
    def subject_key(self) -> str:
        return f"{self.day}|{self.cage_number}_{self.rat_id}"

    @property
    def subject_label(self) -> str:
        return f"{self.day}  Cage {self.cage_number} Rat {self.rat_id} ({self.group}_{self.subject})"

    @property
    def trial_label(self) -> str:
        return f"T{self.trial}"

    @property
    def sort_key(self) -> tuple[str, str, int, str, int]:
        return (self.day, self.cage_number, int(self.rat_id), self.date, self.trial)


def find_dataset_dir(data_root: Path = DATA_ROOT) -> Path:
    if not data_root.exists():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    candidates = [data_root / name for name in PREFERRED_DATASETS]
    candidates.extend(path for path in sorted(data_root.iterdir()) if path.is_dir())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and any(candidate.rglob("*.avi")):
            return candidate

    raise FileNotFoundError(f"No dataset folder with AVI files found under: {data_root}")


def parse_trial_video(path: Path, dataset_dir: Path, root: Path = ROOT) -> TrialVideo | None:
    match = VIDEO_RE.match(path.name)
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
        self.dataset_dir = dataset_dir or find_dataset_dir()
        self.all_videos = self._load_all_videos()
        self.survivor_keys = self._find_d30_survivors(self.all_videos)
        self.videos = self._filter_survivor_videos(self.all_videos)

    def _load_all_videos(self) -> list[TrialVideo]:
        videos: list[TrialVideo] = []
        for path in sorted(self.dataset_dir.rglob("*.avi")):
            trial_video = parse_trial_video(path, self.dataset_dir)
            if trial_video is not None:
                videos.append(trial_video)
        return sorted(videos, key=lambda item: item.sort_key)

    def _find_d30_survivors(self, videos: list[TrialVideo]) -> set[tuple[str, str]]:
        return {video.survivor_key for video in videos if video.day.upper() == D30_DAY}

    def _filter_survivor_videos(self, videos: list[TrialVideo]) -> list[TrialVideo]:
        if not self.survivor_keys:
            return videos
        return sorted(
            [video for video in videos if video.survivor_key in self.survivor_keys],
            key=lambda item: item.sort_key,
        )

    @property
    def days(self) -> list[str]:
        return sorted({video.day for video in self.videos}, key=natural_day_key)

    def subjects_for_day(self, day: str) -> list[str]:
        labels = {video.subject_key: video.subject_label for video in self.videos if video.day == day}
        return [labels[key] for key in sorted(labels, key=subject_key_sort)]

    def subject_key_from_label(self, label: str) -> str:
        day, subject_label = label.split(None, 1)
        match = re.search(r"Cage\s+(\d+)\s+Rat\s+(\d+)", subject_label)
        if match:
            cage, rat_id = match.groups()
            return f"{day}|{cage}_{rat_id}"
        return f"{day}|{subject_label.strip()}"

    def trials_for_subject(self, subject_key: str) -> dict[int, TrialVideo]:
        return {video.trial: video for video in self.videos if video.subject_key == subject_key}


def natural_day_key(day: str) -> tuple[int, str]:
    if day.upper() == "BL":
        return (0, day)
    match = re.search(r"\d+", day)
    if match:
        return (int(match.group()), day)
    return (9999, day)


def subject_key_sort(subject_key: str) -> tuple[tuple[int, str], str, int]:
    day, subject = subject_key.split("|", 1)
    cage, rat_id = subject.rsplit("_", 1)
    return (natural_day_key(day), cage, int(rat_id))
