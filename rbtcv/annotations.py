from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import csv

from .dataset import ROOT


ANNOTATIONS_FILE = ROOT / "outputs" / "annotations.csv"


@dataclass
class TrialAnnotation:
    relative_video: str
    dataset: str
    day: str
    group: str
    subject: str
    trial: int
    fps: float
    start_frame: int
    start_time: float
    start_x: int
    start_y: int
    stop_frame: int
    stop_time: float
    stop_x: int
    stop_y: int
    crossing_time: float
    outcome: str
    distance_cm: int
    max_time_applied: str
    saved_at: str


class AnnotationStore:
    fieldnames = list(TrialAnnotation.__dataclass_fields__.keys())

    def __init__(self, path: Path = ANNOTATIONS_FILE) -> None:
        self.path = path

    def load_by_video(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}

        rows: dict[str, dict[str, str]] = {}
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                relative_video = row.get("relative_video", "")
                if relative_video:
                    rows[relative_video] = row
        return rows

    def save(self, annotation: TrialAnnotation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.load_by_video()
        rows[annotation.relative_video] = {key: str(value) for key, value in asdict(annotation).items()}

        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for key in sorted(rows):
                writer.writerow(rows[key])


def now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
