from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .dataset import ROOT


ANNOTATIONS_FILE = ROOT / "outputs" / "annotations.csv"


@dataclass(frozen=True)
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
    """Persist one current scoring record for each source video."""

    fieldnames = tuple(TrialAnnotation.__dataclass_fields__)

    def __init__(self, path: Path = ANNOTATIONS_FILE) -> None:
        self.path = path

    def load_by_video(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}

        with self.path.open("r", newline="", encoding="utf-8") as handle:
            rows = {
                row["relative_video"]: row
                for row in csv.DictReader(handle)
                if row.get("relative_video")
            }
        return rows

    def save(self, annotation: TrialAnnotation) -> None:
        self.save_many((annotation,))

    def save_many(self, annotations: Iterable[TrialAnnotation]) -> None:
        """Upsert a batch once, so day analysis writes this CSV only once."""
        annotations = tuple(annotations)
        if not annotations:
            return

        rows = self.load_by_video()
        for annotation in annotations:
            rows[annotation.relative_video] = {
                key: str(value) for key, value in asdict(annotation).items()
            }
        self._write_rows(rows)

    def _write_rows(self, rows: dict[str, dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        try:
            with temporary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(rows[key] for key in sorted(rows))
            temporary_path.replace(self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")
