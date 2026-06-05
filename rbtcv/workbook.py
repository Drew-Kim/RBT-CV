from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import re

from .dataset import ROOT, TrialVideo


MANUAL_TIMES = ROOT / "res" / "RBT_Data_Corrected.xlsb"
MAPPING_FILE = ROOT / "outputs" / "manual_mapping.csv"


@dataclass(frozen=True)
class ManualMatch:
    time_seconds: float
    day: str
    s_no: str
    animal_id: str
    trial: int
    source: str


def _norm(value: object) -> str:
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def _as_float(value: object) -> float | None:
    text = _norm(value)
    if not text or text.startswith("#"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _group_number(group: str) -> str:
    match = re.search(r"\d+", group)
    return match.group(0) if match else group


class ManualTimingStore:
    def __init__(
        self,
        workbook_path: Path = MANUAL_TIMES,
        mapping_path: Path = MAPPING_FILE,
    ) -> None:
        self.workbook_path = workbook_path
        self.mapping_path = mapping_path
        self.times: dict[tuple[str, str, str, int], float] = {}
        self.mapping: dict[tuple[str, str, str], tuple[str, str, str]] = {}
        self.load_errors: list[str] = []
        self._load_workbook()
        self._load_mapping()

    def _load_workbook(self) -> None:
        if not self.workbook_path.exists():
            self.load_errors.append(f"Manual workbook not found: {self.workbook_path}")
            return

        try:
            import pandas as pd

            df = pd.read_excel(self.workbook_path, sheet_name="Forelimb", header=None, engine="pyxlsb")
        except Exception as exc:
            self.load_errors.append(f"Could not read manual workbook: {exc}")
            return

        if df.empty or len(df.index) < 4:
            self.load_errors.append("Manual workbook has no readable timing table.")
            return

        time_end = len(df.columns)
        first_row = [_norm(df.iat[0, col]) for col in range(len(df.columns))]
        for col, value in enumerate(first_row):
            if col > 0 and value.lower() in {"distance", "speed", "parameter"}:
                time_end = col
                break

        day_for_col: dict[int, str] = {}
        current_day = ""
        for col in range(2, time_end):
            day_value = _norm(df.iat[1, col])
            if day_value:
                current_day = day_value
            if current_day:
                day_for_col[col] = current_day

        for row in range(3, len(df.index)):
            s_no = _norm(df.iat[row, 0])
            animal_id = _norm(df.iat[row, 1])
            if not s_no and not animal_id:
                continue

            for col in range(2, time_end):
                trial_text = _norm(df.iat[2, col]).upper()
                if not trial_text.startswith("T"):
                    continue
                try:
                    trial = int(trial_text[1:])
                except ValueError:
                    continue

                seconds = _as_float(df.iat[row, col])
                day = day_for_col.get(col, "")
                if seconds is None or not day:
                    continue
                self.times[(day, s_no, animal_id, trial)] = seconds

    def _load_mapping(self) -> None:
        if not self.mapping_path.exists():
            return

        with self.mapping_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                video_key = (
                    row.get("video_day", "").strip(),
                    row.get("video_group", "").strip().upper(),
                    row.get("video_subject", "").strip(),
                )
                workbook_key = (
                    row.get("workbook_day", "").strip(),
                    row.get("workbook_s_no", "").strip(),
                    row.get("workbook_animal_id", "").strip(),
                )
                if all(video_key) and all(workbook_key):
                    self.mapping[video_key] = workbook_key

    def lookup(self, video: TrialVideo) -> ManualMatch | None:
        video_key = (video.day, video.group, video.subject)
        mapped = self.mapping.get(video_key)
        candidates: list[tuple[str, str, str, str]] = []

        if mapped:
            candidates.append((*mapped, "mapping"))

        group_number = _group_number(video.group)
        candidates.extend(
            [
                (video.day, group_number, video.subject, "direct"),
                (video.day, video.group, video.subject, "direct"),
            ]
        )
        if video.day.upper() == "BL":
            candidates.extend(
                [
                    ("BL1", group_number, video.subject, "baseline fallback"),
                    ("BL2", group_number, video.subject, "baseline fallback"),
                    ("BL", group_number, video.subject, "baseline fallback"),
                ]
            )

        for day, s_no, animal_id, source in candidates:
            seconds = self.times.get((day, s_no, animal_id, video.trial))
            if seconds is not None:
                return ManualMatch(seconds, day, s_no, animal_id, video.trial, source)
        return None
