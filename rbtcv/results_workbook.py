from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .dataset import ROOT

if TYPE_CHECKING:
    from .annotations import TrialAnnotation


RESULTS_ROOT = ROOT / "outputs"
RESULTS_FILENAME = "RBT_CV_Results.xlsx"
TIME_SHEET = "Forelimb"
AUDIT_SHEET = "RBT-CV Results"
DISTANCE_TABLE_COLUMN = 26
SPEED_TABLE_ROW = 100


class ResultsWorkbookError(RuntimeError):
    """Raised when the results workbook cannot be read or written."""


class ResultsWorkbook:
    """Write each dataset's results to clean Time, Distance, and Speed tables."""

    def __init__(self, results_root: Path = RESULTS_ROOT) -> None:
        self.results_root = results_root

    def path_for_dataset(self, dataset: str) -> Path:
        return self.results_root / f"{dataset} Results" / RESULTS_FILENAME

    def save(self, annotation: TrialAnnotation) -> Path:
        path = self.path_for_dataset(annotation.dataset)
        workbook = self._load_or_create(path)
        sheet = self._time_sheet(workbook)
        self._ensure_result_tables(sheet)

        self._write_measurement(sheet, 1, 1, annotation, annotation.crossing_time)
        self._write_measurement(sheet, 1, DISTANCE_TABLE_COLUMN, annotation, annotation.distance_cm)
        speed = annotation.distance_cm / annotation.crossing_time if annotation.crossing_time > 0 else None
        self._write_measurement(sheet, SPEED_TABLE_ROW, 1, annotation, speed)

        self._upsert_audit_row(self._audit_sheet(workbook), annotation)
        path.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(path)
        return path

    def create_empty(self, dataset: str) -> Path:
        """Create or replace a dataset's workbook without carrying over old results."""
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            raise ResultsWorkbookError("Excel support is missing. Install the project requirements.") from exc

        path = self.path_for_dataset(dataset)
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = TIME_SHEET
        self._ensure_result_tables(sheet)
        self._audit_sheet(workbook)
        path.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(path)
        return path

    def _load_or_create(self, path: Path):
        try:
            from openpyxl import Workbook, load_workbook
        except ImportError as exc:
            raise ResultsWorkbookError("Excel support is missing. Install the project requirements.") from exc

        if path.exists():
            try:
                return load_workbook(path)
            except OSError as exc:
                raise ResultsWorkbookError(
                    f"Could not open {path.name}. Close it in Excel and try Save Trial Result again."
                ) from exc

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = TIME_SHEET
        self._ensure_result_tables(sheet)
        return workbook

    @staticmethod
    def _time_sheet(workbook):
        if TIME_SHEET in workbook.sheetnames:
            return workbook[TIME_SHEET]
        sheet = workbook.create_sheet(TIME_SHEET, 0)
        ResultsWorkbook._ensure_result_tables(sheet)
        return sheet

    @staticmethod
    def _ensure_result_tables(sheet) -> None:
        ResultsWorkbook._ensure_table_headers(sheet, "TIME (seconds)", 1, 1)
        ResultsWorkbook._ensure_table_headers(sheet, "DISTANCE (cm)", 1, DISTANCE_TABLE_COLUMN)
        ResultsWorkbook._ensure_table_headers(sheet, "SPEED (cm/s)", SPEED_TABLE_ROW, 1)

    @staticmethod
    def _ensure_table_headers(sheet, title: str, title_row: int, start_column: int) -> None:
        if sheet.cell(row=title_row, column=start_column).value is None:
            sheet.cell(row=title_row, column=start_column, value=title)
        header_row = title_row + 2
        if sheet.cell(row=header_row, column=start_column).value is None:
            sheet.cell(row=header_row, column=start_column, value="S.No.")
        if sheet.cell(row=header_row, column=start_column + 1).value is None:
            sheet.cell(row=header_row, column=start_column + 1, value="Animal I.D.")

    @staticmethod
    def _normalized(value: object) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        text = str(value).strip()
        if text.upper().startswith("C") and text[1:].isdigit():
            return str(int(text[1:]))
        return text

    def _animal_row(self, sheet, cage: str, subject: str, title_row: int, start_column: int) -> int:
        """Return an ordered row, moving only this table's existing values if needed."""
        first_row = title_row + 3
        last_column = self._last_table_column(sheet, title_row, start_column)
        records: list[list[object]] = []
        row = first_row
        while (
            sheet.cell(row=row, column=start_column).value is not None
            or sheet.cell(row=row, column=start_column + 1).value is not None
        ):
            records.append(
                [sheet.cell(row=row, column=column).value for column in range(start_column, last_column + 1)]
            )
            row += 1

        target = (self._normalized(cage), self._normalized(subject))
        if not any((self._normalized(record[0]), self._normalized(record[1])) == target for record in records):
            cage_value = int(target[0]) if target[0].isdigit() else target[0]
            subject_value = int(target[1]) if target[1].isdigit() else target[1]
            records.append([cage_value, subject_value, *([None] * (last_column - start_column - 1))])

        records.sort(key=lambda record: self._animal_sort_key(record[0], record[1]))
        for index, record in enumerate(records, start=first_row):
            for offset, value in enumerate(record):
                sheet.cell(row=index, column=start_column + offset, value=value)

        return next(
            index
            for index, record in enumerate(records, start=first_row)
            if (self._normalized(record[0]), self._normalized(record[1])) == target
        )

    @staticmethod
    def _animal_sort_key(cage: object, subject: object) -> tuple[int, int | str, int, int | str]:
        def component(value: object) -> tuple[int, int | str]:
            text = ResultsWorkbook._normalized(value)
            return (0, int(text)) if text.isdigit() else (1, text.casefold())

        return (*component(cage), *component(subject))

    @staticmethod
    def _last_table_column(sheet, title_row: int, start_column: int) -> int:
        header_row = title_row + 2
        last_column = start_column + 1
        for column in range(start_column + 2, sheet.max_column + 1):
            if not str(sheet.cell(row=header_row, column=column).value or "").strip().upper().startswith("T"):
                break
            last_column = column
        return last_column
    @staticmethod
    def _trial_column(sheet, day: str, trial: int, title_row: int, start_column: int) -> int:
        header_row = title_row + 2
        last_column = ResultsWorkbook._last_table_column(sheet, title_row, start_column)


        active_day = ""
        for column in range(start_column + 2, last_column + 1):
            day_value = sheet.cell(row=title_row + 1, column=column).value
            if day_value is not None and str(day_value).strip():
                active_day = str(day_value).strip()
            trial_value = str(sheet.cell(row=header_row, column=column).value or "").strip().upper()
            if active_day.upper() == day.upper() and trial_value == f"T{trial}":
                return column

        first_column = max(start_column + 2, last_column + 1)
        for offset, trial_number in enumerate((1, 2, 3)):
            column = first_column + offset
            if trial_number == 1:
                sheet.cell(row=title_row + 1, column=column, value=day)
            sheet.cell(row=header_row, column=column, value=f"T{trial_number}")
        return first_column + trial - 1

    def _write_measurement(self, sheet, title_row, start_column, annotation, value) -> None:
        row = self._animal_row(sheet, annotation.group, annotation.subject, title_row, start_column)
        column = self._trial_column(sheet, annotation.day, annotation.trial, title_row, start_column)
        cell = sheet.cell(row=row, column=column, value=round(value, 2) if value is not None else None)
        cell.number_format = "0.00"

    @staticmethod
    def _audit_sheet(workbook):
        if AUDIT_SHEET in workbook.sheetnames:
            return workbook[AUDIT_SHEET]

        sheet = workbook.create_sheet(AUDIT_SHEET)
        from .annotations import TrialAnnotation

        for column, name in enumerate(TrialAnnotation.__dataclass_fields__, start=1):
            sheet.cell(row=1, column=column, value=name)
        sheet.freeze_panes = "A2"
        return sheet

    @staticmethod
    def _upsert_audit_row(sheet, annotation: TrialAnnotation) -> None:
        from dataclasses import asdict

        values = asdict(annotation)
        headers = [sheet.cell(row=1, column=column).value for column in range(1, sheet.max_column + 1)]
        relative_video_column = headers.index("relative_video") + 1
        target_row = sheet.max_row + 1
        for row in range(2, sheet.max_row + 1):
            if sheet.cell(row=row, column=relative_video_column).value == annotation.relative_video:
                target_row = row
                break

        for column, name in enumerate(headers, start=1):
            sheet.cell(row=target_row, column=column, value=values.get(name, ""))