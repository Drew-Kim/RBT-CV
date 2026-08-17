from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
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
SPEED_TABLE_ROW = 30


class ResultsWorkbookError(RuntimeError):
    """Raised when the results workbook cannot be read or safely written."""


class ResultsWorkbook:
    """Maintain ordered Time, Distance, Speed, and audit tables for each dataset."""

    def __init__(self, results_root: Path = RESULTS_ROOT) -> None:
        self.results_root = results_root

    def path_for_dataset(self, dataset: str) -> Path:
        return self.results_root / f"{dataset} Results" / RESULTS_FILENAME

    def save(self, annotation: TrialAnnotation) -> Path:
        return self.save_many((annotation,))[annotation.dataset]

    def save_many(self, annotations: Iterable[TrialAnnotation]) -> dict[str, Path]:
        """Upsert a result batch with one workbook write per dataset.

        This is used by animal/day analysis so recalculation of the Speed table
        and the Excel save happen once instead of once for every video.
        """
        by_dataset: dict[str, list[TrialAnnotation]] = defaultdict(list)
        for annotation in annotations:
            by_dataset[annotation.dataset].append(annotation)

        saved_paths: dict[str, Path] = {}
        for dataset, dataset_annotations in by_dataset.items():
            path = self.path_for_dataset(dataset)
            workbook = self._load_or_create(path)
            sheet = self._time_sheet(workbook)

            self._clear_speed_tables(sheet)
            self._ensure_result_tables(sheet)
            for annotation in dataset_annotations:
                self._write_measurement(sheet, 1, 1, annotation, annotation.crossing_time)
                self._write_measurement(
                    sheet,
                    1,
                    DISTANCE_TABLE_COLUMN,
                    annotation,
                    annotation.distance_cm,
                )
                self._upsert_audit_row(self._audit_sheet(workbook), annotation)
            self._rebuild_speed_table(sheet)

            self._save_workbook(
                workbook,
                path,
                "Close it in Excel and run the analysis again.",
            )
            saved_paths[dataset] = path
        return saved_paths

    def _load_or_create(self, path: Path):
        try:
            from openpyxl import Workbook, load_workbook
            from openpyxl.utils.exceptions import InvalidFileException
            from zipfile import BadZipFile
        except ImportError as exc:
            raise ResultsWorkbookError(
                "Excel support is missing. Install the project requirements."
            ) from exc

        if path.exists():
            try:
                return load_workbook(path)
            except (OSError, InvalidFileException, BadZipFile) as exc:
                raise ResultsWorkbookError(
                    f"Could not open {path.name}. Close it in Excel and try Save Trial Result again."
                ) from exc

        workbook = Workbook()
        workbook.active.title = TIME_SHEET
        self._ensure_result_tables(workbook.active)
        return workbook

    @staticmethod
    def _save_workbook(workbook, path: Path, instruction: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
        try:
            workbook.save(temporary_path)
            temporary_path.replace(path)
        except OSError as exc:
            raise ResultsWorkbookError(
                f"Could not save {path.name}. {instruction}"
            ) from exc
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

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

    @staticmethod
    def _ensure_table_headers(
        sheet,
        title: str,
        title_row: int,
        start_column: int,
    ) -> None:
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

    def _animal_row(
        self,
        sheet,
        cage: str,
        subject: str,
        title_row: int,
        start_column: int,
    ) -> int:
        """Find or create an ordered animal row in one result table."""
        first_row = title_row + 3
        last_column = self._last_table_column(sheet, title_row, start_column)
        records = self._table_records(sheet, first_row, last_column, start_column)

        target = (self._normalized(cage), self._normalized(subject))
        if not any(self._record_key(record) == target for record in records):
            records.append(
                [
                    int(target[0]) if target[0].isdigit() else target[0],
                    int(target[1]) if target[1].isdigit() else target[1],
                    *([None] * (last_column - start_column - 1)),
                ]
            )

        records.sort(key=lambda record: self._animal_sort_key(record[0], record[1]))
        for row, record in enumerate(records, start=first_row):
            for offset, value in enumerate(record):
                sheet.cell(row=row, column=start_column + offset, value=value)

        return next(
            row
            for row, record in enumerate(records, start=first_row)
            if self._record_key(record) == target
        )

    @staticmethod
    def _table_records(sheet, first_row: int, last_column: int, start_column: int) -> list[list[object]]:
        records = []
        row = first_row
        while (
            sheet.cell(row=row, column=start_column).value is not None
            or sheet.cell(row=row, column=start_column + 1).value is not None
        ):
            records.append(
                [
                    sheet.cell(row=row, column=column).value
                    for column in range(start_column, last_column + 1)
                ]
            )
            row += 1
        return records

    @staticmethod
    def _record_key(record: list[object]) -> tuple[str, str]:
        return (
            ResultsWorkbook._normalized(record[0]),
            ResultsWorkbook._normalized(record[1]),
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
            header = str(sheet.cell(row=header_row, column=column).value or "").strip()
            if not header.upper().startswith("T"):
                break
            last_column = column
        return last_column

    @staticmethod
    def _table_last_row(sheet, title_row: int, start_column: int) -> int:
        row = title_row + 3
        while (
            sheet.cell(row=row, column=start_column).value is not None
            or sheet.cell(row=row, column=start_column + 1).value is not None
        ):
            row += 1
        return max(title_row + 2, row - 1)

    @staticmethod
    def _trial_columns(
        sheet,
        title_row: int,
        start_column: int,
    ) -> dict[tuple[str, int], int]:
        columns: dict[tuple[str, int], int] = {}
        active_day = ""
        header_row = title_row + 2
        last_column = ResultsWorkbook._last_table_column(sheet, title_row, start_column)
        for column in range(start_column + 2, last_column + 1):
            day_value = sheet.cell(row=title_row + 1, column=column).value
            if day_value is not None and str(day_value).strip():
                active_day = str(day_value).strip()
            trial = str(sheet.cell(row=header_row, column=column).value or "").strip().upper()
            if active_day and trial.startswith("T") and trial[1:].isdigit():
                columns[(active_day, int(trial[1:]))] = column
        return columns

    @staticmethod
    def _table_rows_by_subject(
        sheet,
        title_row: int,
        start_column: int,
    ) -> dict[tuple[str, str], int]:
        first_row = title_row + 3
        last_row = ResultsWorkbook._table_last_row(sheet, title_row, start_column)
        return {
            (
                ResultsWorkbook._normalized(sheet.cell(row=row, column=start_column).value),
                ResultsWorkbook._normalized(sheet.cell(row=row, column=start_column + 1).value),
            ): row
            for row in range(first_row, last_row + 1)
        }

    @staticmethod
    def _numeric(value: object, *, allow_zero: bool = False) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number > 0 or (allow_zero and number == 0):
            return number
        return None

    @staticmethod
    def _clear_speed_tables(sheet) -> None:
        title_rows = [
            row
            for row in range(1, sheet.max_row + 1)
            if str(sheet.cell(row=row, column=1).value or "").strip().upper()
            == "SPEED (CM/S)"
        ]
        for title_row in title_rows:
            last_row = ResultsWorkbook._table_last_row(sheet, title_row, 1)
            last_column = ResultsWorkbook._last_table_column(sheet, title_row, 1)
            for row in range(title_row, last_row + 1):
                for column in range(1, last_column + 1):
                    sheet.cell(row=row, column=column).value = None

    @staticmethod
    def _speed_title_row(sheet) -> int:
        return max(
            SPEED_TABLE_ROW,
            ResultsWorkbook._table_last_row(sheet, 1, 1) + 3,
        )

    def _rebuild_speed_table(self, sheet) -> None:
        speed_title_row = self._speed_title_row(sheet)
        self._ensure_table_headers(sheet, "SPEED (cm/s)", speed_title_row, 1)

        time_columns = self._trial_columns(sheet, 1, 1)
        distance_columns = self._trial_columns(sheet, 1, DISTANCE_TABLE_COLUMN)
        time_rows = self._table_rows_by_subject(sheet, 1, 1)
        distance_rows = self._table_rows_by_subject(sheet, 1, DISTANCE_TABLE_COLUMN)

        for column in range(3, self._last_table_column(sheet, 1, 1) + 1):
            sheet.cell(
                row=speed_title_row + 1,
                column=column,
                value=sheet.cell(row=2, column=column).value,
            )
            sheet.cell(
                row=speed_title_row + 2,
                column=column,
                value=sheet.cell(row=3, column=column).value,
            )

        for offset, (subject_key, time_row) in enumerate(time_rows.items()):
            speed_row = speed_title_row + 3 + offset
            cage, subject = subject_key
            sheet.cell(row=speed_row, column=1, value=sheet.cell(row=time_row, column=1).value)
            sheet.cell(row=speed_row, column=2, value=sheet.cell(row=time_row, column=2).value)

            distance_row = distance_rows.get((cage, subject))
            for trial_key, time_column in time_columns.items():
                time_value = self._numeric(
                    sheet.cell(row=time_row, column=time_column).value
                )
                distance_column = distance_columns.get(trial_key)
                distance_value = (
                    self._numeric(
                        sheet.cell(row=distance_row, column=distance_column).value,
                        allow_zero=True,
                    )
                    if distance_row is not None and distance_column is not None
                    else None
                )
                cell = sheet.cell(row=speed_row, column=time_column)
                cell.value = (
                    round(distance_value / time_value, 2)
                    if time_value is not None and distance_value is not None
                    else None
                )
                cell.number_format = "0.00"

    @staticmethod
    def _trial_column(
        sheet,
        day: str,
        trial: int,
        title_row: int,
        start_column: int,
    ) -> int:
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

    def _write_measurement(
        self,
        sheet,
        title_row: int,
        start_column: int,
        annotation: TrialAnnotation,
        value: float | int | None,
    ) -> None:
        row = self._animal_row(
            sheet,
            annotation.group,
            annotation.subject,
            title_row,
            start_column,
        )
        column = self._trial_column(
            sheet,
            annotation.day,
            annotation.trial,
            title_row,
            start_column,
        )
        cell = sheet.cell(
            row=row,
            column=column,
            value=round(value, 2) if value is not None else None,
        )
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
        headers = [
            sheet.cell(row=1, column=column).value
            for column in range(1, sheet.max_column + 1)
        ]
        relative_video_column = headers.index("relative_video") + 1
        target_row = next(
            (
                row
                for row in range(2, sheet.max_row + 1)
                if sheet.cell(row=row, column=relative_video_column).value
                == annotation.relative_video
            ),
            sheet.max_row + 1,
        )
        for column, name in enumerate(headers, start=1):
            sheet.cell(row=target_row, column=column, value=values.get(name, ""))
