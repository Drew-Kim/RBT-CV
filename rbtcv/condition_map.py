"""Persistent SHAM/STROKE assignments for unique cage-and-rat subjects."""

from __future__ import annotations

import csv
from pathlib import Path

from .dataset import ROOT


OUTPUT_ROOT = ROOT / "outputs"
FILENAME = "condition_map.csv"
VALID_CONDITIONS = frozenset({"SHAM", "STROKE"})
FIELDNAMES = ("dataset", "cage", "animal", "condition")


class ConditionMapStore:
    """Keep one editable condition map per dataset results folder."""

    def __init__(self, output_root: Path = OUTPUT_ROOT) -> None:
        self.output_root = output_root

    def path_for_dataset(self, dataset: str) -> Path:
        return self.output_root / f"{dataset} Results" / FILENAME

    def load(self, dataset: str) -> dict[tuple[str, str], str]:
        path = self.path_for_dataset(dataset)
        if not path.exists():
            return {}
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            return {
                (str(row["cage"]).strip(), str(row["animal"]).strip()): condition
                for row in rows
                if str(row.get("dataset", "")).strip() == dataset
                and (condition := str(row.get("condition", "")).strip().upper()) in VALID_CONDITIONS
                and str(row.get("cage", "")).strip()
                and str(row.get("animal", "")).strip()
            }

    def update_many(
        self,
        dataset: str,
        subjects: list[tuple[str, str]],
        condition: str | None,
    ) -> None:
        if condition is not None:
            condition = condition.strip().upper()
            if condition not in VALID_CONDITIONS:
                raise ValueError(f"Condition must be one of: {', '.join(sorted(VALID_CONDITIONS))}.")

        assignments = self.load(dataset)
        for cage, animal in subjects:
            key = (str(cage).strip(), str(animal).strip())
            if condition is None:
                assignments.pop(key, None)
            else:
                assignments[key] = condition
        self._write(dataset, assignments)

    def _write(self, dataset: str, assignments: dict[tuple[str, str], str]) -> None:
        path = self.path_for_dataset(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            with temporary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                for (cage, animal), condition in sorted(
                    assignments.items(),
                    key=lambda item: (_numeric_sort(item[0][0]), _numeric_sort(item[0][1])),
                ):
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "cage": cage,
                            "animal": animal,
                            "condition": condition,
                        }
                    )
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def _numeric_sort(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value.casefold())
