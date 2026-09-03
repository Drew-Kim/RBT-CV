from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from rbtcv.condition_map import ConditionMapStore


class ConditionMapStoreTests(unittest.TestCase):
    def test_assignments_are_dataset_scoped_and_can_be_replaced_or_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ConditionMapStore(Path(folder))
            store.update_many("Dataset A", [("10", "2"), ("1", "4")], "sham")
            store.update_many("Dataset A", [("10", "2")], "STROKE")

            self.assertEqual(
                store.load("Dataset A"),
                {("1", "4"): "SHAM", ("10", "2"): "STROKE"},
            )
            self.assertEqual(store.load("Dataset B"), {})

            with store.path_for_dataset("Dataset A").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["cage"], "1")
            self.assertEqual(rows[1]["cage"], "10")

            store.update_many("Dataset A", [("1", "4")], None)
            self.assertEqual(store.load("Dataset A"), {("10", "2"): "STROKE"})

    def test_invalid_condition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ConditionMapStore(Path(folder))
            with self.assertRaises(ValueError):
                store.update_many("Dataset", [("1", "1")], "control")

