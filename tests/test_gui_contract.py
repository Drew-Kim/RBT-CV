from __future__ import annotations

import ast
import unittest
from pathlib import Path

from rbtcv.app import RBTReviewApp


class GUIContractTests(unittest.TestCase):
    def test_active_gui_actions_are_available(self) -> None:
        required_actions = {
            "choose_dataset",
            "reload_dataset",
            "auto_detect_ticks",
            "auto_detect_day_ticks",
            "calibrate_and_analyze_day",
            "analyze_current_tracking",
            "analyze_selected_animal",
            "analyze_selected_day",
            "save_annotation",
            "show_frame",
            "draw_mark_overlays",
        }
        self.assertTrue(required_actions.issubset(RBTReviewApp.__dict__))

    def test_gui_class_has_no_shadowed_method_definitions(self) -> None:
        source = Path("rbtcv/app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        app_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RBTReviewApp"
        )
        methods = [
            node.name for node in app_class.body if isinstance(node, ast.FunctionDef)
        ]
        self.assertEqual(len(methods), len(set(methods)))


if __name__ == "__main__":
    unittest.main()