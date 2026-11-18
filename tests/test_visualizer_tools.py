from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.tools import synthesizer_tools, visualizer_tools
from src.tools.interpreter_tools import get_chart_path, save_report


class VisualizerToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.frame = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "x": [1, 2, 3, 4],
                "y": [2, 4, 3, 8],
            }
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_upload_and_load_csv_stays_inside_data_directory(self) -> None:
        data_dir = self.root / "data"
        with patch.object(visualizer_tools, "DATA_DIR", data_dir):
            uploaded = visualizer_tools.save_dataset_upload("sample.csv", b"x,y\n1,2\n")
            loaded = visualizer_tools.load_dataset(uploaded)

            self.assertEqual(uploaded.parent, data_dir)
            self.assertEqual(list(loaded.columns), ["x", "y"])
            with self.assertRaises(ValueError):
                visualizer_tools.load_dataset(self.root / "outside.csv")

    def test_each_chart_type_creates_preview_and_approval_moves_it(self) -> None:
        pending_dir = self.root / "pending"
        chart_dir = self.root / "charts"
        requests = [
            ("line", ["x", "y"]),
            ("bar", ["group", "y"]),
            ("histogram", ["x"]),
            ("box", ["y"]),
            ("scatter", ["x", "y"]),
            ("heatmap", []),
        ]
        with (
            patch.object(visualizer_tools, "PENDING_CHART_DIR", pending_dir),
            patch.object(visualizer_tools, "CHART_DIR", chart_dir),
        ):
            for chart_type, columns in requests:
                preview = visualizer_tools.create_chart(
                    self.frame, chart_type, columns, f"{chart_type} test"
                )
                self.assertTrue(preview.is_file())
                approved = visualizer_tools.approve_chart(preview)
                self.assertTrue(approved.is_file())
                self.assertEqual(approved.parent, chart_dir)
                self.assertFalse(preview.exists())

    def test_scatter_rejects_non_numeric_columns(self) -> None:
        with patch.object(visualizer_tools, "PENDING_CHART_DIR", self.root / "pending"):
            with self.assertRaises(visualizer_tools.ChartValidationError):
                visualizer_tools.create_chart(
                    self.frame, "scatter", ["group", "y"], "invalid scatter"
                )

    def test_reports_can_be_collected_and_final_summary_saved(self) -> None:
        report_dir = self.root / "reports"
        summary_path = self.root / "final.txt"
        with (
            patch("src.tools.interpreter_tools.REPORT_DIR", report_dir),
            patch.object(synthesizer_tools, "REPORT_DIR", report_dir),
            patch.object(synthesizer_tools, "SUMMARY_PATH", summary_path),
        ):
            report_path = save_report("sales.png", "Revenue increased.")
            reports = synthesizer_tools.read_all_reports()
            saved_summary = synthesizer_tools.save_final_summary("Overall growth.")

            self.assertEqual(report_path.name, "sales.txt")
            self.assertIn("Revenue increased.", reports)
            self.assertEqual(saved_summary.read_text(encoding="utf-8"), "Overall growth.\n")

    def test_chart_lookup_rejects_path_traversal(self) -> None:
        with patch("src.tools.interpreter_tools.CHART_DIR", self.root / "charts"):
            with self.assertRaises(ValueError):
                get_chart_path("../outside.png")


if __name__ == "__main__":
    unittest.main()
