from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.graph.workflow import (
    build_workflow,
    review_interpretation_node,
    review_synthesis_node,
    review_visualization_node,
    save_chart_node,
)
from src.graph.state import initial_state


class HumanReviewTests(unittest.TestCase):
    def test_chart_approval_routes_to_saving(self) -> None:
        result = review_visualization_node(
            {"user_input": "setuju", "pending_review": "visualizer"}
        )
        self.assertTrue(result["visualizer_approved"])
        self.assertEqual(result["review_route"], "save_chart")

    def test_bare_chart_rejection_asks_for_revision_and_keeps_review_pending(self) -> None:
        result = review_visualization_node(
            {"user_input": "tidak", "pending_review": "visualizer"}
        )
        self.assertEqual(result["review_route"], "end")
        self.assertEqual(result["pending_review"], "visualizer")

    def test_missing_chart_preview_is_reported_without_crashing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            result = save_chart_node(
                {"chart_preview_path": str(Path(temp_dir) / "missing.png")}
            )

        self.assertEqual(result["pending_review"], "")
        self.assertEqual(result["chart_preview_path"], "")
        self.assertIn("dibuat ulang", result["messages"][0].content)

    def test_interpretation_revision_is_saved_as_user_text(self) -> None:
        result = review_interpretation_node(
            {
                "user_input": "revisi: Penjualan naik pada kuartal kedua.",
                "pending_review": "interpreter",
            }
        )
        self.assertEqual(
            result["current_interpretation"], "Penjualan naik pada kuartal kedua."
        )
        self.assertEqual(result["review_route"], "save_report")

    def test_synthesis_rejection_routes_back_to_synthesizer(self) -> None:
        result = review_synthesis_node(
            {"user_input": "tidak setuju, perhatikan penurunan biaya"}
        )
        self.assertEqual(result["review_route"], "synthesize")
        self.assertEqual(result["review_feedback"], "tidak setuju, perhatikan penurunan biaya")

    def test_graph_runs_chart_and_interpretation_reviews_in_order(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "sales.txt"
            def preview_chart(_):
                return {
                    "pending_review": "visualizer",
                    "chart_preview_path": str(Path(temp_dir) / "preview.png"),
                    "messages": [AIMessage(content="Chart preview")],
                }

            def interpret_chart(_):
                return {
                    "pending_review": "interpreter",
                    "current_interpretation": "Revenue increased.",
                    "messages": [AIMessage(content="Chart interpretation")],
                }

            with (
                patch("src.graph.workflow.visualizer_node", preview_chart),
                patch("src.graph.workflow.interpreter_node", interpret_chart),
                patch(
                    "src.graph.workflow.approve_chart",
                    return_value=Path(temp_dir) / "sales.png",
                ),
                patch("src.graph.workflow.save_report", return_value=report_path),
            ):
                graph = build_workflow()
                state = initial_state("data/sample.csv")
                state["user_input"] = "Create a sales chart"
                reviewed_chart = graph.invoke(state)

                self.assertEqual(reviewed_chart["pending_review"], "visualizer")
                reviewed_chart["user_input"] = "setuju"
                reviewed_interpretation = graph.invoke(reviewed_chart)

                self.assertEqual(reviewed_interpretation["pending_review"], "interpreter")
                reviewed_interpretation["user_input"] = "setuju"
                saved_report = graph.invoke(reviewed_interpretation)

                self.assertEqual(saved_report["pending_review"], "")
                self.assertTrue(saved_report["interpreter_approved"])


if __name__ == "__main__":
    unittest.main()
