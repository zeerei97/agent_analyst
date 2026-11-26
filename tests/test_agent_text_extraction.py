from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.interpreter import extract_text as extract_interpreter_text
from src.agents.interpreter import interpreter_node
from src.agents.synthesizer import extract_text as extract_synthesizer_text
from src.agents.synthesizer import synthesizer_node


class AgentTextExtractionTests(unittest.TestCase):
    def test_extract_text_handles_strings_and_multimodal_content(self) -> None:
        for extractor in (extract_interpreter_text, extract_synthesizer_text):
            with self.subTest(extractor=extractor.__module__):
                self.assertEqual(extractor("Plain text"), "Plain text")
                self.assertEqual(
                    extractor(
                        [
                            {"type": "text", "text": "Chart "},
                            {"type": "image_url", "image_url": "ignored"},
                            "analysis",
                        ]
                    ),
                    "Chart analysis",
                )
                self.assertEqual(extractor(None), "None")

    def test_interpreter_stores_only_text_from_model_content(self) -> None:
        with TemporaryDirectory() as temp_dir:
            chart_path = Path(temp_dir) / "chart.png"
            chart_path.write_bytes(b"chart")
            response = SimpleNamespace(
                content=[
                    {"type": "text", "text": "Sales increased."},
                    {"type": "image_url", "image_url": "ignored"},
                ]
            )
            with patch("src.agents.interpreter.get_chat_model") as get_model:
                get_model.return_value.invoke.return_value = response
                result = interpreter_node({"chart_path": str(chart_path)})

        self.assertEqual(result["current_interpretation"], "Sales increased.")
        self.assertIn("Sales increased.", result["messages"][0].content)
        self.assertNotIn("image_url", result["messages"][0].content)

    def test_synthesizer_stores_only_text_from_model_content(self) -> None:
        response = SimpleNamespace(
            content=[
                {"type": "text", "text": "Revenue increased."},
                {"type": "image_url", "image_url": "ignored"},
            ]
        )
        with (
            patch("src.agents.synthesizer.read_all_reports", return_value="Reports"),
            patch("src.agents.synthesizer.get_chat_model") as get_model,
        ):
            get_model.return_value.invoke.return_value = response
            result = synthesizer_node({})

        self.assertEqual(result["final_summary"], "Revenue increased.")
        self.assertIn("Revenue increased.", result["messages"][0].content)
        self.assertNotIn("image_url", result["messages"][0].content)


if __name__ == "__main__":
    unittest.main()
