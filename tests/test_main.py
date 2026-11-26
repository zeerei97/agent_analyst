from __future__ import annotations

import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage

from main import _load_dataset, _render_chat, _render_dataset_metadata


class RenderChatTests(unittest.TestCase):
    def test_approved_chart_remains_visible_during_interpretation_review(self) -> None:
        with TemporaryDirectory() as temp_dir:
            chart_path = Path(temp_dir) / "approved-chart.png"
            chart_path.write_bytes(b"chart")
            stale_preview_path = Path(temp_dir) / "stale-preview.png"

            with (
                patch("main.st.chat_message", return_value=nullcontext()),
                patch("main.st.markdown"),
                patch("main.st.image") as image,
            ):
                _render_chat(
                    {
                        "messages": [AIMessage(content="Tinjau interpretasi.")],
                        "pending_review": "interpreter",
                        "chart_preview_path": str(stale_preview_path),
                        "chart_path": str(chart_path),
                    }
                )

        image.assert_called_once_with(str(chart_path), use_container_width=True)

    def test_dataset_metadata_is_rendered_as_a_dataframe(self) -> None:
        metadata = pd.DataFrame(
            [{"column": "sales", "dtype": "int64"}]
        )
        with (
            patch("main.st.session_state") as session_state,
            patch("main.st.markdown"),
            patch("main.st.dataframe") as dataframe,
        ):
            session_state.dataset_metadata = metadata
            _render_dataset_metadata()

        dataframe.assert_called_once_with(
            metadata, hide_index=True, use_container_width=True
        )

    def test_load_dataset_accepts_string_path(self) -> None:
        frame = pd.DataFrame({"sales": [1]})
        metadata = pd.DataFrame([{"column": "sales", "dtype": "int64"}])
        with (
            patch("main.load_dataset", return_value=frame),
            patch("main.get_column_metadata", return_value=metadata),
            patch("main.st.session_state") as session_state,
        ):
            _load_dataset("/workspace/data/sales.csv")

        message = session_state.agent_state["messages"][0].content
        self.assertIn("Dataset `sales.csv` berhasil dimuat", message)


if __name__ == "__main__":
    unittest.main()
