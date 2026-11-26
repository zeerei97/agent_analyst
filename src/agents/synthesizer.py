from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from src.agents.model import get_chat_model
from src.graph.state import AgentState
from src.tools.synthesizer_tools import read_all_reports


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return str(content)


def synthesizer_node(state: AgentState) -> dict:
    try:
        reports = read_all_reports()
    except FileNotFoundError as error:
        return {
            "messages": [AIMessage(content=str(error))],
            "pending_review": "",
        }

    feedback = state.get("review_feedback", "")
    response = get_chat_model().invoke(
        [
            (
                "system",
                "Synthesize the supplied chart reports into one evidence-based, "
                "coherent conclusion. Preserve important findings and caveats. "
                "Do not invent data. Write the synthesis in Indonesian.",
            ),
            (
                "human",
                f"Reports:\n{reports}\n\nRevision feedback: {feedback or 'none'}",
            ),
        ]
    )
    summary = extract_text(response.content).strip()
    return {
        "final_summary": summary,
        "pending_review": "synthesizer",
        "synthesizer_approved": False,
        "review_feedback": "",
        "messages": [
            AIMessage(
                content=(
                    f"**Draf rangkuman akhir**\n\n{summary}\n\n"
                    "Ketik **setuju** untuk menyimpan ke final_analyst.txt, "
                    "**revisi: ...** untuk mengganti teks final, atau "
                    "**tidak setuju** beserta masukan untuk membuat ulang."
                )
            )
        ],
    }