from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.agents.model import get_chat_model
from src.graph.state import AgentState
from src.tools.interpreter_tools import get_chart_path

_CHART_REFERENCE = re.compile(r"(?<![\w.-])([\w.-]+\.(?:png|jpe?g))\b", re.IGNORECASE)


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


def _chart_path_from_state(state: AgentState) -> Path | None:
    chart_path = state.get("chart_path")
    if chart_path:
        path = Path(chart_path)
        if path.is_file():
            return path

    match = _CHART_REFERENCE.search(state.get("user_input", ""))
    if not match:
        return None
    return get_chart_path(match.group(1))


def interpreter_node(state: AgentState) -> dict:
    try:
        chart_path = _chart_path_from_state(state)
    except (FileNotFoundError, ValueError) as error:
        return {
            "messages": [AIMessage(content=f"{error} Periksa nama grafik di outputs/charts.")],
            "pending_review": "",
        }
    if chart_path is None:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Sebutkan nama file grafik yang tersimpan di outputs/charts "
                        "(contoh: `sales_chart.png`)."
                    )
                )
            ],
            "pending_review": "",
        }

    image_data = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    suffix = chart_path.suffix.lower().lstrip(".")
    mime_type = "image/jpeg" if suffix in {"jpg", "jpeg"} else "image/png"
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Analyze this data visualization. Explain the main patterns, "
                    "comparisons, outliers, and caveats visible in the chart. "
                    "Do not claim information that cannot be read from it. "
                    "Respond in Indonesian."
                ),
            },
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{image_data}",
            },
        ]
    )
    response = get_chat_model().invoke([message])
    interpretation = extract_text(response.content).strip()
    return {
        "chart_path": str(chart_path),
        "chart_name": chart_path.name,
        "current_interpretation": interpretation,
        "pending_review": "interpreter",
        "interpreter_approved": False,
        "messages": [
            AIMessage(
                content=(
                    f"**Interpretasi {chart_path.name}**\n\n{interpretation}\n\n"
                    "Ketik **setuju** untuk menyimpan, **revisi: ...** untuk menyimpan "
                    "versi Anda, atau **tidak setuju** disertai masukan agar analisis dibuat ulang."
                )
            )
        ],
    }