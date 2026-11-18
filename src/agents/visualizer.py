from __future__ import annotations

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from src.agents.model import get_chat_model
from src.graph.state import AgentState
from src.tools.visualizer_tools import (
    ChartType,
    ChartValidationError,
    create_chart,
    discard_chart,
    get_column_metadata,
    load_dataset,
)


class ChartPlan(BaseModel):
    chart_type: ChartType
    columns: list[str] = Field(
        description=(
            "Columns in axis order: two for line/bar/scatter, one for histogram/box, "
            "and an empty list for heatmap."
        )
    )
    title: str = Field(min_length=1)


def visualizer_node(state: AgentState) -> dict:
    dataset_path = state.get("dataset_path")
    if not dataset_path:
        return {
            "messages": [
                AIMessage(content="Unggah dataset CSV terlebih dahulu sebelum membuat grafik.")
            ],
            "pending_review": "",
        }

    if state.get("chart_preview_path"):
        discard_chart(state["chart_preview_path"])

    frame = load_dataset(dataset_path)
    metadata = get_column_metadata(frame)
    user_request = state.get("chart_request") or state.get("user_input", "")
    feedback = state.get("review_feedback", "")
    prompt = (
        "You are a data visualization planner. Choose exactly one supported chart and "
        "the required columns based on the request. Preserve column names exactly. "
        "Supported chart_type values: line, bar, histogram, box, scatter, heatmap. "
        "Rules: line/bar use [category_or_x, numeric_value], histogram/box use "
        "[numeric_column], scatter uses [numeric_x, numeric_y], heatmap uses []. "
        "A bar chart aggregates the numeric value by category. "
        "Use an Indonesian title when the request is in Indonesian. "
        f"\nAvailable columns and pandas dtypes: {metadata}"
        f"\nUser request: {user_request}"
        f"\nRevision feedback, if any: {feedback or 'none'}"
        "\nReturn a concise, user-friendly chart title."
    )
    plan = get_chat_model().with_structured_output(ChartPlan).invoke(prompt)

    try:
        preview = create_chart(frame, plan.chart_type, plan.columns, plan.title)
    except ChartValidationError as error:
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Permintaan grafik belum valid: {error}\n\n"
                        f"Kolom tersedia: {metadata}. Silakan minta jenis grafik dan "
                        "kolom yang sesuai, lalu coba lagi."
                    )
                )
            ],
            "pending_review": "",
            "chart_preview_path": "",
            "review_feedback": "",
        }

    used_columns = ", ".join(plan.columns) if plan.columns else "semua kolom numerik"
    return {
        "chart_plan": plan.model_dump(),
        "chart_preview_path": str(preview),
        "pending_review": "visualizer",
        "visualizer_approved": False,
        "review_feedback": "",
        "messages": [
            AIMessage(
                content=(
                    f"Pratinjau {plan.chart_type} **{plan.title}** berhasil dibuat "
                    f"menggunakan kolom: {used_columns}.\n\n"
                    "Tinjau grafik di bawah. Ketik **setuju** untuk menyimpan, atau "
                    "berikan revisi (misalnya: `revisi, gunakan ...`)."
                )
            )
        ],
    }