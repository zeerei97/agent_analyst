from __future__ import annotations

import re

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from src.agents.interpreter import interpreter_node
from src.agents.synthesizer import synthesizer_node
from src.agents.visualizer import visualizer_node
from src.graph.state import AgentState, initial_state
from src.tools.interpreter_tools import save_report
from src.tools.synthesizer_tools import save_final_summary
from src.tools.visualizer_tools import approve_chart

_YES = {"yes", "y", "approve", "approved", "agree", "setuju", "ya", "ok", "oke"}
_NO = {"no", "n", "reject", "rejected", "disagree", "tidak", "enggak", "engga"}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _affirmative(text: str) -> bool:
    normalized = _normalized(text).strip(" .!?")
    return normalized in _YES or normalized.startswith(("setuju,", "agree,", "approved"))


def _is_bare_no(text: str) -> bool:
    return _normalized(text).strip(" .!?") in _NO


def _explicit_rejection(text: str) -> bool:
    normalized = _normalized(text)
    return any(
        phrase in normalized
        for phrase in ("tidak setuju", "do not agree", "don't agree", "disagree", "rejected")
    ) or _is_bare_no(text)


def dispatch_node(state: AgentState) -> dict:
    if state.get("pending_review"):
        return {}

    text = _normalized(state.get("user_input", ""))
    if any(
        term in text
        for term in (
            "rangkum",
            "merangkum",
            "rangkuman",
            "summarize",
            "summary",
            "synthesize",
            "synthesis",
            "kesimpulan",
        )
    ):
        return {"next_action": "synthesize"}
    if any(
        term in text
        for term in (
            "interpretasi",
            "interpret",
            "jelaskan grafik",
            "analisis grafik",
            "explain chart",
            "analyze chart",
        )
    ):
        return {"next_action": "interpret"}
    return {
        "next_action": "visualize",
        "chart_request": state.get("user_input", ""),
        "review_feedback": "",
    }


def review_visualization_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    if _affirmative(user_input):
        return {
            "visualizer_approved": True,
            "pending_review": "",
            "review_route": "save_chart",
        }
    if _is_bare_no(user_input):
        return {
            "review_route": "end",
            "pending_review": "visualizer",
            "messages": [
                AIMessage(content="Baik. Jelaskan bagian grafik yang ingin diubah, lalu saya buat ulang.")
            ],
        }
    return {
        "pending_review": "",
        "review_feedback": user_input,
        "review_route": "visualize",
    }


def save_chart_node(state: AgentState) -> dict:
    preview_path = state.get("chart_preview_path", "")
    try:
        chart_path = approve_chart(preview_path)
    except FileNotFoundError:
        return {
            "pending_review": "",
            "chart_preview_path": "",
            "messages": [
                AIMessage(
                    content=(
                        "Pratinjau grafik sudah tidak tersedia, jadi grafik belum disimpan. "
                        "Silakan minta grafik dibuat ulang."
                    )
                )
            ],
        }
    return {
        "chart_path": str(chart_path),
        "chart_name": chart_path.name,
        "messages": [
            AIMessage(
                content=(
                    f"Grafik disetujui dan disimpan sebagai `{chart_path.name}`. "
                    "Saya akan menyiapkan interpretasinya."
                )
            )
        ],
    }


def review_interpretation_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    if _affirmative(user_input):
        return {
            "interpreter_approved": True,
            "pending_review": "",
            "review_route": "save_report",
        }
    if _is_bare_no(user_input):
        return {
            "review_route": "end",
            "pending_review": "interpreter",
            "messages": [
                AIMessage(
                    content="Jelaskan apa yang perlu dikaji ulang, lalu saya buat ulang interpretasinya."
                )
            ],
        }
    if _explicit_rejection(user_input):
        return {
            "pending_review": "",
            "review_feedback": user_input,
            "review_route": "interpret",
        }

    revision = re.sub(r"^\s*(revisi|revision)\s*:\s*", "", user_input, flags=re.IGNORECASE).strip()
    return {
        "current_interpretation": revision or user_input.strip(),
        "interpreter_approved": True,
        "pending_review": "",
        "review_route": "save_report",
    }


def save_report_node(state: AgentState) -> dict:
    chart_name = state.get("chart_name", "")
    interpretation = state.get("current_interpretation", "")
    report_path = save_report(chart_name, interpretation)
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Interpretasi disimpan di `{report_path.relative_to(report_path.parents[2])}`. "
                    "Anda dapat meminta grafik lain atau mengetik **rangkum laporan**."
                )
            )
        ]
    }


def review_synthesis_node(state: AgentState) -> dict:
    user_input = state.get("user_input", "")
    if _affirmative(user_input):
        return {
            "synthesizer_approved": True,
            "pending_review": "",
            "review_route": "save_summary",
        }
    if _explicit_rejection(user_input):
        return {
            "pending_review": "",
            "review_feedback": user_input,
            "review_route": "synthesize",
        }

    revision = re.sub(r"^\s*(revisi|revision)\s*:\s*", "", user_input, flags=re.IGNORECASE).strip()
    return {
        "final_summary": revision or user_input.strip(),
        "synthesizer_approved": True,
        "pending_review": "",
        "review_route": "save_summary",
    }


def save_summary_node(state: AgentState) -> dict:
    summary_path = save_final_summary(state.get("final_summary", ""))
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Rangkuman disetujui dan disimpan di `{summary_path.name}`. "
                    "Ketik **exit** atau **done** untuk mengakhiri sesi."
                )
            )
        ]
    }


def _route_after_dispatch(state: AgentState) -> str:
    pending = state.get("pending_review", "")
    if pending == "visualizer":
        return "review_visualization"
    if pending == "interpreter":
        return "review_interpretation"
    if pending == "synthesizer":
        return "review_synthesis"
    action = state.get("next_action", "visualize")
    return {
        "visualize": "visualizer",
        "interpret": "interpreter",
        "synthesize": "synthesizer",
    }[action]


def _route_from_review(state: AgentState) -> str:
    return state.get("review_route", "end")


def build_workflow():
    builder = StateGraph(AgentState)
    builder.add_node("dispatch", dispatch_node)
    builder.add_node("visualizer", visualizer_node)
    builder.add_node("review_visualization", review_visualization_node)
    builder.add_node("save_chart", save_chart_node)
    builder.add_node("interpreter", interpreter_node)
    builder.add_node("review_interpretation", review_interpretation_node)
    builder.add_node("save_report", save_report_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("review_synthesis", review_synthesis_node)
    builder.add_node("save_summary", save_summary_node)

    builder.add_edge(START, "dispatch")
    builder.add_conditional_edges(
        "dispatch",
        _route_after_dispatch,
        {
            "visualizer": "visualizer",
            "review_visualization": "review_visualization",
            "save_chart": "save_chart",
            "interpreter": "interpreter",
            "review_interpretation": "review_interpretation",
            "save_report": "save_report",
            "synthesizer": "synthesizer",
            "review_synthesis": "review_synthesis",
            "save_summary": "save_summary",
        },
    )
    builder.add_conditional_edges(
        "review_visualization",
        _route_from_review,
        {"save_chart": "save_chart", "visualize": "visualizer", "end": END},
    )
    builder.add_edge("save_chart", "interpreter")
    builder.add_edge("visualizer", END)
    builder.add_conditional_edges(
        "review_interpretation",
        _route_from_review,
        {"save_report": "save_report", "interpret": "interpreter", "end": END},
    )
    builder.add_edge("interpreter", END)
    builder.add_edge("save_report", END)
    builder.add_edge("synthesizer", END)
    builder.add_conditional_edges(
        "review_synthesis",
        _route_from_review,
        {"save_summary": "save_summary", "synthesize": "synthesizer"},
    )
    builder.add_edge("save_summary", END)
    return builder.compile()


workflow = build_workflow()

__all__ = ["build_workflow", "initial_state", "workflow"]