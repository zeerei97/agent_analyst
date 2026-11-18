from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


ReviewStage = Literal["", "visualizer", "interpreter", "synthesizer"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    dataset_path: str
    user_input: str
    next_action: Literal["visualize", "interpret", "synthesize"]
    pending_review: ReviewStage
    review_route: str
    review_feedback: str
    chart_request: str
    chart_plan: dict[str, object]
    chart_preview_path: str
    chart_path: str
    chart_name: str
    visualizer_approved: bool
    interpreter_approved: bool
    current_interpretation: str
    synthesizer_approved: bool
    final_summary: str
    finished: bool


def initial_state(dataset_path: str = "") -> AgentState:
    return {
        "messages": [],
        "dataset_path": dataset_path,
        "user_input": "",
        "next_action": "visualize",
        "pending_review": "",
        "review_route": "",
        "review_feedback": "",
        "chart_request": "",
        "chart_plan": {},
        "chart_preview_path": "",
        "chart_path": "",
        "chart_name": "",
        "visualizer_approved": False,
        "interpreter_approved": False,
        "current_interpretation": "",
        "synthesizer_approved": False,
        "final_summary": "",
        "finished": False,
    }