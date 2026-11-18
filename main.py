from __future__ import annotations

import hashlib

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import AgentState, initial_state
from src.graph.workflow import workflow
from src.tools.visualizer_tools import get_column_metadata, load_dataset, save_dataset_upload

st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="centered")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #101318; }
    [data-testid="stChatMessage"] {
        border: 1px solid #29313d;
        border-radius: 14px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.75rem;
    }
    [data-testid="stChatInput"] { border-color: #394454; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _ensure_session() -> None:
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = initial_state()
    if "upload_fingerprint" not in st.session_state:
        st.session_state.upload_fingerprint = ""


def _render_chat(state: AgentState) -> None:
    messages = state.get("messages", [])
    for index, message in enumerate(messages):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(str(message.content))
            if (
                role == "assistant"
                and index == len(messages) - 1
                and state.get("pending_review") == "visualizer"
                and state.get("chart_preview_path")
            ):
                st.image(state["chart_preview_path"], use_container_width=True)


def _save_upload(uploaded_file) -> None:
    content = uploaded_file.getvalue()
    fingerprint = f"{uploaded_file.name}:{hashlib.sha256(content).hexdigest()}"
    if fingerprint == st.session_state.upload_fingerprint:
        return

    dataset_path = save_dataset_upload(uploaded_file.name, content)
    frame = load_dataset(dataset_path)
    metadata = get_column_metadata(frame)
    state = initial_state(str(dataset_path))
    state["messages"] = [
        AIMessage(
            content=(
                f"Dataset `{dataset_path.name}` berhasil dimuat "
                f"({len(frame)} baris, {len(frame.columns)} kolom).\n\n"
                f"Kolom dan tipe data: {metadata}\n\n"
                "Minta grafik, misalnya `buat histogram untuk usia` atau "
                "`buat scatter plot antara tinggi dan berat`."
            )
        )
    ]
    st.session_state.agent_state = state
    st.session_state.upload_fingerprint = fingerprint


def main() -> None:
    _ensure_session()
    st.title("📊 Data Analyst Agent")
    st.caption("Visualisasi, interpretasi grafik, dan rangkuman laporan dengan Gemini.")

    with st.chat_message("assistant"):
        st.markdown("Unggah dataset bersih berformat CSV untuk memulai.")
        uploaded_file = st.file_uploader(
            "Unggah dataset CSV",
            type=["csv"],
            accept_multiple_files=False,
            key="dataset_uploader",
        )
        if uploaded_file is not None:
            _save_upload(uploaded_file)

    state: AgentState = st.session_state.agent_state
    _render_chat(state)

    if state.get("finished"):
        if state.get("synthesizer_approved") and state.get("final_summary"):
            st.success("Rangkuman akhir")
            st.markdown(state["final_summary"])
        else:
            st.info(
                "Sesi berakhir, tetapi belum ada rangkuman yang disetujui. "
                "Mulai sesi baru dan minta rangkuman sebelum mengetik `done` atau `exit`."
            )
        return

    prompt = st.chat_input(
        "Tulis instruksi atau keputusan review...",
        disabled=not bool(state.get("dataset_path")),
    )
    if prompt is None:
        return

    prompt = prompt.strip()
    if not prompt:
        return

    next_state = dict(state)
    next_state["messages"] = [*state.get("messages", []), HumanMessage(content=prompt)]
    next_state["user_input"] = prompt
    if prompt.casefold() in {"exit", "done"}:
        next_state["finished"] = True
        st.session_state.agent_state = next_state
        st.rerun()

    st.session_state.agent_state = workflow.invoke(next_state)
    st.rerun()


if __name__ == "__main__":
    main()