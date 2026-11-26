from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import AgentState, initial_state
from src.graph.workflow import workflow
from src.tools.visualizer_tools import (
    get_column_metadata,
    list_data_files,
    load_dataset,
    save_dataset_upload,
)

_UPLOAD_DATASET_OPTION = "Unggah dataset lain..."

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
    if "dataset_metadata" not in st.session_state:
        dataset_path = st.session_state.agent_state.get("dataset_path", "")
        st.session_state.dataset_metadata = (
            get_column_metadata(load_dataset(dataset_path))
            if dataset_path and Path(dataset_path).is_file()
            else None
        )


def _render_chat(state: AgentState) -> None:
    messages = state.get("messages", [])
    for message in messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(str(message.content))

    is_visualizer_review = state.get("pending_review") == "visualizer"
    chart_path = (
        state.get("chart_preview_path", "")
        if is_visualizer_review
        else state.get("chart_path", "")
    )
    if chart_path:
        image_path = Path(chart_path)
        with st.chat_message("assistant"):
            if image_path.is_file():
                st.image(str(image_path), use_container_width=True)
            elif is_visualizer_review:
                st.warning(
                    "File pratinjau grafik sudah tidak tersedia. "
                    "Minta grafik dibuat ulang sebelum menyetujuinya."
                )


def _load_dataset(dataset_path: str | Path) -> None:
    frame = load_dataset(dataset_path)
    metadata = get_column_metadata(frame)
    dataset_name = Path(dataset_path).name
    state = initial_state(str(dataset_path))
    state["messages"] = [
        AIMessage(
            content=(
                f"Dataset `{dataset_name}` berhasil dimuat "
                f"({len(frame)} baris, {len(frame.columns)} kolom). "
                "Kolom dan tipe data ditampilkan pada tabel.\n\n"
                "Minta grafik, misalnya `buat histogram untuk usia` atau "
                "`buat scatter plot antara tinggi dan berat`."
            )
        )
    ]
    st.session_state.agent_state = state
    st.session_state.dataset_metadata = metadata


def _render_dataset_metadata() -> None:
    metadata = st.session_state.dataset_metadata
    if metadata is not None:
        st.markdown("Kolom dan tipe data:")
        st.dataframe(metadata, hide_index=True, use_container_width=True)


def _save_upload(uploaded_file) -> bool:
    content = uploaded_file.getvalue()
    fingerprint = f"{uploaded_file.name}:{hashlib.sha256(content).hexdigest()}"
    if fingerprint == st.session_state.upload_fingerprint:
        return False

    dataset_path = save_dataset_upload(uploaded_file.name, content)
    _load_dataset(dataset_path)
    st.session_state.upload_fingerprint = fingerprint
    return True


def main() -> None:
    _ensure_session()
    st.title("📊 Data Analyst Agent")
    st.caption("Visualisasi, interpretasi grafik, dan rangkuman laporan dengan Gemini.")

    available_datasets = list_data_files()
    with st.chat_message("assistant"):
        if available_datasets:
            st.markdown(
                "Dataset CSV yang tersedia di `data/`. Pilih dataset yang ingin digunakan. "
                "Jika tidak ada yang sesuai, pilih opsi unggah."
            )
            selected_dataset = st.selectbox(
                "Pilih dataset",
                options=[
                    "",
                    *(str(path) for path in available_datasets),
                    _UPLOAD_DATASET_OPTION,
                ],
                format_func=lambda option: (
                    "Pilih dataset..."
                    if option == ""
                    else Path(option).name
                    if option != _UPLOAD_DATASET_OPTION
                    else option
                ),
                key="dataset_selector",
            )
            if selected_dataset == _UPLOAD_DATASET_OPTION:
                st.info("Silakan unggah file CSV yang ingin digunakan.")
                uploaded_file = st.file_uploader(
                    "Unggah dataset CSV",
                    type=["csv"],
                    accept_multiple_files=False,
                    key="dataset_uploader",
                )
            else:
                uploaded_file = None
        else:
            st.info("Belum ada file CSV di folder `data/`. Silakan unggah dataset.")
            selected_dataset = ""
            uploaded_file = st.file_uploader(
                "Unggah dataset CSV",
                type=["csv"],
                accept_multiple_files=False,
                key="dataset_uploader",
            )

        upload_loaded = uploaded_file is not None and _save_upload(uploaded_file)
        if (
            not upload_loaded
            and selected_dataset
            and selected_dataset != _UPLOAD_DATASET_OPTION
            and selected_dataset != st.session_state.agent_state.get("dataset_path")
        ):
            _load_dataset(selected_dataset)
        _render_dataset_metadata()

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