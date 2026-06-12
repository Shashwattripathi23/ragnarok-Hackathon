"""Calculemus Study Agent — Streamlit chat over a metadata-rich PDF index.

Pipeline: upload/point at PDFs -> parallel parse + image captioning + local
embeddings -> ChromaDB -> intent-routed, tool-calling RAG agent with citations.
"""
import os
from pathlib import Path

import streamlit as st

from study.config import IMAGES_DIR, INDEX_DIR, ROOT
from study.indexer import index_files
from study.retriever import Retriever
from study.embeddings import warm_up
from study.agent.orchestrator import answer

st.set_page_config(page_title="Calculemus Study Agent", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource(show_spinner="Loading embedding model…")
def _warm_embeddings() -> bool:
    warm_up()
    return True


# --- State -----------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

_warm_embeddings()
retriever = get_retriever()

st.title("🎓 Calculemus: Personalized Study Agent")
st.caption("Accurate, cited answers over your course PDFs — text, tables, and figures.")


# --- Sidebar: build the knowledge base -------------------------------------
with st.sidebar:
    st.header("📂 Knowledge Base")
    st.metric("Chunks indexed", retriever.store.count())

    folder = st.text_input("Course folder (PDFs)", value=str(ROOT / "CourseMaterial"))
    uploaded = st.file_uploader(
        "…or upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    rebuild = st.checkbox("Rebuild from scratch", value=False,
                          help="Wipe the index and re-parse everything.")

    if st.button("⚙️ Build / Update Index", type="primary"):
        paths = []
        if folder and Path(folder).is_dir():
            paths += [str(p) for p in Path(folder).rglob("*.pdf")]
        if uploaded:
            up_dir = INDEX_DIR / "uploads"
            up_dir.mkdir(parents=True, exist_ok=True)
            for uf in uploaded:
                dest = up_dir / uf.name
                dest.write_bytes(uf.getbuffer())
                paths.append(str(dest))

        if not paths:
            st.warning("No PDFs found. Add a valid folder path or upload files.")
        else:
            bar = st.progress(0.0)
            status = st.empty()

            def on_progress(stage: str, cur: int, tot: int):
                frac = cur / tot if tot else 1.0
                bar.progress(min(frac, 1.0))
                status.write(f"**{stage}** · {cur}/{tot}")

            with st.spinner(f"Indexing {len(paths)} PDF(s)…"):
                summary = index_files(paths, on_progress=on_progress, rebuild=rebuild)
            get_retriever.clear()  # refresh retriever to see new data
            bar.empty()
            status.empty()
            st.success(
                f"Indexed {summary['indexed_files']} file(s) · "
                f"{summary['chunks']} chunks ({summary['images']} images) · "
                f"{summary['skipped']} unchanged."
            )
            st.rerun()

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()


# --- Helpers ----------------------------------------------------------------
def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for h in sources:
            meta = h["metadata"]
            st.markdown(f"**{h['citation']}** · _{meta.get('type', 'text')}_")
            if meta.get("type") == "image" and meta.get("image_path") \
                    and os.path.exists(meta["image_path"]):
                st.image(meta["image_path"], width=240)
            snippet = (h["text"] or "")[:300].strip()
            if snippet:
                st.caption(snippet + ("…" if len(h["text"]) > 300 else ""))
            st.divider()


# --- Chat -------------------------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

if prompt := st.chat_input("Ask about your course materials…"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if retriever.store.count() == 0:
        warning = "⚠️ The index is empty — build the knowledge base in the sidebar first."
        with st.chat_message("assistant"):
            st.markdown(warning)
        st.session_state.chat_history.append({"role": "assistant", "content": warning, "sources": []})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                # History excludes the just-added user turn.
                result = answer(prompt, history=st.session_state.chat_history[:-1],
                                retriever=retriever)
            st.markdown(result["answer"])
            render_sources(result["sources"])
            st.caption(f"intent: `{result['intent']}`")
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
