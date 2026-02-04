from __future__ import annotations
from pathlib import Path
import os
import streamlit as st

from rag.config import Settings
from rag.system import RAGSystem

st.set_page_config(page_title="Chat With Your Docs", layout="wide")
st.title("Chat With Your Docs")

with st.sidebar:
    st.subheader("Config")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    top_k = st.slider("Top-K", 2, 10, 5)
    min_score = st.slider("Min score (confidence)", 0.0, 0.6, 0.25, 0.01)

    st.caption("Tip: For ECS, set OPENAI_API_KEY as an env var.")

@st.cache_resource
def get_system(api_key_val: str, top_k_val: int, min_score_val: float) -> RAGSystem:
    s = Settings(top_k=top_k_val, min_score=min_score_val)
    sys = RAGSystem(s)
    # try load existing store
    try:
        sys.load()
    except Exception:
        pass
    return sys

if not api_key:
    st.info("Enter an OpenAI API key in the sidebar (or set OPENAI_API_KEY).")
    st.stop()

system = get_system(api_key, top_k, min_score)

st.header("1) Ingest Documents")
uploaded = st.file_uploader("Upload PDFs/TXTs", type=["pdf", "txt"], accept_multiple_files=True)

col1, col2 = st.columns([1, 2])
with col1:
    ingest_clicked = st.button("Ingest")
with col2:
    if st.button("Reload existing index"):
        try:
            system.load()
            st.success("Loaded existing index from vector_store/")
        except Exception as e:
            st.error(str(e))

if ingest_clicked:
    if not uploaded:
        st.warning("Upload at least one file.")
    else:
        docs_dir = Path("documents")
        docs_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for f in uploaded:
            out = docs_dir / f.name
            out.write_bytes(f.getbuffer())
            paths.append(out)

        with st.spinner("Indexing..."):
            result = system.ingest(paths)
        st.success(f"Indexed {result.get('chunks_indexed', 0)} chunks.")

st.divider()
st.header("2) Ask Questions")

if "chat" not in st.session_state:
    st.session_state.chat = []

query = st.text_input("Question", placeholder="Ask something about the uploaded docs…")
ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Thinking..."):
        out = system.answer(query, chat_history=None)

    st.session_state.chat.append({"q": query, "a": out["answer"], "sources": out["sources"], "score": out["best_score"]})

for turn in reversed(st.session_state.chat):
    st.markdown(f"**Q:** {turn['q']}")
    st.markdown(f"**A:** {turn['a']}")
    with st.expander(f"Sources (best_score={turn['score']:.3f})"):
        if not turn["sources"]:
            st.write("No sources (below confidence threshold).")
        else:
            st.json(turn["sources"])
    st.divider()
