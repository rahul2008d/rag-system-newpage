from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def ensure_dir(d: str) -> Path:
    """Ensure directory exists."""
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_vectorstore(
    docs: List[Document], embeddings: OpenAIEmbeddings
) -> FAISS:
    """Create a FAISS vector store from documents."""
    return FAISS.from_documents(docs, embeddings)


def save_store(store_dir: str, vs: FAISS) -> None:
    """Save vector store locally."""
    ensure_dir(store_dir)
    vs.save_local(store_dir)


def load_store(
    store_dir: str, embeddings: OpenAIEmbeddings, *, allow_unsafe: bool = True
) -> FAISS:
    """Load vector store locally."""
    # LangChain requires allow_dangerous_deserialization when loading pickled data
    return FAISS.load_local(
        store_dir,
        embeddings,
        allow_dangerous_deserialization=allow_unsafe,
    )
