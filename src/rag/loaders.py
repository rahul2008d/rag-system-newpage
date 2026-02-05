from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_files(paths: Iterable[Path]) -> List[Document]:
    """Load PDF/TXT files into LangChain Documents (PDF is page-split)."""
    docs: List[Document] = []

    for p in paths:
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            # PyPDFLoader returns one Document per page with metadata incl. page index
            docs.extend(loader.load())
        elif p.suffix.lower() == ".txt":
            loader = TextLoader(str(p), encoding="utf-8")
            d = loader.load()
            # Ensure source is consistent with PDFs
            for doc in d:
                doc.metadata.setdefault("source", str(p))
                doc.metadata.setdefault("page", 0)
            docs.extend(d)

    return docs
