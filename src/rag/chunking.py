from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: List[Document], chunk_size: int, overlap: int
) -> List[Document]:
    """Split documents into overlapping chunks and add stable chunk metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(pages)

    per_source_page_counter: dict[tuple[str, int], int] = {}
    for d in chunks:
        source = str(d.metadata.get("source", "unknown"))
        page0 = int(d.metadata.get("page", 0))
        page = page0 + 1
        key = (source, page)
        per_source_page_counter[key] = per_source_page_counter.get(key, 0) + 1
        cnum = per_source_page_counter[key]

        # Keep your existing style: filename:p{page}:c{chunk}
        fname = Path(source).name
        d.metadata["chunk_id"] = f"{fname}:p{page}:c{cnum}"
        d.metadata["page"] = page  # store 1-index

    return chunks
