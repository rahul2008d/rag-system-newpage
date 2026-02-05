from langchain_core.documents import Document

from rag.chunking import chunk_pages


def test_chunking_adds_chunk_metadata() -> None:
    docs = [
        Document(
            page_content="A" * 2000,
            metadata={"source": "file.txt", "page": 0},
        )
    ]

    chunks = chunk_pages(docs, chunk_size=500, overlap=100)

    assert len(chunks) > 1
    assert "chunk_id" in chunks[0].metadata
    assert chunks[0].metadata["page"] == 1
