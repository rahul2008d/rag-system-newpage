from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag.store import build_vectorstore, save_store, load_store


def test_faiss_store_roundtrip(tmp_path) -> None:
    docs = [
        Document(
            page_content="Test document",
            metadata={"source": "test.txt", "page": 1},
        )
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vs = build_vectorstore(docs, embeddings)
    save_store(str(tmp_path), vs)

    loaded = load_store(str(tmp_path), embeddings)

    results = loaded.similarity_search("test")
    assert len(results) == 1
