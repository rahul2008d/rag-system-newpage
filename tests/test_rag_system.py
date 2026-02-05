import pytest
from pathlib import Path

from rag.config import Settings
from rag.system import RAGSystem


def test_query_without_ingest_raises() -> None:
    settings = Settings(openai_api_key="test-key")
    rag = RAGSystem(settings)

    with pytest.raises(ValueError):
        rag.retrieve("What is this?")
