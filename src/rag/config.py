from dataclasses import dataclass
import os

from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

    chunk_size: int = int(os.environ.get("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.environ.get("TOP_K", "5"))

    min_score: float = float(os.environ.get("MIN_SCORE", "0.25"))

    store_dir: str = os.environ.get("STORE_DIR", "vector_store")
