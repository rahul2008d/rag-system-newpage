from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import json

from .chunking import Chunk

def ensure_dir(d: str) -> Path:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_store(store_dir: str, index: faiss.Index, chunks: List[Chunk]) -> None:
    p = ensure_dir(store_dir)
    faiss.write_index(index, str(p / "index.faiss"))

    # store chunks w/ metadata in jsonl for easy debugging + review
    with (p / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

def load_store(store_dir: str) -> Tuple[faiss.Index, List[Chunk]]:
    p = Path(store_dir)
    index_path = p / "index.faiss"
    chunks_path = p / "chunks.jsonl"
    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Vector store not found. Ingest docs first.")

    index = faiss.read_index(str(index_path))

    chunks: List[Chunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(Chunk(**obj))
    return index, chunks

def normalize(vectors: np.ndarray) -> np.ndarray:
    # in-place normalize for cosine via inner product
    faiss.normalize_L2(vectors)
    return vectors
