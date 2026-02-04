from __future__ import annotations
from dataclasses import dataclass
from typing import List
import re

from .loaders import DocPage

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page: int
    text: str

def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    # simple paragraph-ish normalization
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]

def chunk_pages(pages: List[DocPage], chunk_size: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page in pages:
        parts = _split_text(page.text, chunk_size, overlap)
        for j, part in enumerate(parts, start=1):
            cid = f"{page.source}:p{page.page}:c{j}"
            chunks.append(Chunk(chunk_id=cid, source=page.source, page=page.page, text=part))
    return chunks
