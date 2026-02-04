from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from openai import OpenAI

from .config import Settings
from .loaders import load_files
from .chunking import chunk_pages, Chunk
from .store import save_store, load_store, normalize

@dataclass
class SourceHit:
    chunk_id: str
    source: str
    page: int
    score: float
    text: str

class RAGSystem:
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.s = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.index: faiss.Index | None = None
        self.chunks: List[Chunk] = []

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        # Batch embeddings call
        resp = self.client.embeddings.create(model=self.s.embedding_model, input=texts)
        vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return normalize(vectors)

    def ingest(self, files: List[Path]) -> Dict[str, Any]:
        pages = load_files(files)
        chunks = chunk_pages(pages, self.s.chunk_size, self.s.chunk_overlap)
        if not chunks:
            return {"chunks_indexed": 0, "message": "No extractable text found."}

        embeddings = self._embed_texts([c.text for c in chunks])
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.chunks = chunks
        save_store(self.s.store_dir, index, chunks)

        return {"chunks_indexed": len(chunks), "files": sorted({c.source for c in chunks})}

    def load(self) -> None:
        self.index, self.chunks = load_store(self.s.store_dir)

    def retrieve(self, query: str) -> Tuple[List[SourceHit], float]:
        if not self.index:
            raise ValueError("Index not loaded. Ingest first.")
        q = self._embed_texts([query])
        scores, idx = self.index.search(q, self.s.top_k)  # scores shape (1, k)
        hits: List[SourceHit] = []
        best = float(scores[0][0]) if len(scores[0]) else 0.0

        for score, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i == -1:
                continue
            c = self.chunks[i]
            hits.append(SourceHit(
                chunk_id=c.chunk_id, source=c.source, page=c.page, score=float(score), text=c.text
            ))
        return hits, best

    def answer(self, query: str, chat_history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        hits, best = self.retrieve(query)

        if not hits or best < self.s.min_score:
            return {
                "answer": "I don’t know based on the provided documents.",
                "sources": [],
                "best_score": best,
            }

        context_lines = []
        for h in hits:
            context_lines.append(f"[{h.chunk_id}] (score={h.score:.3f})\n{h.text}")

        system_msg = (
            "You are a helpful assistant. Answer ONLY using the provided context. "
            "If the answer is not in the context, say you don't know. "
            "Cite sources inline using the chunk id in square brackets, e.g. [file.pdf:p2:c1]."
        )

        user_msg = (
            "Context:\n"
            + "\n\n".join(context_lines)
            + f"\n\nQuestion: {query}\nAnswer:"
        )

        messages = [{"role": "system", "content": system_msg}]
        if chat_history:
            # Keep history minimal; it’s optional for the assignment
            messages.extend(chat_history[-6:])
        messages.append({"role": "user", "content": user_msg})

        resp = self.client.chat.completions.create(
            model=self.s.chat_model,
            messages=messages,
            temperature=0.2,
        )

        return {
            "answer": resp.choices[0].message.content,
            "sources": [
                {"chunk_id": h.chunk_id, "file": h.source, "page": h.page, "score": h.score}
                for h in hits
            ],
            "best_score": best,
        }
