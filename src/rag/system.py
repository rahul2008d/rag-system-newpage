from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .chunking import chunk_pages
from .config import Settings
from .loaders import load_files
from .store import build_vectorstore, load_store, save_store


@dataclass
class SourceHit:
    """A retrieved source chunk with score."""

    chunk_id: str
    source: str
    page: int
    score: float
    text: str


class RAGSystem:
    """Retrieval-Augmented Generation system (LangChain-based)."""

    def __init__(self, settings: Settings):
        """Initialize RAG system with settings and OpenAI clients."""
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

        self.s = settings
        self.embeddings = OpenAIEmbeddings(model=self.s.embedding_model)
        self.llm = ChatOpenAI(model=self.s.chat_model, temperature=0.2)

        self.vs: FAISS | None = None

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Answer ONLY using the provided context. "
                "If the answer is not in the context, say you don't know. "
                "Cite sources inline using the chunk id in square brackets, e.g. [file.pdf:p2:c1].",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {input}\nAnswer:"),
        ])

    def _format_docs(self, docs: List[Document]) -> str:
        """Format docs for prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def ingest(self, files: List[Path]) -> Dict[str, Any]:
        """Ingest files, split into chunks, embed, and build/persist a FAISS store."""
        pages = load_files(files)
        if not pages:
            return {
                "chunks_indexed": 0,
                "message": "No extractable text found.",
            }

        chunks = chunk_pages(pages, self.s.chunk_size, self.s.chunk_overlap)
        if not chunks:
            return {
                "chunks_indexed": 0,
                "message": "No extractable text found.",
            }

        self.vs = build_vectorstore(chunks, self.embeddings)
        save_store(self.s.store_dir, self.vs)

        return {
            "chunks_indexed": len(chunks),
            "files": sorted({
                Path(d.metadata.get("source", "")).name for d in chunks
            }),
        }

    def load(self) -> None:
        """Load persisted FAISS store from disk."""
        path = Path(self.s.store_dir)
        if not (path / "index.faiss").exists():
            print("⚠️ No index found. Run ingest first.")
            return
        self.vs = load_store(
            self.s.store_dir, self.embeddings, allow_unsafe=True
        )
        # Rebuild retriever and chain after load
        if self.vs:
            retriever = self.vs.as_retriever(search_kwargs={"k": self.s.top_k})
            self.rag_chain = (
                {
                    "context": retriever | self._format_docs,
                    "input": RunnablePassthrough(),
                }
                | self.prompt
                | self.llm
            )

    def retrieve(self, query: str) -> Tuple[List[SourceHit], float]:
        """Retrieve relevant chunks and return hits + best score."""
        if self.vs is None:
            raise ValueError("Vector store not loaded. Ingest first.")

        docs_scores = self.vs.similarity_search_with_relevance_scores(
            query, k=self.s.top_k
        )

        hits: List[SourceHit] = []
        best = 0.0
        for doc, distance in docs_scores:
            score = 1.0 - (
                float(distance) ** 2 / 2.0
            )  # Convert cosine distance to similarity
            best = max(best, score)
            hits.append(
                SourceHit(
                    chunk_id=str(doc.metadata.get("chunk_id", "")),
                    source=Path(str(doc.metadata.get("source", ""))).name,
                    page=int(doc.metadata.get("page", 0)),
                    score=score,
                    text=doc.page_content,
                )
            )
        if not hits or best < self.s.min_score:
            return [], best
        return hits, best

    def answer(
        self, query: str, chat_history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any]:
        """Answer using RAG + recent chat history."""
        hits, best = self.retrieve(query)

        if not hits or best < self.s.min_score:
            return {
                "answer": "I don't know based on the provided documents.",
                "sources": [],
                "best_score": best,
            }

        # Convert history to LangChain messages
        history_msgs = []
        if chat_history:
            recent_history = chat_history[-6:]  # Last 6 turns (3 Q&A)
            for msg in recent_history:
                role = msg["role"]
                if role == "user":
                    history_msgs.append(HumanMessage(content=msg["content"]))
                elif role == "assistant":
                    history_msgs.append(AIMessage(content=msg["content"]))

        # Updated prompt with history + context
        contextual_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Use chat history and context to answer. "
                "Answer ONLY using the provided context if available. "
                "If the answer is not in the context, say you don't know. "
                "Cite sources inline using the chunk id in square brackets, e.g. [file.pdf:p2:c1].",
            ),
            *history_msgs,  # Recent conversation history
            (
                "human",
                "Context:\n{context}\n\nCurrent question: {input}\nAnswer:",
            ),
        ])

        # Rebuild chain with contextual prompt
        rag_chain_with_history = (
            {
                "context": self.vs.as_retriever(
                    search_kwargs={"k": self.s.top_k}
                )
                | self._format_docs,
                "input": RunnablePassthrough(),
            }
            | contextual_prompt
            | self.llm
        )

        result = rag_chain_with_history.invoke(query)
        answer_text = (
            result.content if hasattr(result, "content") else str(result)
        )

        return {
            "answer": answer_text,
            "sources": [
                {
                    "chunk_id": h.chunk_id,
                    "file": h.source,
                    "page": h.page,
                    "score": h.score,
                }
                for h in hits
            ],
            "best_score": best,
        }
