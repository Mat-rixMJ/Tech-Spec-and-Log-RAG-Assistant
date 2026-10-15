"""
Document ingestion + retrieval for the knowledge base (support/policy docs).

Text is split into overlapping chunks, vectorized with TF-IDF, and kept in an
in-memory store keyed by a document id. Retrieval is cosine similarity.

ponytail: in-memory, single-process store with TF-IDF retrieval. Ceiling —
documents vanish on restart and it won't scale past one worker or very large
corpora. Upgrade path: swap TfidfVectorizer + this dict for a persistent vector
DB (e.g. Chroma) behind the same ingest()/retrieve() interface.
"""

from __future__ import annotations

import uuid

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping character windows on whitespace boundaries."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return [c for c in chunks if c]


class _Doc:
    def __init__(self, name: str, chunks: list[str]):
        self.name = name
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(chunks)


class DocumentStore:
    def __init__(self):
        self._docs: dict[str, _Doc] = {}

    def ingest(self, name: str, text: str) -> dict:
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No extractable text found in the document.")
        doc_id = uuid.uuid4().hex[:12]
        self._docs[doc_id] = _Doc(name, chunks)
        return {"doc_id": doc_id, "name": name, "num_chunks": len(chunks)}

    def exists(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def retrieve(self, doc_id: str, query: str, k: int = 4) -> list[dict]:
        doc = self._docs.get(doc_id)
        if doc is None:
            raise KeyError(doc_id)
        q_vec = doc.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, doc.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = [i for i in ranked[:k] if scores[i] > 0]
        # Small-doc fallback so the LLM always gets context.
        if not top and len(doc.chunks) <= k:
            top = ranked[: len(doc.chunks)]
        return [
            {"text": doc.chunks[i], "score": round(float(scores[i]), 4)} for i in top
        ]


store = DocumentStore()
