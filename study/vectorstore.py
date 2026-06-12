"""Persistent ChromaDB collection holding chunk embeddings + metadata.

We pass our own (bge) embeddings explicitly, so Chroma does no embedding itself.
Metadata filtering (``where``) is what powers intent-aware retrieval, e.g.
restricting to ``type == 'image'`` for a "what's in the flow chart" question.
"""
from __future__ import annotations

from typing import List, Optional

import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME, ensure_dirs
from .schema import Chunk


class VectorStore:
    def __init__(self):
        ensure_dirs()
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection (full re-index)."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def delete_source(self, source: str) -> None:
        """Remove all chunks belonging to one file (for incremental re-index)."""
        try:
            self.collection.delete(where={"source": source})
        except Exception as e:  # pragma: no cover
            print(f"[vectorstore] delete_source({source}) failed: {e}")

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
            embeddings=embeddings,
        )

    def query(self, query_embedding: List[float], n_results: int = 8,
              where: Optional[dict] = None) -> List[dict]:
        """Return hits as dicts: {text, metadata, distance}."""
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"text": doc, "metadata": meta, "distance": dist})
        return hits
