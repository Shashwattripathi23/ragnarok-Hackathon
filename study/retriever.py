"""Embed a query, search Chroma, and format hits for the LLM + the UI."""
from __future__ import annotations

from typing import List, Optional

from .config import TOP_K
from .embeddings import embed_query
from .vectorstore import VectorStore


def _citation(meta: dict) -> str:
    loc = f"{meta.get('source', '?')} p.{meta.get('page', '?')}"
    hp = meta.get("heading_path") or meta.get("chapter") or ""
    if hp:
        loc += f" ({hp})"
    return loc


class Retriever:
    def __init__(self, store: Optional[VectorStore] = None):
        self.store = store or VectorStore()

    def search(self, query: str, k: int = TOP_K, modality: str = "any") -> List[dict]:
        """Return hits with text + metadata + a ready-made citation label."""
        where = None
        if modality == "image":
            where = {"type": "image"}
        elif modality == "text":
            # text or table, excluding images
            where = {"type": {"$in": ["text", "table"]}}

        hits = self.store.query(embed_query(query), n_results=k, where=where)
        for h in hits:
            h["citation"] = _citation(h["metadata"])
        return hits

    @staticmethod
    def format_context(hits: List[dict]) -> str:
        """Render hits as a numbered, citation-tagged context block for the LLM."""
        if not hits:
            return "No relevant passages were found in the course materials."
        blocks = []
        for i, h in enumerate(hits, 1):
            tag = "IMAGE" if h["metadata"].get("type") == "image" else "TEXT"
            blocks.append(f"[{i}] ({tag}) Source: {h['citation']}\n{h['text']}")
        return "\n\n".join(blocks)
