"""Local sentence-transformer embeddings.

Groq has no embeddings endpoint, so embeddings run locally on CPU. bge-small is
a strong, light default. Vectors are L2-normalized so cosine == dot product.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from .config import EMBED_MODEL, EMBED_QUERY_PREFIX


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so the heavy torch import only happens when needed.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def embed_passages(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed document/passage texts for indexing."""
    if not texts:
        return []
    vecs = _model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a search query (bge wants the retrieval instruction prefix)."""
    vec = _model().encode(
        [EMBED_QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False
    )
    return vec[0].tolist()


def warm_up() -> None:
    """Trigger model download/load up front (e.g. on app start)."""
    _model()
