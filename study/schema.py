"""The Chunk: the single retrieval unit that flows through the whole pipeline.

A Chunk carries both the human-readable text (for display/citation) and the
text we actually embed (which has the heading breadcrumb prepended so isolated
paragraphs still embed meaningfully). Metadata is sanitized for ChromaDB, which
only accepts str/int/float/bool values (no None, no lists).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


def _clean(value) -> object:
    """Coerce a value into something ChromaDB metadata accepts."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return " > ".join(str(v) for v in value if v)
    return str(value)


@dataclass
class Chunk:
    chunk_id: str
    source: str                       # filename, e.g. "Lecture5.pdf"
    page: int                         # 1-based page/slide number
    text: str                         # raw text shown to the user / cited
    type: str = "text"                # "text" | "table"
    chapter: str = ""
    section: str = ""
    subtitle: str = ""
    heading_path: str = ""            # "Ch 3 Neural Nets > 3.2 Backprop"
    source_path: str = ""             # absolute path to the PDF (for on-demand page render)

    def embed_text(self) -> str:
        """Text fed to the embedder: breadcrumb + source context + body.

        The breadcrumb disambiguates otherwise-generic passages so they land
        near the right queries in vector space.
        """
        crumbs = [c for c in (self.source, self.heading_path) if c]
        prefix = f"[{' > '.join(crumbs)}]\n" if crumbs else ""
        return prefix + self.text

    def metadata(self) -> dict:
        """Chroma-safe metadata dict (everything except the body text)."""
        d = asdict(self)
        d.pop("text", None)
        return {k: _clean(v) for k, v in d.items()}

    def citation(self) -> str:
        """Short human-readable source label for grounding answers."""
        loc = f"{self.source} p.{self.page}"
        if self.heading_path:
            loc += f" ({self.heading_path})"
        return loc
