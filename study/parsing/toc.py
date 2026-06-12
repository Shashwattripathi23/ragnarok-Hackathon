"""Resolve chapter / section / subtitle context for any page of a PDF.

Primary signal: the PDF's own table of contents (bookmarks) via
``doc.get_toc()``. For a given page, the chapter is the most recent level-1
bookmark at or before that page, the section the most recent level-2, etc.

Fallback (no bookmarks — common for slide decks): infer a page title from the
largest-font text spans on the page.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF


@dataclass
class PageHeadings:
    chapter: str = ""
    section: str = ""
    subtitle: str = ""

    @property
    def path(self) -> str:
        parts = [p for p in (self.chapter, self.section, self.subtitle) if p]
        return " > ".join(parts)


class TocResolver:
    """Maps a 1-based page number to its heading breadcrumb."""

    def __init__(self, toc: List[Tuple[int, str, int]]):
        # toc entries are [level, title, page] (1-based page).
        self._entries = sorted(
            ((lvl, title.strip(), page) for lvl, title, page in toc if title.strip()),
            key=lambda e: e[2],
        )
        self.has_toc = bool(self._entries)

    def resolve(self, page: int) -> PageHeadings:
        h = PageHeadings()
        for lvl, title, p in self._entries:
            if p > page:
                break
            if lvl == 1:
                h.chapter, h.section, h.subtitle = title, "", ""
            elif lvl == 2:
                h.section, h.subtitle = title, ""
            elif lvl >= 3:
                h.subtitle = title
        return h


def _looks_like_title(text: str) -> bool:
    """Filter out stylized-font noise ('op', 'BS<') that isn't a real title."""
    if not (4 <= len(text) <= 120):
        return False
    alpha = sum(c.isalpha() for c in text)
    return alpha >= 4 and alpha / len(text) >= 0.5


def font_title(page: "fitz.Page") -> str:
    """Best-effort page title from the largest-font line (slide-deck fallback)."""
    try:
        data = page.get_text("dict")
    except Exception:
        return ""
    best_size, best_text = 0.0, ""
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            size = max(s.get("size", 0) for s in spans)
            text = "".join(s.get("text", "") for s in spans).strip()
            if size > best_size and _looks_like_title(text):
                best_size, best_text = size, text
    return best_text


def build_resolver(doc: "fitz.Document") -> TocResolver:
    try:
        toc = doc.get_toc(simple=True)
    except Exception:
        toc = []
    return TocResolver(toc)
