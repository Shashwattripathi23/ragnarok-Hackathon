"""Turn parsed pages into retrieval-ready Chunks.

Strategy (structure-first, then size):
  1. Split each page's markdown on its own headings into coherent sections,
     refining the page's TOC breadcrumb with the in-page heading.
  2. Keep markdown tables atomic (slicing a table makes it unparseable).
  3. Window long prose into ~CHUNK_CHARS pieces with overlap, snapping cuts to
     whitespace so we never split mid-word.
  4. Merge tiny trailing fragments into the previous chunk.

Image chunks are created here too, but with empty text — their caption is
filled in later by ``study.images`` (which needs Groq).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .config import CHUNK_CHARS, CHUNK_OVERLAP_CHARS, MIN_CHUNK_CHARS
from .schema import Chunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _merge_heading(base_path: str, md_heading: str) -> str:
    """Append an in-page markdown heading to the TOC breadcrumb (deduped)."""
    if not md_heading:
        return base_path
    if not base_path:
        return md_heading
    if md_heading.lower() in base_path.lower():
        return base_path
    return f"{base_path} > {md_heading}"


def _split_sections(text: str, base_path: str) -> List[Tuple[str, str]]:
    """Split markdown into (heading_path, body) sections on heading lines."""
    sections: List[Tuple[str, List[str]]] = []
    current_path = base_path
    buf: List[str] = []

    def flush():
        if buf:
            sections.append((current_path, "\n".join(buf).strip()))

    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            flush()
            buf = []
            current_path = _merge_heading(base_path, m.group(2).strip())
        else:
            buf.append(line)
    flush()
    return [(p, b) for p, b in sections if b]


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _split_tables(body: str) -> List[Tuple[str, bool]]:
    """Separate a body into (segment, is_table) parts, keeping tables whole."""
    segments: List[Tuple[str, bool]] = []
    prose: List[str] = []
    table: List[str] = []

    def flush_prose():
        if prose:
            segments.append(("\n".join(prose).strip(), False))
            prose.clear()

    def flush_table():
        if table:
            segments.append(("\n".join(table).strip(), True))
            table.clear()

    for line in body.splitlines():
        if _is_table_row(line):
            flush_prose()
            table.append(line)
        else:
            flush_table()
            prose.append(line)
    flush_table()
    flush_prose()
    return [(seg, t) for seg, t in segments if seg]


def _window(text: str, size: int, overlap: int) -> List[str]:
    """Sliding char window with whitespace-snapped cuts and a small-tail merge."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    step = max(1, size - overlap)
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:  # snap to the last whitespace in the back half of the window
            ws = max(text.rfind(" ", i + step, end), text.rfind("\n", i + step, end))
            if ws > i:
                end = ws
        piece = text[i:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        i = max(end - overlap, i + 1)

    # Merge a too-small final fragment into the previous chunk.
    if len(out) >= 2 and len(out[-1]) < MIN_CHUNK_CHARS:
        out[-2] = f"{out[-2]}\n{out.pop()}"
    return out


def _chunk_page_text(text: str, source: str, source_path: str, page: int,
                     headings: dict, counter: List[int]) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not text or not text.strip():
        return chunks
    base_path = headings.get("heading_path", "")

    for sec_path, body in _split_sections(text, base_path):
        for segment, is_table in _split_tables(body):
            pieces = [segment] if is_table else _window(segment, CHUNK_CHARS, CHUNK_OVERLAP_CHARS)
            for piece in pieces:
                if len(piece) < MIN_CHUNK_CHARS and not is_table:
                    continue
                counter[0] += 1
                chunks.append(Chunk(
                    chunk_id=f"{source}::p{page}::c{counter[0]}",
                    source=source,
                    source_path=source_path,
                    page=page,
                    text=piece,
                    type="table" if is_table else "text",
                    chapter=headings.get("chapter", ""),
                    section=headings.get("section", ""),
                    subtitle=headings.get("subtitle", ""),
                    heading_path=sec_path,
                ))
    return chunks


def chunk_document(parsed: dict) -> List[Chunk]:
    """Split a parsed document into text/table chunks."""
    source = parsed["source"]
    source_path = parsed.get("source_path", "")
    chunks: List[Chunk] = []
    counter = [0]

    for pg in parsed["pages"]:
        headings = {
            "chapter": pg["chapter"],
            "section": pg["section"],
            "subtitle": pg["subtitle"],
            "heading_path": pg["heading_path"],
        }
        chunks.extend(
            _chunk_page_text(pg["text"], source, source_path, pg["page"], headings, counter)
        )
    return chunks
