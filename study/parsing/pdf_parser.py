"""Parse a single PDF into per-page text + heading metadata.

Network-free and CPU-only so it can run inside a ``ProcessPoolExecutor`` worker;
returns plain, picklable dicts. Per page we capture markdown text (via
pymupdf4llm — headings, tables, reading order) and the chapter/section/subtitle
breadcrumb (from the PDF's TOC, with a font-size title fallback for slide decks).

Images are NOT extracted or captioned — retrieval is over native slide/book
text. The UI renders the cited page on demand for display.
"""
from __future__ import annotations

import os
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm

from .toc import build_resolver, font_title

# pymupdf4llm defaults to "layout mode": a heavy ML layout model that OCRs EVERY
# page (slow, noisy, and degrades already-digital text). University course PDFs
# almost always have a real text layer, so we use the classic OCR-free extractor.
# Opt back into layout+OCR for genuinely scanned PDFs via STUDY_USE_LAYOUT=1.
try:
    pymupdf4llm.use_layout(os.getenv("STUDY_USE_LAYOUT", "0") == "1")
except Exception:
    pass

# Silence MuPDF's own warning spam (broken-glyph, tiny-image, etc.).
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass


def parse_pdf(path: str, _unused: str = "") -> dict:
    """Parse one PDF. Returns a picklable dict (safe for ProcessPool workers)."""
    abspath = os.path.abspath(path)
    source = os.path.basename(path)

    doc = fitz.open(path)
    resolver = build_resolver(doc)

    # Per-page markdown (headings, tables, reading order).
    try:
        md_pages = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    except Exception as e:
        print(f"[parser] markdown failed for {source}: {e}; falling back to plain text")
        md_pages = [{"text": doc[i].get_text()} for i in range(doc.page_count)]

    pages = []
    for i in range(doc.page_count):
        page = doc[i]
        page_no = i + 1
        text = (md_pages[i].get("text") if i < len(md_pages) else "") or ""

        headings = resolver.resolve(page_no)
        # Slide decks rarely have bookmarks: fall back to the big-font title.
        if not resolver.has_toc:
            title = font_title(page)
            if title:
                headings.subtitle = title

        pages.append({
            "page": page_no,
            "text": text,
            "chapter": headings.chapter,
            "section": headings.section,
            "subtitle": headings.subtitle,
            "heading_path": headings.path,
        })

    n_pages = doc.page_count
    doc.close()
    return {"source": source, "source_path": abspath, "n_pages": n_pages, "pages": pages}
