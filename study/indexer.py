"""Build the index: parallel parse -> caption images -> embed -> ChromaDB.

Pipeline stages (each reported via the optional ``on_progress`` callback):
  parse    PDFs parsed in a process pool (CPU-bound, GIL -> processes)
  caption  image chunks captioned via Groq vision (thread pool, I/O-bound)
  embed    all chunks embedded locally (bge-small) and upserted to Chroma

A per-file content hash cache lets re-runs skip unchanged files.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Optional

from .config import CACHE_FILE, IMAGES_DIR, ensure_dirs
from .chunking import chunk_document
from .embeddings import embed_passages
from .images import caption_chunks
from .parsing.pdf_parser import parse_pdf
from .vectorstore import VectorStore

# on_progress(stage: str, current: int, total: int)
ProgressFn = Callable[[str, int, int], None]


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache))
    except Exception as e:  # pragma: no cover
        print(f"[indexer] could not save cache: {e}")


def _executor(mode: str):
    """Pick a parallel executor. 'process' is fastest for CPU-bound parsing but
    can be fragile under Streamlit's spawn; 'thread'/'sequential' are fallbacks
    (set STUDY_PARSE_MODE to override)."""
    if mode == "thread":
        from concurrent.futures import ThreadPoolExecutor
        return ThreadPoolExecutor()
    return ProcessPoolExecutor()


def _parse_all(paths: List[str], on_progress: Optional[ProgressFn]) -> List[dict]:
    """Parse PDFs in parallel, falling back to sequential on failure."""
    results: List[dict] = []
    total = len(paths)
    mode = os.getenv("STUDY_PARSE_MODE", "process")
    if mode == "sequential":
        for i, p in enumerate(paths, 1):
            try:
                results.append(parse_pdf(p, str(IMAGES_DIR)))
            except Exception as e:  # noqa: BLE001
                print(f"[indexer] parse failed for {p}: {e}")
            if on_progress:
                on_progress("parse", i, total)
        return results
    try:
        with _executor(mode) as ex:
            futures = {ex.submit(parse_pdf, p, str(IMAGES_DIR)): p for p in paths}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    print(f"[indexer] parse failed for {futures[fut]}: {e}")
                if on_progress:
                    on_progress("parse", i, total)
    except Exception as e:  # noqa: BLE001 - process pool unavailable -> sequential
        print(f"[indexer] process pool failed ({e}); parsing sequentially")
        results = []
        for i, p in enumerate(paths, 1):
            try:
                results.append(parse_pdf(p, str(IMAGES_DIR)))
            except Exception as ex2:  # noqa: BLE001
                print(f"[indexer] parse failed for {p}: {ex2}")
            if on_progress:
                on_progress("parse", i, total)
    return results


def index_files(paths: List[str], on_progress: Optional[ProgressFn] = None,
                rebuild: bool = False) -> dict:
    """Index a list of PDF paths. Returns a summary dict."""
    ensure_dirs()
    paths = [os.path.abspath(p) for p in paths if p.lower().endswith(".pdf")]
    vs = VectorStore()

    cache = {} if rebuild else _load_cache()
    if rebuild:
        vs.reset()

    todo = [p for p in paths if rebuild or cache.get(p) != _file_hash(p)]
    skipped = len(paths) - len(todo)
    if not todo:
        return {"indexed_files": 0, "skipped": skipped, "chunks": 0,
                "images": 0, "total_in_store": vs.count()}

    # 1) Parse (parallel)
    parsed_docs = _parse_all(todo, on_progress)

    # 2) Chunk every doc; collect image chunks for one batched caption pass.
    doc_text = []                       # (source, [text/table chunks])
    all_images = []
    for parsed in parsed_docs:
        text_chunks, image_chunks = chunk_document(parsed)
        doc_text.append((parsed["source"], text_chunks))
        all_images.extend(image_chunks)

    # 3) Caption images (thread pool, with backoff for rate limits)
    if all_images and on_progress:
        on_progress("caption", 0, len(all_images))
    kept_images = caption_chunks(
        all_images,
        progress=(lambda d, t: on_progress("caption", d, t)) if on_progress else None,
    )
    images_by_source = defaultdict(list)
    for c in kept_images:
        images_by_source[c.source].append(c)

    # 4) Embed + upsert, per source (so incremental re-index can replace cleanly)
    total_chunks, total_images = 0, len(kept_images)
    n_sources = len(doc_text)
    for i, (source, text_chunks) in enumerate(doc_text, 1):
        chunks = text_chunks + images_by_source.get(source, [])
        if chunks:
            vs.delete_source(source)
            embeddings = embed_passages([c.embed_text() for c in chunks])
            vs.add(chunks, embeddings)
            total_chunks += len(chunks)
        # mark this file as indexed in the cache
        for p in todo:
            if os.path.basename(p) == source:
                cache[p] = _file_hash(p)
        if on_progress:
            on_progress("embed", i, n_sources)

    _save_cache(cache)
    return {
        "indexed_files": len(doc_text),
        "skipped": skipped,
        "chunks": total_chunks,
        "images": total_images,
        "total_in_store": vs.count(),
    }
