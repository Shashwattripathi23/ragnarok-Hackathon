"""Central configuration: models, paths, and chunking parameters.

Everything here is intentionally overridable via environment variables so the
evaluators can swap in their own Groq key / models without code changes.
"""
import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # MiniHack/
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / ".index"))
CHROMA_DIR = INDEX_DIR / "chroma"
IMAGES_DIR = INDEX_DIR / "images"
CACHE_FILE = INDEX_DIR / "file_cache.json"
COLLECTION_NAME = "course_materials"

# --- Groq models (verify current IDs at console.groq.com/docs/rate-limits) -
# Groq is the ONLY allowed API. Note: Groq has no embeddings endpoint, so the
# embedding model below runs locally instead.
GEN_MODEL = os.getenv("GROQ_GEN_MODEL", "llama-3.3-70b-versatile")
INTENT_MODEL = os.getenv("GROQ_INTENT_MODEL", "llama-3.1-8b-instant")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# --- Local embedding model -------------------------------------------------
# bge-small: 384-dim, strong on CPU. Truncates at 512 tokens, so we keep
# chunks well under that (see CHUNK_CHARS).
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models expect this prefix on the *query* side for retrieval.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Chunking --------------------------------------------------------------
# ~1 token ≈ 4 chars. 1500 chars ≈ ~375 tokens, safely under bge's 512 cap.
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1500"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))
MIN_CHUNK_CHARS = 120          # fragments smaller than this merge into the previous chunk

# --- Retrieval -------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "8"))

# --- Image captioning ------------------------------------------------------
# Vision calls are token-heavy (~2-3k tokens each). The free tier's 30k TPM
# limit is easy to burst past, so keep concurrency low and images small.
CAPTION_MAX_WORKERS = int(os.getenv("CAPTION_MAX_WORKERS", "2"))
CAPTION_MAX_IMAGE_DIM = int(os.getenv("CAPTION_MAX_IMAGE_DIM", "768"))
MIN_IMAGE_PIXELS = 100 * 100   # skip tiny logos/bullets/decorations


def ensure_dirs() -> None:
    """Create the on-disk index directories if missing."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
