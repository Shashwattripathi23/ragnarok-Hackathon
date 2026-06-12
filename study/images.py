"""Caption extracted images with a Groq vision model so they become searchable.

Each image chunk's text is replaced with a concise caption that, for diagrams /
flowcharts / charts, names the concept or algorithm illustrated and its key
elements. This is what lets a query like "what algorithm is in the flow chart on
slide X" retrieve the right image. Decorative images are dropped.

Captioning is I/O-bound (network), so we use a thread pool with backoff. It runs
ONCE at index time only.
"""
from __future__ import annotations

import base64
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from PIL import Image

from .config import CAPTION_MAX_IMAGE_DIM, CAPTION_MAX_WORKERS, VISION_MODEL
from .llm import get_client, with_retry
from .schema import Chunk

_MAX_DIM = CAPTION_MAX_IMAGE_DIM  # downscale before sending to cap tokens/payload size
_NON_INFORMATIVE = "NON_INFORMATIVE"

_CAPTION_PROMPT = (
    "You are labeling an image extracted from university course material so it "
    "can be found by search. Describe it in 2-4 concise sentences. If it is a "
    "diagram, flow chart, chart, or illustration, explicitly name the concept or "
    "algorithm it depicts and list its key elements or steps.\n"
    f"ONLY IF the image is purely decorative (a logo, background, icon, or plain "
    f"bullet) with no informational content, your ENTIRE reply must be the single "
    f"word {_NON_INFORMATIVE} and nothing else. Otherwise never use that word."
)


def _clean_caption(caption: str) -> Optional[str]:
    """Validate/clean a caption; drop decorative ones.

    The model occasionally appends the sentinel after a real caption, so we
    strip a stray sentinel rather than discarding a useful description.
    """
    if not caption:
        return None
    if caption.strip().upper() == _NON_INFORMATIVE:
        return None
    cleaned = re.sub(_NON_INFORMATIVE, "", caption, flags=re.IGNORECASE).strip()
    return cleaned if len(cleaned) >= 15 else None


def _encode_image(path: str) -> Optional[str]:
    """Load, downscale, and base64-encode an image as a PNG data URI."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:  # pragma: no cover
        print(f"[images] cannot open {path}: {e}")
        return None
    if max(img.size) > _MAX_DIM:
        img.thumbnail((_MAX_DIM, _MAX_DIM))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _caption_one(chunk: Chunk) -> Optional[str]:
    data_uri = _encode_image(chunk.image_path)
    if data_uri is None:
        return None
    client = get_client()

    def call():
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            temperature=0.2,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()

    try:
        caption = with_retry(call)
    except Exception as e:  # noqa: BLE001
        print(f"[images] caption failed for {chunk.image_path}: {e}")
        return None
    return _clean_caption(caption)


def caption_chunks(image_chunks: List[Chunk],
                   progress: Optional[Callable[[int, int], None]] = None) -> List[Chunk]:
    """Caption image chunks in parallel; return only the informative ones."""
    if not image_chunks:
        return []
    kept: List[Chunk] = []
    total = len(image_chunks)
    done = 0
    with ThreadPoolExecutor(max_workers=CAPTION_MAX_WORKERS) as pool:
        futures = {pool.submit(_caption_one, c): c for c in image_chunks}
        for fut in as_completed(futures):
            chunk = futures[fut]
            caption = fut.result()
            if caption:
                chunk.text = caption
                kept.append(chunk)
            done += 1
            if progress:
                progress(done, total)
    return kept
