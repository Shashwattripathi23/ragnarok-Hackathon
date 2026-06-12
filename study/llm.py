"""Shared Groq client + a small retry helper (free tier => rate limits happen)."""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from .config import ROOT


@lru_cache(maxsize=1)
def get_client() -> Groq:
    """Singleton Groq client. Key comes from MiniHack/.env (never hardcoded)."""
    load_dotenv(ROOT / ".env")
    # Groq() reads GROQ_API_KEY from the environment.
    client = Groq()
    if not client.api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Create MiniHack/.env with GROQ_API_KEY=gsk_..."
        )
    return client


def _retry_after(e) -> Optional[float]:
    """Honor the server's Retry-After hint on a 429, if present."""
    resp = getattr(e, "response", None)
    if resp is not None:
        val = resp.headers.get("retry-after")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
    return None


def with_retry(fn, *, tries: int = 5, base_delay: float = 3.0):
    """Call ``fn`` with exponential backoff. Re-raises the last error if all fail.

    Targets Groq free-tier rate limits without a hard dependency on the
    exception type (the SDK surfaces these as APIStatusError subclasses). When
    the server sends a Retry-After header we wait exactly that long (+ jitter).
    """
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - back off on any transient error
            last = e
            status = getattr(e, "status_code", None)
            transient = status in (429, 500, 502, 503, None)
            if attempt == tries - 1 or not transient:
                raise
            delay = _retry_after(e) or base_delay * (2 ** attempt)
            time.sleep(delay + 0.5)
    raise last  # pragma: no cover
