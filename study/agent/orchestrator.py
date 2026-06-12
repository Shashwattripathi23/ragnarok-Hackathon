"""The agent loop: classify intent -> (forced) retrieval tool call -> grounded answer.

Token budget per query: 1 cheap intent call + 1 generation that issues at most
one retrieval tool call + 1 final generation. Retrieval is *forced* for the
knowledge intents so answers are always grounded; for ``user_response`` the model
may answer directly (e.g. grading a quiz reply).
"""
from __future__ import annotations

import json
from typing import List, Optional

from ..config import GEN_MODEL
from ..llm import get_client, with_retry
from ..retriever import Retriever
from .intent import classify
from .tools import SEARCH_TOOL, TOOL_NAME

_MAX_HISTORY = 6  # recent turns passed to the model (token control)

_SYSTEM_PROMPT = (
    "You are an accurate, encouraging university study assistant. You answer ONLY "
    "from the course materials returned by the search_course_materials tool.\n"
    "RULES:\n"
    "1. Ground every factual claim in retrieved passages and CITE the source inline "
    "as (filename p.X) — use the Source labels from the tool results.\n"
    "2. Passages tagged (IMAGE) are descriptions of figures, diagrams, flow charts, or "
    "slides that appear in the materials — treat them as valid evidence about what those "
    "visuals show, and cite them like any other source.\n"
    "3. If the materials do not contain the answer, say so plainly: 'I couldn't find "
    "that in the provided course materials.' Never invent facts, page numbers, or citations.\n"
    "4. Do NOT give away final answers to graded assignments — explain the concepts and "
    "method so the student learns instead.\n"
    "5. Be concise and clear. Use the student's own course wording where possible."
)

# Intents for which we force a retrieval so the answer is always grounded.
_FORCE_RETRIEVAL = {"explain_topic", "assignment_question", "follow_up"}


def _history_messages(history: List[dict]) -> List[dict]:
    msgs = [m for m in history if m.get("role") in ("user", "assistant")]
    return [{"role": m["role"], "content": m["content"]} for m in msgs[-_MAX_HISTORY:]]


def _run_tool_call(tc, retriever: Retriever, sources: list) -> dict:
    """Execute one search_course_materials call; return the tool message dict."""
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    query = args.get("query", "")
    modality = args.get("modality", "any")
    hits = retriever.search(query, modality=modality) if query else []
    sources.extend(hits)
    return {
        "role": "tool",
        "tool_call_id": tc.id,
        "name": TOOL_NAME,
        "content": retriever.format_context(hits),
    }


def _dedup_sources(hits: list) -> list:
    seen, out = set(), []
    for h in hits:
        key = (h["metadata"].get("source"), h["metadata"].get("page"),
               h["metadata"].get("type"))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def answer(message: str, history: Optional[List[dict]] = None,
           retriever: Optional[Retriever] = None) -> dict:
    """Answer a user message. Returns {answer, intent, sources}."""
    history = history or []
    retriever = retriever or Retriever()
    client = get_client()

    intent = classify(message, history)
    tool_choice = (
        {"type": "function", "function": {"name": TOOL_NAME}}
        if intent in _FORCE_RETRIEVAL else "auto"
    )

    messages = (
        [{"role": "system", "content": _SYSTEM_PROMPT
          + f"\n\n(Detected intent: {intent}.)"}]
        + _history_messages(history)
        + [{"role": "user", "content": message}]
    )

    sources: list = []

    def first_call():
        return client.chat.completions.create(
            model=GEN_MODEL, messages=messages, tools=[SEARCH_TOOL],
            tool_choice=tool_choice, temperature=0.3, max_tokens=1024,
        )

    msg = with_retry(first_call).choices[0].message

    if msg.tool_calls:
        # Echo the assistant tool-call turn, then append each tool result.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            messages.append(_run_tool_call(tc, retriever, sources))

        def final_call():
            return client.chat.completions.create(
                model=GEN_MODEL, messages=messages, temperature=0.3, max_tokens=1024,
            )

        final = with_retry(final_call).choices[0].message.content
    else:
        final = msg.content

    return {"answer": final or "", "intent": intent, "sources": _dedup_sources(sources)}
