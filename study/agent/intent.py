"""Classify a user message into one of four intents (cheap, fast Groq model).

The intent decides how we retrieve and answer:
  explain_topic        -> retrieve course content, explain (no homework answers)
  assignment_question  -> retrieve assignment/deadline/logistics info
  follow_up            -> a follow-up to the previous turn; retrieve with context
  user_response        -> the user is answering a question WE asked (e.g. a quiz);
                          continue the dialog, retrieval optional
"""
from __future__ import annotations

import json
from typing import List

from ..config import INTENT_MODEL
from ..llm import get_client, with_retry

INTENTS = {"explain_topic", "assignment_question", "follow_up", "user_response"}
DEFAULT_INTENT = "explain_topic"

_SYSTEM = (
    "You classify a student's message to a study assistant into exactly one intent. "
    "Respond with JSON: {\"intent\": \"<one of: explain_topic, assignment_question, "
    "follow_up, user_response>\"}.\n"
    "- explain_topic: asks to explain/define/summarize a course concept or find info.\n"
    "- assignment_question: about assignments, deadlines, exams, grading, logistics.\n"
    "- follow_up: refers back to the previous answer (\"why?\", \"give an example\", "
    "\"and the second one?\") without naming a new topic.\n"
    "- user_response: the student is answering a question the assistant just asked "
    "(e.g. replying to a quiz/flashcard prompt)."
)


def classify(message: str, history: List[dict] | None = None) -> str:
    """Return one of INTENTS. Falls back to explain_topic on any error."""
    history = history or []
    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    user_block = message
    if last_assistant:
        user_block = (
            f"Previous assistant message:\n\"\"\"{last_assistant[:600]}\"\"\"\n\n"
            f"New student message:\n\"\"\"{message}\"\"\""
        )

    client = get_client()

    def call():
        resp = client.chat.completions.create(
            model=INTENT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_block},
            ],
            temperature=0.0,
            max_tokens=30,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    try:
        raw = with_retry(call)
        intent = json.loads(raw).get("intent", "").strip()
        return intent if intent in INTENTS else DEFAULT_INTENT
    except Exception as e:  # noqa: BLE001
        print(f"[intent] classification failed: {e}")
        return DEFAULT_INTENT
