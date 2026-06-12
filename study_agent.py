"""
Study Agent — LLM bridge for the Calculemus Study Guide.

Every request to the LLM carries:
    • request_type : "fresh" | "follow_up"
    • conversation_summary : condensed history (only when follow_up)
    • context : reference text from the parsed corpus
    • template : { category, user_query, output_syntax }

Every response is normalised to:
    [ { "category": "<type>", "content": { ... } }, ... ]
matching the renderers in app.py exactly.
"""

import os
import json
from enum import Enum

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.getenv("GROQ_API_KEY")
client = None
if api_key and Groq is not None:
    client = Groq(api_key=api_key)


# ── Category enum ─────────────────────────────────────────────────────────────
class Category(str, Enum):
    EXPLANATION = "explanation"
    FLASHCARD   = "flashcard"
    QUIZ        = "quiz"
    LOCATION    = "location"
    FLOWCHART   = "flowchart"


# ── Output-syntax templates (must match what the UI renderers consume) ────────
# Each tells the LLM *exactly* what JSON shape to produce inside "content".

_OUTPUT_SYNTAX = {
    Category.EXPLANATION: """\
Return a JSON array with exactly 1 object:
[
  {
    "category": "explanation",
    "content": {
      "title": "<short heading>",
      "body": "<markdown explanation — use ### headings, **bold**, bullet lists, LaTeX with $…$ or $$…$$ as needed. Do NOT give direct answers to graded assignments.>"
    }
  }
]""",

    Category.FLASHCARD: """\
Return a JSON array with exactly 1 object:
[
  {
    "category": "flashcard",
    "content": {
      "cards": [
        { "front": "<question or term>", "back": "<answer or definition, may use **bold** and markdown>" },
        ... (produce 4–8 cards)
      ]
    }
  }
]""",

    Category.QUIZ: """\
Return a JSON array with exactly 1 object:
[
  {
    "category": "quiz",
    "content": {
      "title": "<quiz title>",
      "questions": [
        {
          "question": "<question text>",
          "options": ["A) …", "B) …", "C) …", "D) …"],
          "answer": "<A, B, C, or D>",
          "explanation": "<why that answer is correct>"
        },
        ... (produce 3–5 questions)
      ]
    }
  }
]""",

    Category.LOCATION: """\
Return a JSON array with exactly 1 object:
[
  {
    "category": "location",
    "content": {
      "file": "<source filename from the provided context>",
      "section": "<section or chapter name/number>",
      "page": "<page reference if available, else 'N/A'>",
      "excerpt": "<short verbatim quote from the material (≤ 30 words)>",
      "relevance": "<one sentence on why this location answers the query>"
    }
  }
]""",

    Category.FLOWCHART: """\
Return a JSON array with exactly 1 object:
[
  {
    "category": "flowchart",
    "content": {
      "title": "<descriptive title>",
      "mermaid": "<valid Mermaid graph TD syntax — use \\n for newlines, e.g. graph TD\\n    A[Step] --> B[Step]>",
      "description": "<1–2 sentence explanation of the flowchart>"
    }
  }
]""",
}


# ── Build the full system prompt ──────────────────────────────────────────────

def _build_system_prompt(
    *,
    category: Category,
    context: str,
    request_type: str,
    conversation_summary: str,
) -> str:
    """
    Assemble the system prompt sent to the LLM.

    Includes:
      • base persona instructions
      • request type (fresh / follow_up)
      • conversation summary (if follow_up)
      • reference context from parser
      • output syntax template
    """
    parts: list[str] = []

    # ─ Persona ────────────────────────────────────────────────────
    parts.append(
        "You are a highly intelligent AI Study Assistant called Calculemus. "
        "Your goal is to help students learn effectively based on the provided "
        "course material. NEVER give direct answers to graded assignments — "
        "explain concepts instead."
    )

    # ─ Request type ───────────────────────────────────────────────
    parts.append(f"\nREQUEST TYPE: {request_type}")
    if request_type == "follow_up" and conversation_summary:
        parts.append(
            f"\nCONVERSATION HISTORY SUMMARY:\n{conversation_summary}\n"
            "Use the above summary as context for the student's follow-up. "
            "Maintain continuity with previous answers."
        )

    # ─ Reference context from parser ──────────────────────────────
    if context:
        parts.append(f"\nCOURSE MATERIAL CONTEXT:\n{context}")

    # ─ Output template ────────────────────────────────────────────
    parts.append(
        "\nOUTPUT INSTRUCTIONS:\n"
        "You MUST respond with ONLY a valid JSON array — no extra prose, "
        "no markdown fences, no trailing commas.\n\n"
        f"EXPECTED OUTPUT SYNTAX:\n{_OUTPUT_SYNTAX[category]}"
    )

    return "\n".join(parts)


# ── Category auto-detection ───────────────────────────────────────────────────

def detect_category(user_query: str) -> Category:
    """
    Ask the LLM to classify the user query into one of the five categories.
    """
    if client is None:
        return Category.EXPLANATION
    classify_prompt = (
        "Classify the following student query into exactly one category.\n"
        "Categories:\n"
        "  explanation  – wants a concept explained in text\n"
        "  flashcard    – wants flashcard-style Q&A pairs\n"
        "  quiz         – wants practice quiz / multiple-choice questions\n"
        "  location     – wants to find where something is in the course material\n"
        "  flowchart    – wants a diagram or step-by-step process map\n\n"
        f'Query: "{user_query}"\n\n'
        "Respond with ONLY one word, the category name."
    )
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": classify_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip().lower()
        return Category(raw)
    except (ValueError, Exception):
        return Category.EXPLANATION


# ── Conversation summariser ───────────────────────────────────────────────────

def summarise_history(chat_history: list[dict]) -> str:
    """
    Produce a compact summary of the conversation so far.
    Keeps only the last 6 exchanges to stay within token limits.

    Parameters
    ----------
    chat_history : list[dict]
        The full chat_history list from st.session_state.
        Each item has "role" ("user" | "assistant") and either
        "content" (user msgs) or "blocks" (assistant msgs).
    """
    if not chat_history:
        return ""

    recent = chat_history[-12:]  # last 6 user+assistant pairs max
    lines: list[str] = []
    for msg in recent:
        if msg["role"] == "user":
            lines.append(f"Student: {msg.get('content', '')}")
        else:
            # Summarise assistant blocks by category
            blocks = msg.get("blocks", [])
            cats = [b.get("category", "unknown") for b in blocks]
            lines.append(f"Assistant: responded with {', '.join(cats)}")

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def ask_study_agent(
    user_query: str,
    *,
    context: str = "",
    category: Category | None = None,
    request_type: str = "fresh",
    chat_history: list[dict] | None = None,
) -> list[dict]:
    """
    Query the study agent. Always returns a list of dicts in the format:
        [ { "category": "...", "content": { ... } } ]

    Parameters
    ----------
    user_query : str
        The student's question or request.
    context : str
        Reference material from the parsed corpus.
    category : Category | None
        Force a response type. If None the agent auto-detects.
    request_type : "fresh" | "follow_up"
        Whether this is a new question or a follow-up.
    chat_history : list[dict] | None
        The session chat_history — used to build a summary for follow-ups.

    Returns
    -------
    list[dict]
        Normalised response array for the UI renderers.
    """
    # ── Guard: no API key available ────────────────────────────────
    if client is None:
        return [{
            "category": "explanation",
            "content": {
                "title": "API Key Missing",
                "body": (
                    "No `GROQ_API_KEY` found in your `.env` file.\n\n"
                    "Please add your key and restart, or switch to **Demo Mode** "
                    "in the sidebar to test with dummy data."
                ),
            },
        }]

    # ── Resolve category ──────────────────────────────────────────
    if category is None:
        category = detect_category(user_query)

    # ── Build conversation summary for follow-ups ─────────────────
    conversation_summary = ""
    if request_type == "follow_up" and chat_history:
        conversation_summary = summarise_history(chat_history)

    # ── Assemble system prompt ────────────────────────────────────
    system_prompt = _build_system_prompt(
        category=category,
        context=context,
        request_type=request_type,
        conversation_summary=conversation_summary,
    )

    # ── Call the LLM ──────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_query},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=2048,
        )
        raw_text = response.choices[0].message.content.strip()

        # Strip markdown fences the model may add despite instructions
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        # strict=False allows literal control chars (newlines, tabs) inside strings
        try:
            parsed = json.loads(raw_text, strict=False)
        except json.JSONDecodeError:
            # Last-resort repair: strip control chars that break JSON
            import re
            cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', raw_text)
            parsed = json.loads(cleaned)

        if not isinstance(parsed, list):
            parsed = [parsed]

        # ── Normalise to { category, content } ────────────────────
        normalised: list[dict] = []
        for obj in parsed:
            # Already in correct format
            if "category" in obj and "content" in obj:
                normalised.append(obj)
            else:
                # Flatten: pull "type" or stamp category, wrap rest in content
                obj_cat = obj.pop("type", category.value)
                obj.pop("category", None)
                normalised.append({
                    "category": obj_cat,
                    "content": obj,
                })

        return normalised

    except json.JSONDecodeError as e:
        return [{
            "category": "explanation",
            "content": {
                "title": "Response Error",
                "body": (
                    f"The AI returned a response that could not be parsed.\n\n"
                    f"**Error:** `{e}`\n\n"
                    f"**Raw response:**\n```\n{raw_text[:500]}\n```"
                ),
            },
        }]
    except Exception as e:
        return [{
            "category": "explanation",
            "content": {
                "title": "Agent Error",
                "body": f"Something went wrong:\n\n`{e}`",
            },
        }]


# ── Quick CLI demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_context = """
    Chapter 3 – The Water Cycle
    The water cycle (hydrological cycle) describes how water moves through Earth's systems.
    Key stages: evaporation, condensation, precipitation, collection.
    Evaporation occurs when solar energy heats surface water, converting it to vapour.
    Condensation forms clouds when water vapour cools at altitude.
    Precipitation returns water to the surface as rain, snow, or hail.
    """

    queries = [
        ("Explain how evaporation works",          None),
        ("Make me flashcards for the water cycle", None),
        ("Give me a quiz on precipitation",        None),
        ("Where does the book talk about clouds?", None),
        ("Draw the steps of the water cycle",      None),
    ]

    for query, forced_cat in queries:
        print(f"\n{'='*60}")
        print(f"QUERY : {query}")
        result = ask_study_agent(
            query,
            context=sample_context,
            category=forced_cat,
            request_type="fresh",
        )
        print(f"CAT   : {result[0].get('category')}")
        print(f"RESULT:\n{json.dumps(result, indent=2)}")
