"""
Study Agent — LLM bridge for the Ragnarok Study Guide.

Every request to the LLM carries:
    • request_type : "fresh" | "follow_up"
    • conversation_summary : condensed history (only when follow_up)
    • context : reference text from the parsed corpus
    • template : { category, user_query, output_syntax }

Every response is normalised to:
    [ { "category": "<type>", "content": { ... } }, ... ]
matching the renderers in app.py exactly.

Anti-hallucination policy:
    • temperature = 0.1  (near-deterministic for factual recall)
    • LLM is EXPLICITLY instructed to ONLY use information from the provided context.
    • If the context does not contain enough information, the LLM must say so.
    • Category detection uses a few-shot prompt with strict single-word output.
    • JSON repair preserves whitespace inside string values (Mermaid-safe).
"""

import os
import re
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

# Maximum characters of context to send to the LLM.
# Prevents context from swamping the user query in the attention window.
_MAX_CONTEXT_CHARS = 6000


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
      "title": "<short heading — taken from or directly inspired by the provided context>",
      "body": "<markdown explanation — use ### headings, **bold**, bullet lists, LaTeX with $…$ or $$…$$ as needed. ONLY use facts from the provided COURSE MATERIAL CONTEXT. If a fact is not in the context, say 'This is not covered in the provided material.'>"
    },
    "follow_ups": ["<2-4 category names from: explanation, flashcard, quiz, location, flowchart — pick the ones most useful as next steps>"]
  }
]""",

    Category.FLASHCARD: """\
Return a JSON array with exactly 1 object:
[
  {
    "category": "flashcard",
    "content": {
      "cards": [
        { "front": "<question or term from the provided context>", "back": "<answer or definition — must be grounded in the context, may use **bold** and markdown>" },
        ... (produce 4–8 cards, ALL derived from the COURSE MATERIAL CONTEXT)
      ]
    },
    "follow_ups": ["<2-4 category names from: explanation, flashcard, quiz, location, flowchart>"]
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
          "question": "<question — must be answerable from the COURSE MATERIAL CONTEXT>",
          "options": ["A) …", "B) …", "C) …", "D) …"],
          "answer": "<A, B, C, or D>",
          "explanation": "<why that answer is correct — cite the relevant part of the context>"
        },
        ... (produce 3–5 questions, ALL based on the COURSE MATERIAL CONTEXT)
      ]
    },
    "follow_ups": ["<2-4 category names from: explanation, flashcard, quiz, location, flowchart>"]
  }
]""",

    Category.LOCATION: """\
Return a JSON array with exactly 1 object:
[
  {
    "category": "location",
    "content": {
      "file": "<source filename — MUST be one of the filenames listed in the COURSE MATERIAL CONTEXT headers>",
      "section": "<section or chapter name/number from the context>",
      "page": "<page reference from the context header, e.g. 'Page 3', or 'N/A'>",
      "excerpt": "<SHORT verbatim quote copied EXACTLY from the provided context (≤ 30 words)>",
      "relevance": "<one sentence on why this location answers the query>"
    },
    "follow_ups": ["<2-4 category names from: explanation, flashcard, quiz, location, flowchart>"]
  }
]""",

    Category.FLOWCHART: """\
Return a JSON array with exactly 1 object:
[
  {
    "category": "flowchart",
    "content": {
      "title": "<descriptive title based on the context>",
      "mermaid": "<valid Mermaid graph TD syntax — use \\n for newlines, e.g. graph TD\\n    A[Step] --> B[Step]. Steps must reflect the process described in the COURSE MATERIAL CONTEXT.>",
      "description": "<1–2 sentence explanation grounded in the context>"
    },
    "follow_ups": ["<2-4 category names from: explanation, flashcard, quiz, location, flowchart>"]
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

    Key anti-hallucination measures:
      • Persona explicitly limits the LLM to only use the provided context.
      • Context is clearly delimited with BEGIN/END markers.
      • Output instructions demand JSON-only, no prose.
      • Context is hard-truncated to _MAX_CONTEXT_CHARS.
    """
    parts: list[str] = []

    # ─ Persona with hard anti-hallucination constraint ─────────────
    parts.append(
        "You are Calculemus, an AI Study Assistant. "
        "Your ONLY knowledge source for answering questions is the COURSE MATERIAL CONTEXT provided below. "
        "You MUST NOT use any outside knowledge, training data, or assumptions beyond what is explicitly stated in that context. "
        "If the context does not contain enough information to answer the query fully, you MUST say so clearly — do NOT fabricate facts. "
        "NEVER give direct answers to graded assignments — explain concepts instead."
    )

    # ─ Request type ───────────────────────────────────────────────
    parts.append(f"\nREQUEST TYPE: {request_type}")
    if request_type == "follow_up" and conversation_summary:
        parts.append(
            f"\nCONVERSATION HISTORY (for context continuity):\n{conversation_summary}\n"
            "Your new response should be consistent with the above history. "
            "Do NOT contradict what was said before unless the user explicitly asks you to revisit it."
        )

    # ─ Reference context from parser ──────────────────────────────
    if context:
        # Hard-truncate to prevent context from flooding the attention window
        truncated = context[:_MAX_CONTEXT_CHARS]
        if len(context) > _MAX_CONTEXT_CHARS:
            truncated += "\n[... context truncated for length ...]"
        parts.append(
            f"\n=== BEGIN COURSE MATERIAL CONTEXT ===\n{truncated}\n=== END COURSE MATERIAL CONTEXT ===\n"
            "Answer EXCLUSIVELY from the above context. "
            "If the answer is not there, respond with the appropriate JSON structure but include a note in the content body saying the topic is not covered in the uploaded material."
        )
    else:
        parts.append(
            "\nNO COURSE MATERIAL HAS BEEN PROVIDED. "
            "You MUST inform the student that no documents have been uploaded/trained yet, "
            "and ask them to upload course materials first. "
            "Do not attempt to answer from general knowledge."
        )

    # ─ Output template ────────────────────────────────────────────
    parts.append(
        "\nOUTPUT INSTRUCTIONS:\n"
        "You MUST respond with ONLY a valid JSON array — no extra prose, "
        "no markdown fences, no code blocks, no trailing commas, no comments.\n\n"
        f"EXPECTED OUTPUT SYNTAX:\n{_OUTPUT_SYNTAX[category]}"
    )

    return "\n".join(parts)


# ── Category auto-detection ───────────────────────────────────────────────────

# Few-shot examples to anchor the classifier — prevents off-by-one category errors
_CLASSIFY_FEW_SHOTS = """\
Examples:
  Query: "Explain how photosynthesis works"           → explanation
  Query: "What is photosynthesis?"                    → explanation
  Query: "Make flashcards for Chapter 3"              → flashcard
  Query: "Give me some Q&A cards on Newton's laws"    → flashcard
  Query: "Quiz me on thermodynamics"                  → quiz
  Query: "Test my knowledge of the water cycle"       → quiz
  Query: "Where in the notes does it mention osmosis?"→ location
  Query: "Which page covers integration by parts?"    → location
  Query: "Draw a diagram of the Krebs cycle"          → flowchart
  Query: "Show me a flowchart of the process"         → flowchart
"""

_VALID_CATEGORIES = {c.value for c in Category}


def detect_category(user_query: str) -> Category:
    """
    Ask the LLM to classify the user query into one of the five categories.
    Uses a few-shot prompt with temperature=0 for determinism.
    Falls back to EXPLANATION on any failure.
    """
    if client is None:
        return Category.EXPLANATION

    classify_prompt = (
        "Classify the following student query into exactly one of these five categories:\n"
        "  explanation  – wants a concept explained in text\n"
        "  flashcard    – wants flashcard-style Q&A pairs\n"
        "  quiz         – wants practice quiz / multiple-choice questions\n"
        "  location     – wants to find where something is in the course material\n"
        "  flowchart    – wants a diagram or step-by-step process map\n\n"
        f"{_CLASSIFY_FEW_SHOTS}\n"
        f'Query: "{user_query}"\n\n'
        "Respond with ONLY one word — the exact category name (no punctuation, no quotes)."
    )
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": classify_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip().lower()
        # Strip any trailing punctuation the model might add
        raw = raw.strip(".,!?;:'\"")
        if raw in _VALID_CATEGORIES:
            return Category(raw)
        # Try to find a valid category as a substring (e.g. "explanation." → "explanation")
        for cat in _VALID_CATEGORIES:
            if cat in raw:
                return Category(cat)
        return Category.EXPLANATION
    except Exception:
        return Category.EXPLANATION


# ── Conversation summariser ───────────────────────────────────────────────────

def summarise_history(chat_history: list[dict]) -> str:
    """
    Produce a compact but semantically rich summary of the conversation so far.
    Keeps the last 8 exchanges (4 user + 4 assistant) to stay within token limits.

    For assistant messages, we extract the actual title/content instead of just
    listing the category — this preserves topical continuity.

    Parameters
    ----------
    chat_history : list[dict]
        The full chat_history list from st.session_state.
        Each item has "role" ("user" | "assistant") and either
        "content" (user msgs) or "blocks" (assistant msgs).
    """
    if not chat_history:
        return ""

    recent = chat_history[-8:]  # last 4 user+assistant pairs max
    lines: list[str] = []
    for msg in recent:
        if msg["role"] == "user":
            lines.append(f"Student asked: {msg.get('content', '').strip()}")
        else:
            # Extract meaningful content from each assistant block
            blocks = msg.get("blocks", [])
            for block in blocks:
                cat = block.get("category", "unknown")
                content = block.get("content", {})
                # Pull a short summary from the block content
                if cat == "explanation":
                    title = content.get("title", "")
                    body_preview = content.get("body", "")[:120].replace("\n", " ")
                    lines.append(f"Assistant explained [{title}]: {body_preview}...")
                elif cat == "flashcard":
                    n = len(content.get("cards", []))
                    topics = [c.get("front", "") for c in content.get("cards", [])[:3]]
                    lines.append(f"Assistant provided {n} flashcards on: {', '.join(topics)}")
                elif cat == "quiz":
                    title = content.get("title", "")
                    n = len(content.get("questions", []))
                    lines.append(f"Assistant gave a {n}-question quiz titled [{title}]")
                elif cat == "location":
                    file_ = content.get("file", "")
                    section = content.get("section", "")
                    lines.append(f"Assistant located topic in [{file_}], section [{section}]")
                elif cat == "flowchart":
                    title = content.get("title", "")
                    lines.append(f"Assistant drew a flowchart: [{title}]")
                else:
                    lines.append(f"Assistant responded with {cat}")

    return "\n".join(lines)


# ── Robust JSON extraction ────────────────────────────────────────────────────

def _extract_json_array(raw_text: str) -> list[dict]:
    """
    Robustly extract a JSON array from a potentially messy LLM response.

    Strategy (in order):
    1. Direct parse (happy path).
    2. Strip markdown fences (```json ... ```) — handles single and triple fences.
    3. Find the outermost JSON array using bracket counting.
    4. Mermaid-safe repair: only strip true control chars outside of string literals.

    Raises json.JSONDecodeError if all strategies fail.
    """
    text = raw_text.strip()

    # ── Strategy 1: Direct parse ──────────────────────────────────
    try:
        result = json.loads(text, strict=False)
        return result if isinstance(result, list) else [result]
    except json.JSONDecodeError:
        pass

    # ── Strategy 2: Strip markdown code fences ────────────────────
    # Handles ```json ... ``` and ``` ... ```
    fence_pattern = re.compile(
        r"```(?:json)?\s*([\s\S]*?)```",
        re.IGNORECASE
    )
    fence_match = fence_pattern.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            result = json.loads(candidate, strict=False)
            return result if isinstance(result, list) else [result]
        except json.JSONDecodeError:
            text = candidate  # use the de-fenced text for further repair

    # ── Strategy 3: Find outermost [...] array ────────────────────
    start = text.find("[")
    if start != -1:
        depth = 0
        end = -1
        for i, ch in enumerate(text[start:], start=start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            candidate = text[start:end]
            try:
                result = json.loads(candidate, strict=False)
                return result if isinstance(result, list) else [result]
            except json.JSONDecodeError:
                text = candidate

    # ── Strategy 4: Mermaid-safe control-char repair ──────────────
    # Only strip bare control chars (0x00-0x08, 0x0b, 0x0c, 0x0e-0x1f)
    # PRESERVE \n (0x0a), \r (0x0d), \t (0x09) as json.loads handles them.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Remove trailing commas before ] or }
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    result = json.loads(cleaned, strict=False)
    return result if isinstance(result, list) else [result]


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
        [ { "category": "...", "content": { ... }, "follow_ups": [...] } ]

    Parameters
    ----------
    user_query : str
        The student's question or request.
    context : str
        Reference material from the parsed corpus (pre-filtered by retriever).
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
            "follow_ups": [],
        }]

    # ── Validate and sanitise the query ───────────────────────────
    user_query = user_query.strip()
    if not user_query:
        return [{
            "category": "explanation",
            "content": {
                "title": "Empty Query",
                "body": "Please type a question or request.",
            },
            "follow_ups": [],
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
    raw_text = ""
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_query},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,   # Near-deterministic: reduces hallucination & JSON drift
            max_tokens=2048,
        )
        raw_text = response.choices[0].message.content.strip()

        # ── Parse JSON with multi-strategy extractor ───────────────
        parsed = _extract_json_array(raw_text)

        # ── Normalise to { category, content, follow_ups } ──────────
        valid_cats = {c.value for c in Category}
        normalised: list[dict] = []

        for obj in parsed:
            if not isinstance(obj, dict):
                continue

            # Safely pop follow_ups — must be a list of valid category strings
            raw_follow_ups = obj.pop("follow_ups", [])
            if not isinstance(raw_follow_ups, list):
                raw_follow_ups = []
            follow_ups = [
                f for f in raw_follow_ups
                if isinstance(f, str) and f.strip(".,") in valid_cats
            ]

            # Case A: well-formed { "category": ..., "content": ... }
            if "category" in obj and "content" in obj:
                obj_cat = obj["category"]
                # Ensure the category value is valid
                if obj_cat not in valid_cats:
                    obj_cat = category.value
                obj["category"] = obj_cat
                obj["follow_ups"] = follow_ups
                normalised.append(obj)
            # Case B: LLM returned { "type": ..., ... } without "content" wrapper
            elif "type" in obj or len(obj) > 0:
                obj_cat = obj.pop("type", None) or obj.pop("category", None) or category.value
                if obj_cat not in valid_cats:
                    obj_cat = category.value
                normalised.append({
                    "category": obj_cat,
                    "content": obj,
                    "follow_ups": follow_ups,
                })

        if not normalised:
            raise ValueError("LLM returned an empty or unrecognised JSON structure.")

        return normalised

    except json.JSONDecodeError as e:
        preview = raw_text[:600] if raw_text else "(no response)"
        return [{
            "category": "explanation",
            "content": {
                "title": "Response Parse Error",
                "body": (
                    "The AI returned a response that could not be parsed as JSON.\n\n"
                    f"**Error:** `{e}`\n\n"
                    f"**Raw response (first 600 chars):**\n```\n{preview}\n```\n\n"
                    "_Try rephrasing your query or re-training the corpus._"
                ),
            },
            "follow_ups": [],
        }]
    except Exception as e:
        return [{
            "category": "explanation",
            "content": {
                "title": "Agent Error",
                "body": (
                    f"Something went wrong communicating with the AI:\n\n`{e}`\n\n"
                    "_Please check your API key and internet connection._"
                ),
            },
            "follow_ups": [],
        }]


# ── Quick CLI demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_context = """
--- Source: notes.pdf (Page 1) ---
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
