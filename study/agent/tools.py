"""The retrieval tool exposed to Groq function-calling."""
from __future__ import annotations

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_course_materials",
        "description": (
            "Search the indexed course materials (lecture slides, textbooks, notes) "
            "for passages relevant to a query. Returns numbered passages with their "
            "source filename, page number, and chapter/section. Call this whenever "
            "you need facts from the materials to answer accurately."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query describing the information needed.",
                },
                "modality": {
                    "type": "string",
                    "enum": ["any", "text", "image"],
                    "description": (
                        "Use 'image' when the question is about a figure, diagram, "
                        "flow chart, chart, or slide visual; 'text' for prose; "
                        "'any' otherwise."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_NAME = "search_course_materials"
