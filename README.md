# 🎓 Calculemus Study Agent

A **Personalized Study Agent** that turns a folder of course PDFs (textbooks &
lecture slides, including their figures and flow charts) into an accurate, cited
chat assistant.

Built around a metadata-rich vector store: every answer is grounded in retrieved
passages and cites `filename p.X (chapter)`, so it stays honest about what's in
the materials and what isn't.

## 🧱 Architecture

```
PDFs ─► parse (parallel) ─► chunk + metadata ─► caption images ─► embed ─► ChromaDB
                                                                              │
        user question ─► intent classify ─► [tool] retrieve ───────────────► answer (cited)
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Parse | `study/parsing/pdf_parser.py`, `toc.py` | PyMuPDF4LLM → per-page Markdown (headings, tables); raw PyMuPDF `get_toc()` → chapter/section breadcrumb; extracts figures and renders visual/flow-chart slides. Runs in a **process pool**. |
| Chunk | `study/chunking.py` | Heading-first then size-capped (~1500 chars, 200 overlap) windows; tables kept atomic; **heading breadcrumb prepended** to each chunk's embedded text. |
| Images | `study/images.py` | Captions figures/slides with a **Groq vision model** so "what's in the flow chart on slide X" is searchable. Drops decorative images. |
| Embed | `study/embeddings.py` | Local `bge-small-en-v1.5` (Groq has no embeddings endpoint). |
| Store | `study/vectorstore.py` | Persistent **ChromaDB** with metadata filtering (`type`, `source`, …). |
| Index | `study/indexer.py` | Orchestrates the above with a file-hash cache for incremental re-index. |
| Retrieve | `study/retriever.py` | Embeds the query, searches Chroma, formats cited context. |
| Agent | `study/agent/` | `intent.py` (4-way classify) → `orchestrator.py` forces a `search_course_materials` tool call → grounded, cited generation. |
| UI | `app.py` | Streamlit: build index w/ progress, chat, sources panel with image thumbnails. |

**Intents:** `explain_topic` · `assignment_question` · `follow_up` · `user_response`.

## 🛠️ Quick Start

```bash
# 1. Fresh venv (the repo's stock venv has broken paths — recreate it)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Groq key (only allowed API). Create .env in this folder:
echo 'GROQ_API_KEY=gsk_your_key_here' > .env

# 3. Run
streamlit run app.py
```

In the sidebar: point **Course folder** at your PDFs (or upload them) → **Build /
Update Index** → ask questions. Image captioning makes the first build slower.

## ⚙️ Config (env vars, all optional)

`GROQ_GEN_MODEL`, `GROQ_INTENT_MODEL`, `GROQ_VISION_MODEL`, `EMBED_MODEL`,
`TOP_K`, `CHUNK_CHARS`, `CAPTION_MAX_WORKERS`, `INDEX_DIR`,
`STUDY_PARSE_MODE` (`process`|`thread`|`sequential`).

> ⚠️ **Free-tier rate limits:** vision captioning is token-heavy (~30k TPM cap).
> Captioning concurrency is kept low and backs off on 429s; it's a one-time
> index cost. Lower `CAPTION_MAX_WORKERS` if you still hit limits.

## ⚠️ Rules

1. **Empower, don't cheat** — the agent explains concepts; it won't hand over
   graded-assignment answers (enforced in the system prompt).
2. **Safety first** — no Leiden credentials anywhere; the Groq key lives only in
   `.env` (gitignored).

---
_The original baseline files (`agent.py`, `document_parser.py`) are left in place
for reference; the app now runs through the `study/` package._
