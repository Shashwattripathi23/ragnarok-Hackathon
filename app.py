import streamlit as st
import time
import json
from dummy_data import (
    DUMMY_FILES,
    get_dummy_response,
    get_initial_dummy_response,
)

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calculemus Study Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── CSS Variables ──────────────────────────────────────────── */
:root {
    --bg-primary: #0f0f1a;
    --bg-card: #1a1a2e;
    --bg-card-hover: #22223a;
    --accent-purple: #7c3aed;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-indigo: #6366f1;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --border-subtle: rgba(148, 163, 184, 0.12);
    --glass-bg: rgba(26, 26, 46, 0.75);
    --glass-border: rgba(124, 58, 237, 0.2);
    --radius: 14px;
    --shadow-glow: 0 0 25px rgba(124, 58, 237, 0.15);
}

/* ── Global ─────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(145deg, #0f0f1a 0%, #1a1033 50%, #0f0f1a 100%) !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13132b 0%, #1a1033 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}

/* ── Header ─────────────────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 1.2rem 0 0.6rem;
    margin-bottom: 0.5rem;
}
.hero-header h1 {
    background: linear-gradient(135deg, #7c3aed, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero-header p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 4px;
}

/* ── Sidebar section headers ────────────────────────────────── */
.sidebar-section {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 1rem 1rem 0.8rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
}
.sidebar-section h3 {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--accent-purple);
    margin: 0 0 0.7rem;
}

/* ── Train button ───────────────────────────────────────────── */
.train-btn button {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    transition: transform 0.15s, box-shadow 0.15s !important;
    width: 100%;
}
.train-btn button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-glow) !important;
}

/* ── Response cards (shared) ────────────────────────────────── */
.response-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin: 0.6rem 0;
    backdrop-filter: blur(12px);
    box-shadow: var(--shadow-glow);
    animation: fadeSlideUp 0.35s ease-out;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.response-card .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.8rem;
    font-weight: 700;
    font-size: 1.05rem;
}
.response-card .card-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Explanation ────────────────────────────────────────────── */
.badge-explanation { background: rgba(99,102,241,0.18); color: #818cf8; }
.explanation-card { border-left: 3px solid var(--accent-indigo); }

/* ── Location ───────────────────────────────────────────────── */
.badge-location { background: rgba(16,185,129,0.18); color: #34d399; }
.location-card  { border-left: 3px solid var(--accent-green); }
.location-meta  {
    display: flex; flex-wrap: wrap; gap: 0.6rem;
    margin-bottom: 0.8rem;
}
.location-tag {
    background: rgba(16,185,129,0.12);
    color: #34d399;
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
}
.location-excerpt {
    background: rgba(16,185,129,0.06);
    border-left: 3px solid #34d399;
    padding: 0.8rem 1rem;
    border-radius: 0 8px 8px 0;
    color: var(--text-primary);
    font-style: italic;
    line-height: 1.6;
}

/* ── Flashcards ─────────────────────────────────────────────── */
.badge-flashcard { background: rgba(245,158,11,0.18); color: #fbbf24; }
.flashcard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 0.8rem;
    margin-top: 0.4rem;
}
.flashcard {
    background: linear-gradient(145deg, rgba(245,158,11,0.08), rgba(245,158,11,0.03));
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    transition: transform 0.18s, box-shadow 0.18s;
    min-height: 100px;
}
.flashcard:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(245,158,11,0.12);
}
.flashcard-front {
    font-weight: 700;
    font-size: 0.92rem;
    color: #fbbf24;
    margin-bottom: 0.5rem;
}
.flashcard-num {
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

/* ── Quiz ───────────────────────────────────────────────────── */
.badge-quiz { background: rgba(244,63,94,0.18); color: #fb7185; }
.quiz-card  { border-left: 3px solid var(--accent-rose); }
.quiz-question-block {
    background: rgba(244,63,94,0.05);
    border: 1px solid rgba(244,63,94,0.15);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.7rem;
}
.quiz-q-text {
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    font-size: 0.95rem;
}
.quiz-result-correct {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-top: 0.4rem;
    color: #34d399;
    font-size: 0.85rem;
}
.quiz-result-wrong {
    background: rgba(244,63,94,0.12);
    border: 1px solid rgba(244,63,94,0.3);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-top: 0.4rem;
    color: #fb7185;
    font-size: 0.85rem;
}

/* ── Flowchart ──────────────────────────────────────────────── */
.badge-flowchart { background: rgba(6,182,212,0.18); color: #22d3ee; }
.flowchart-card   { border-left: 3px solid var(--accent-cyan); }
.flowchart-desc {
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.55;
    margin-top: 0.6rem;
}

/* ── Follow-up action bar ───────────────────────────────────── */
.action-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--border-subtle);
}
.action-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.12s, box-shadow 0.12s;
    border: 1px solid;
    text-decoration: none;
}
.action-pill:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.pill-flashcard  { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.3); color: #fbbf24; }
.pill-quiz       { background: rgba(244,63,94,0.1);  border-color: rgba(244,63,94,0.3);  color: #fb7185; }
.pill-location   { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.3); color: #34d399; }
.pill-explain    { background: rgba(99,102,241,0.1); border-color: rgba(99,102,241,0.3); color: #818cf8; }
.pill-flowchart  { background: rgba(6,182,212,0.1);  border-color: rgba(6,182,212,0.3);  color: #22d3ee; }

/* ── Corpus stats chip ──────────────────────────────────────── */
.corpus-stat {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #a78bfa;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}

/* ── File list chip ─────────────────────────────────────────── */
.file-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 8px;
    padding: 3px 9px;
    font-size: 0.75rem;
    color: #60a5fa;
    font-weight: 500;
    margin: 2px;
}

/* ── Misc ───────────────────────────────────────────────────── */
div[data-testid="stChatMessage"] {
    background: transparent !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────
defaults = {
    "documents": {},
    "trained": False,
    "selected_files": [],
    "chat_history": [],       # list[dict] with role, content, response_type
    "pending_action": None,   # ("category", msg_index)
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Hero header ────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-header">'
    "<h1>🎓 Calculemus</h1>"
    "<p>Your AI-powered study companion — upload, train, and master any subject</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Upload section ─────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section"><h3>📤 Upload Corpus</h3></div>',
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Drop your study materials here",
        type=["pdf", "md", "txt", "csv", "docx", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Quick-load dummy files button
    if st.button("📦 Load Demo Corpus", use_container_width=True):
        st.session_state.documents = dict(DUMMY_FILES)
        st.session_state.trained = False
        st.toast("Demo corpus loaded!", icon="📦")

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) staged")

    st.markdown("---")

    # ── Train button ───────────────────────────────────────────
    st.markdown(
        '<div class="sidebar-section"><h3>🚀 Train Parser</h3></div>',
        unsafe_allow_html=True,
    )

    corpus_ready = bool(st.session_state.documents) or bool(uploaded_files)
    st.markdown('<div class="train-btn">', unsafe_allow_html=True)
    train_clicked = st.button(
        "⚡ Train on Corpus" if not st.session_state.trained else "✅ Re-Train",
        use_container_width=True,
        disabled=not corpus_ready,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if train_clicked:
        # If real files were uploaded, store their names (parser logic placeholder)
        if uploaded_files and not st.session_state.documents:
            for f in uploaded_files:
                st.session_state.documents[f.name] = f"[Content of {f.name} — parser placeholder]"

        progress = st.progress(0, text="Parsing corpus…")
        for i in range(100):
            time.sleep(0.012)
            progress.progress(i + 1, text=f"Processing… {i+1}%")
        progress.empty()
        st.session_state.trained = True
        st.toast("Training complete!", icon="🎉")
        st.rerun()

    if st.session_state.trained:
        n_files = len(st.session_state.documents)
        n_chars = sum(len(v) for v in st.session_state.documents.values())
        chips = (
            f'<span class="corpus-stat">📄 {n_files} files</span>'
            f'<span class="corpus-stat">📝 {n_chars:,} chars</span>'
        )
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown("---")

    # ── File context selector ──────────────────────────────────
    st.markdown(
        '<div class="sidebar-section"><h3>🎯 Select Context</h3></div>',
        unsafe_allow_html=True,
    )
    if st.session_state.documents:
        file_names = list(st.session_state.documents.keys())
        selected = st.multiselect(
            "Choose files to include in chat context",
            options=file_names,
            default=file_names[:2] if len(file_names) >= 2 else file_names,
            label_visibility="collapsed",
        )
        st.session_state.selected_files = selected

        if selected:
            ctx_size = sum(len(st.session_state.documents[f]) for f in selected)
            st.markdown(
                f'<span class="corpus-stat">🎯 Context: {ctx_size:,} chars</span>',
                unsafe_allow_html=True,
            )
            with st.expander("Selected files", expanded=False):
                for fn in selected:
                    st.markdown(
                        f'<span class="file-chip">📄 {fn}</span>',
                        unsafe_allow_html=True,
                    )
    else:
        st.caption("Upload & train first to select context files.")


# ══════════════════════════════════════════════════════════════════
#  RESPONSE RENDERERS
# ══════════════════════════════════════════════════════════════════

def render_explanation(content):
    """Render an explanation block."""
    title = content.get("title", "Explanation")
    body = content.get("body", "")
    st.markdown(
        f'<div class="response-card explanation-card">'
        f'<div class="card-header">'
        f'<span class="card-badge badge-explanation">📝 Explanation</span>'
        f"<span>{title}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(body)


def render_location(content):
    """Render a corpus-location block."""
    file_name = content.get("file", "Unknown")
    section = content.get("section", "")
    page = content.get("page", "")
    excerpt = content.get("excerpt", "")
    relevance = content.get("relevance", "")
    st.markdown(
        f'<div class="response-card location-card">'
        f'<div class="card-header">'
        f'<span class="card-badge badge-location">📍 Location</span>'
        f"<span>Found in Corpus</span>"
        f"</div>"
        f'<div class="location-meta">'
        f'<span class="location-tag">📄 {file_name}</span>'
        f'<span class="location-tag">📑 {section}</span>'
        f'<span class="location-tag">📖 {page}</span>'
        f"</div>"
        f'<div class="location-excerpt">"{excerpt}"</div>'
        f'<p style="color:var(--text-secondary);font-size:0.85rem;margin-top:0.6rem;">'
        f"💡 {relevance}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_flashcards(content):
    """Render flashcard grid with expanders for answers."""
    cards = content.get("cards", [])
    st.markdown(
        '<div class="response-card" style="border-left:3px solid var(--accent-amber);">'
        '<div class="card-header">'
        '<span class="card-badge badge-flashcard">🃏 Flashcards</span>'
        f"<span>{len(cards)} Cards</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(min(len(cards), 3))
    for idx, card in enumerate(cards):
        with cols[idx % len(cols)]:
            st.markdown(
                f'<div class="flashcard">'
                f'<div class="flashcard-num">CARD {idx + 1}</div>'
                f'<div class="flashcard-front">{card["front"]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("🔄 Flip to see answer"):
                st.markdown(card["back"])


def render_quiz(content, msg_idx):
    """Render an interactive quiz with radio buttons."""
    title = content.get("title", "Quiz")
    questions = content.get("questions", [])
    st.markdown(
        f'<div class="response-card quiz-card">'
        f'<div class="card-header">'
        f'<span class="card-badge badge-quiz">🧪 Quiz</span>'
        f"<span>{title}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    quiz_key = f"quiz_{msg_idx}"
    if quiz_key not in st.session_state:
        st.session_state[quiz_key] = {"submitted": False, "answers": {}}

    for q_idx, q in enumerate(questions):
        st.markdown(
            f'<div class="quiz-question-block">'
            f'<div class="quiz-q-text">Q{q_idx+1}. {q["question"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        user_ans = st.radio(
            f"Select answer for Q{q_idx+1}",
            options=q["options"],
            key=f"{quiz_key}_q{q_idx}",
            label_visibility="collapsed",
        )
        st.session_state[quiz_key]["answers"][q_idx] = user_ans

        if st.session_state[quiz_key]["submitted"]:
            correct = q["answer"]
            picked_letter = user_ans[0] if user_ans else ""
            if picked_letter == correct:
                st.markdown(
                    f'<div class="quiz-result-correct">✅ Correct! {q["explanation"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="quiz-result-wrong">❌ Wrong — correct: {correct}. '
                    f'{q["explanation"]}</div>',
                    unsafe_allow_html=True,
                )

    if not st.session_state[quiz_key]["submitted"]:
        if st.button("✅ Check Answers", key=f"{quiz_key}_submit"):
            st.session_state[quiz_key]["submitted"] = True
            st.rerun()


def render_flowchart(content):
    """Render a Mermaid flowchart."""
    title = content.get("title", "Flowchart")
    mermaid_code = content.get("mermaid", "")
    description = content.get("description", "")

    st.markdown(
        f'<div class="response-card flowchart-card">'
        f'<div class="card-header">'
        f'<span class="card-badge badge-flowchart">🔀 Flowchart</span>'
        f"<span>{title}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Render using streamlit-mermaid
    try:
        from streamlit_mermaid import st_mermaid
        st_mermaid(mermaid_code, height=420)
    except ImportError:
        # Fallback: render via mermaid.ink
        import base64
        encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("ascii")
        st.image(f"https://mermaid.ink/img/{encoded}", use_container_width=True)

    if description:
        st.markdown(
            f'<div class="flowchart-desc">💡 {description}</div>',
            unsafe_allow_html=True,
        )


# Dispatcher
RENDERERS = {
    "explanation": lambda c, idx: render_explanation(c),
    "location": lambda c, idx: render_location(c),
    "flashcard": lambda c, idx: render_flashcards(c),
    "quiz": lambda c, idx: render_quiz(c, idx),
    "flowchart": lambda c, idx: render_flowchart(c),
}


def render_response(response_blocks, msg_idx):
    """Render a list of response blocks."""
    for block in response_blocks:
        cat = block.get("category", "explanation")
        content = block.get("content", {})
        renderer = RENDERERS.get(cat, RENDERERS["explanation"])
        renderer(content, msg_idx)


def render_followup_actions(msg_idx):
    """Render the follow-up action pills after a response."""
    st.markdown(
        f"""
        <div class="action-bar" id="actions-{msg_idx}">
            <span style="color: var(--text-secondary); font-size: 0.8rem; margin-right: 0.3rem;">
                Want more? →
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    actions = [
        ("🃏 Flashcards", "flashcard", "pill-flashcard"),
        ("🧪 Quiz", "quiz", "pill-quiz"),
        ("📍 Locate", "location", "pill-location"),
        ("📝 Explain", "explanation", "pill-explain"),
        ("🔀 Flowchart", "flowchart", "pill-flowchart"),
    ]
    for col, (label, cat, _css) in zip(cols, actions):
        with col:
            if st.button(label, key=f"action_{cat}_{msg_idx}", use_container_width=True):
                followup = get_dummy_response(cat)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "blocks": followup,
                        "response_type": cat,
                    }
                )
                st.rerun()


# ══════════════════════════════════════════════════════════════════
#  CHAT WINDOW
# ══════════════════════════════════════════════════════════════════

# Render existing chat history
for idx, msg in enumerate(st.session_state.chat_history):
    role = msg["role"]
    if role == "user":
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            render_response(msg.get("blocks", []), idx)
            render_followup_actions(idx)

# Chat input
if prompt := st.chat_input("Ask anything — explanations, quizzes, flashcards…"):
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate dummy response (explanation by default)
    response_blocks = get_initial_dummy_response(prompt)
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "blocks": response_blocks,
            "response_type": "explanation",
        }
    )
    st.rerun()