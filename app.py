"""
app.py  -  Streamlit UI for the Zyro Dynamics HR Assistant
==========================================================

Run it from the project folder with:

    streamlit run app.py

This file contains ONLY user-interface code. All the RAG work (loading PDFs,
chunking, embeddings, FAISS, the LLM, the guardrail) lives in rag_bot.py and
is used through one function:  ask_bot(question)

ask_bot() returns a dictionary:
    {"answer":  "text of the answer",
     "sources": [LangChain Document, ...]}      <- empty list if out of scope
"""

import html
import time

import streamlit as st

# ---------------------------------------------------------------------------
# 1. PAGE SETUP  (must be the very first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🏢",
    layout="centered",
)

# ---------------------------------------------------------------------------
# 1a. THEME / CSS
# ---------------------------------------------------------------------------
# Palette (flat, no gradients):
#   Navy        #0B2447  - header, primary buttons, headings
#   Blue accent #2F6FED  - links, focus states, active elements
#   Ink         #101828  - body text
#   Muted       #667085  - secondary text
#   Page bg     #F7F8FA  - app background
#   Card        #FFFFFF  - cards / bubbles
#   Border      #E4E7EC  - hairline borders
st.markdown(
    """
    <style>
        :root {
            --navy: #0B2447;
            --blue: #2F6FED;
            --ink: #101828;
            --muted: #667085;
            --page-bg: #F7F8FA;
            --card: #FFFFFF;
            --border: #E4E7EC;
            --user-bubble: #EAF1FE;
            --success: #12B76A;
        }

        .stApp { background: var(--page-bg); }
        .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 840px; }

        /* ================= HEADER ================= */
        .hr-header {
            display: flex; align-items: center; gap: 14px;
            padding: 18px 22px;
            background: var(--navy);
            border-radius: 12px;
            margin-bottom: 4px;
        }
        .hr-header .hr-logo {
            width: 42px; height: 42px; border-radius: 8px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.18);
            display: flex; align-items: center; justify-content: center;
            font-size: 15px; font-weight: 700; letter-spacing: 0.02em;
            color: #FFFFFF; flex-shrink: 0;
        }
        .hr-header .hr-title { color: #FFFFFF; font-size: 1.28rem; font-weight: 650; line-height: 1.25; }
        .hr-header .hr-tagline { color: #AEC1DC; font-size: 0.82rem; margin-top: 2px; }
        .app-subtitle {
            color: var(--muted); font-size: 0.88rem;
            margin: 10px 2px 18px 2px; line-height: 1.5;
        }

        /* ================= CHAT MESSAGE LAYOUT ================= */
        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        div[data-testid="stChatMessage"] {
            animation: fadeSlideIn 0.32s ease;
            padding: 3px 0;
            background: transparent;
        }
        div[data-testid="stChatMessageAvatarUser"],
        div[data-testid="stChatMessageAvatarAssistant"] {
            background: var(--navy) !important;
        }

        /* User message: right-aligned bubble */
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"])
            div[data-testid="stChatMessageContent"] {
            background: var(--user-bubble);
            border: 1px solid #D7E6FD;
            border-radius: 14px 14px 3px 14px;
            padding: 10px 14px;
            color: var(--ink);
            margin-left: auto;
        }

        /* Assistant message: left-aligned card */
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"])
            div[data-testid="stChatMessageContent"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px 14px 14px 3px;
            padding: 12px 15px;
            color: var(--ink);
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }

        /* ================= SOURCE CITATION CARDS ================= */
        .source-label {
            font-size: 0.72rem; font-weight: 700; color: var(--muted);
            margin-top: 12px; margin-bottom: 6px; text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .source-card {
            display: flex; align-items: center; gap: 9px;
            background: var(--page-bg); border: 1px solid var(--border);
            border-radius: 8px; padding: 7px 11px; margin: 0 0 6px 0;
            font-size: 0.84rem; color: var(--ink);
            transition: border-color 0.15s ease;
        }
        .source-card:hover { border-color: var(--blue); }
        .source-card .src-icon { font-size: 0.95rem; flex-shrink: 0; opacity: 0.75; }
        .source-card .src-file { font-weight: 600; }
        .source-card .src-page { color: var(--muted); font-size: 0.78rem; }

        /* ================= OUT-OF-SCOPE CARD ================= */
        .oos-card {
            display: flex; gap: 10px; align-items: flex-start;
            background: #FFFAEB; border: 1px solid #FEDF89;
            border-left: 3px solid #B54708;
            border-radius: 10px; padding: 12px 14px; margin-top: 2px;
        }
        .oos-card .oos-title { font-weight: 700; color: #93370D; font-size: 0.78rem;
                                text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 3px; }
        .oos-card .oos-text { color: #7A2E0E; font-size: 0.9rem; line-height: 1.45; }

        /* ================= TYPING INDICATOR ================= */
        .typing-dots { display: inline-flex; align-items: center; gap: 4px; padding: 2px 0; }
        .typing-dots span {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--muted);
            animation: typingPulse 1.1s infinite ease-in-out;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
        @keyframes typingPulse {
            0%, 80%, 100% { opacity: 0.25; }
            40% { opacity: 1; }
        }

        /* ================= SIDEBAR NAV ================= */
        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: var(--navy);
        }
        .nav-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
        .nav-brand .nav-logo {
            width: 30px; height: 30px; border-radius: 7px; background: var(--navy);
            display: flex; align-items: center; justify-content: center;
            color: #fff; font-size: 12px; font-weight: 700;
        }
        .nav-brand .nav-name { font-weight: 650; color: var(--navy); font-size: 0.95rem; }

        .nav-section-label {
            font-size: 0.72rem; font-weight: 700; color: var(--muted);
            text-transform: uppercase; letter-spacing: 0.06em; margin: 4px 0 6px 0;
        }
        .policy-list { margin: 0 0 4px 0; padding: 0; list-style: none; }
        .policy-list li {
            display: flex; align-items: center; gap: 8px;
            font-size: 0.85rem; color: var(--ink);
            padding: 5px 2px;
        }
        .policy-list li::before {
            content: ""; width: 5px; height: 5px; border-radius: 50%;
            background: var(--blue); flex-shrink: 0;
        }

        section[data-testid="stSidebar"] .stButton > button {
            width: 100%; text-align: left;
            background: var(--card); border: 1px solid var(--border);
            color: var(--ink); border-radius: 9px;
            padding: 9px 12px; font-size: 0.84rem;
            box-shadow: 0 1px 2px rgba(16,24,40,0.03);
            transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            border-color: var(--blue);
            box-shadow: 0 2px 6px rgba(47,111,237,0.14);
            transform: translateY(-1px);
        }
        section[data-testid="stSidebar"] .stButton > button:active {
            transform: translateY(0);
        }
        section[data-testid="stSidebar"] .stButton > button:focus:not(:active) {
            border-color: var(--blue); color: var(--navy);
        }

        .status-footer {
            display: flex; align-items: center; gap: 7px;
            font-size: 0.78rem; color: var(--muted); margin-top: 4px;
        }
        .status-dot {
            width: 7px; height: 7px; border-radius: 50%; background: var(--success);
            flex-shrink: 0;
        }

        /* ================= CHAT INPUT ================= */
        div[data-testid="stChatInput"] {
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(16,24,40,0.05);
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stChatInput"]:focus-within {
            border-color: var(--blue);
            box-shadow: 0 0 0 3px rgba(47,111,237,0.12);
        }

        /* ================= RESPONSIVE ================= */
        @media (max-width: 640px) {
            .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
            .hr-header { padding: 14px 16px; }
            .hr-header .hr-title { font-size: 1.08rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 2. LOAD THE RAG BACKEND ONCE
# ---------------------------------------------------------------------------
# Streamlit re-runs this whole file every time the user clicks or types.
# Loading PDFs + embedding model + FAISS is slow, so we must NOT redo it on
# every rerun. @st.cache_resource runs the function once and remembers the
# result for as long as the app is running.
#
# Importing rag_bot (a normal .py file) is what runs your PDF loading,
# chunking, embeddings, FAISS and LLM setup. We import it INSIDE the function
# so that the "Loading..." spinner is shown while it happens.
@st.cache_resource(show_spinner="Loading HR policies and AI models (first run only)...")
def load_bot():
    from rag_bot import ask_bot
    return ask_bot


try:
    ask_bot = load_bot()
except Exception as e:  # e.g. missing GROQ_API_KEY or missing PDF folder
    st.error(
        "The HR assistant backend could not start. Check that your `.env` file "
        "contains GROQ_API_KEY and that the `zyro-dynamics-hr-corpus` folder is "
        "next to app.py."
    )
    st.exception(e)
    st.stop()


# ---------------------------------------------------------------------------
# 3. SESSION STATE = chat history
# ---------------------------------------------------------------------------
# st.session_state is a dictionary that survives reruns (but is reset when the
# browser tab is refreshed). Each message we store looks like:
#   {"role": "user" | "assistant",
#    "content": "text",
#    "kind": "answer" | "out_of_scope" | "error",   (assistant messages)
#    "sources": [{"file": "02_Leave_Policy.pdf", "page": 3}, ...]}
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def extract_sources(docs):
    """Turn LangChain Documents into a simple, de-duplicated list of
    {"file": ..., "page": ...} dictionaries.

    - Each retrieved chunk carries metadata: {"source": <pdf path>, "page": <int>}.
    - PyPDF page numbers start at 0, so we add 1 to show the page a human
      would see when opening the PDF.
    - Several chunks can come from the same page, so we remove duplicates
      while keeping the original (most relevant first) order.
    """
    seen = set()
    sources = []
    for doc in docs:
        meta = doc.metadata or {}
        # keep only the file name, e.g. "02_Leave_Policy.pdf" (handles / and \)
        file_name = str(meta.get("source", "Unknown source")).replace("\\", "/").split("/")[-1]
        page = meta.get("page")
        page_number = page + 1 if isinstance(page, int) else None

        key = (file_name, page_number)
        if key not in seen:
            seen.add(key)
            sources.append({"file": file_name, "page": page_number})
    return sources


def safe_text(text):
    """Streamlit treats $...$ as a math formula. Escape $ so amounts such as
    "$50 and $100" are shown as normal text."""
    return text.replace("$", "\\$")


def render_sources(sources):
    """Show source citations as compact document cards (PDF name + page)."""
    if not sources:
        return
    cards = ""
    for s in sources:
        file_name = html.escape(s["file"])
        page_html = (
            f'<span class="src-page">Page {s["page"]}</span>'
            if s["page"] is not None
            else ""
        )
        cards += (
            '<div class="source-card">'
            '<span class="src-icon">📄</span>'
            f'<span class="src-file">{file_name}</span>{page_html}'
            "</div>"
        )
    st.markdown(
        f'<div class="source-label">Source documents</div>{cards}',
        unsafe_allow_html=True,
    )


def render_message(msg):
    """Draw ONE chat message. Used both for replaying history and for
    showing a brand-new reply, so both look identical."""
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(safe_text(msg["content"]))
        return

    with st.chat_message("assistant"):
        if msg["kind"] == "out_of_scope":
            # Clearly flag questions the guardrail rejected, styled to match
            # the rest of the design system rather than a generic warning box.
            st.markdown(
                '<div class="oos-card">'
                '<div><div class="oos-title">Outside HR policy scope</div>'
                f'<div class="oos-text">{html.escape(msg["content"])}</div></div>'
                "</div>",
                unsafe_allow_html=True,
            )
        elif msg["kind"] == "error":
            st.error(msg["content"])
        else:
            st.markdown(safe_text(msg["content"]))
            render_sources(msg["sources"])


def get_reply(question):
    """Call your existing ask_bot() and convert its result into a message."""
    try:
        result = ask_bot(question)
    except Exception as e:
        return {
            "role": "assistant",
            "kind": "error",
            "content": f"Sorry, I couldn't get an answer right now. Please try again. ({type(e).__name__}: {e})",
            "sources": [],
        }

    # ask_bot() returns sources=[] only when the guardrail said OUT_OF_SCOPE,
    # so an empty source list = out-of-scope question.
    if not result["sources"]:
        kind = "out_of_scope"
    else:
        kind = "answer"

    return {
        "role": "assistant",
        "kind": kind,
        "content": result["answer"],
        "sources": extract_sources(result["sources"]),
    }


# ---------------------------------------------------------------------------
# 5. HEADER + SIDEBAR
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hr-header">
        <div class="hr-logo">ZD</div>
        <div>
            <div class="hr-title">Zyro Dynamics HR Assistant</div>
            <div class="hr-tagline">Internal policy assistant · answers backed by official HR documents</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Ask a question about leave, benefits, work from home, '
    "conduct, or any other company policy. Every answer cites the exact "
    "document and page it came from.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="nav-brand"><div class="nav-logo">ZD</div>'
        '<div class="nav-name">Zyro Dynamics</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("HR Policy Assistant")
    st.divider()

    st.markdown('<div class="nav-section-label">Policy categories</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="policy-list">
            <li>Leave & Attendance</li>
            <li>Work From Home</li>
            <li>Compensation & Benefits</li>
            <li>Performance Reviews</li>
            <li>Travel & Expenses</li>
            <li>IT & Data Security</li>
            <li>Code of Conduct &amp; POSH</li>
            <li>Onboarding & Separation</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<div class="nav-section-label">Suggested questions</div>', unsafe_allow_html=True)
    example_questions = [
        "How many days of Earned Leave can be carried forward?",
        "What is the internet reimbursement limit?",
        "How many casual leaves do I get per year?",
    ]
    for i, q in enumerate(example_questions):
        if st.button(q, key=f"example_{i}"):
            # We can't call the bot from here, so we "queue" the question;
            # it is picked up by the input-handling code near the bottom.
            st.session_state.queued_question = q

    st.caption("Out-of-scope example")
    if st.button("What's the capital of France?", key="example_oos"):
        st.session_state.queued_question = "What's the capital of France?"

    st.divider()
    st.markdown('<div class="nav-section-label">Session</div>', unsafe_allow_html=True)
    if st.button("Clear conversation", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        '<div class="status-footer"><span class="status-dot"></span>'
        "System ready</div>",
        unsafe_allow_html=True,
    )
    st.caption("Confidential · Internal use only. Chat history is kept for this session only.")


# ---------------------------------------------------------------------------
# 6. SHOW CHAT HISTORY
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hello, I'm the Zyro Dynamics HR Assistant. Ask me anything about "
            "company HR policies, for example *“How many casual leaves do I get "
            "per year?”*"
        )

for message in st.session_state.messages:
    render_message(message)


# ---------------------------------------------------------------------------
# 7. HANDLE A NEW QUESTION
# ---------------------------------------------------------------------------
# The question can come from the chat box OR from a sidebar example button.
user_input = st.chat_input("Ask a question about company HR policies...")
queued = st.session_state.pop("queued_question", None)
question = (user_input or queued or "").strip()

if question:
    # 1) show + store the user's question
    user_msg = {"role": "user", "content": question}
    render_message(user_msg)
    st.session_state.messages.append(user_msg)

    # 2) show a typing indicator in an assistant bubble while ask_bot() runs
    #    (guardrail -> retrieval -> LLM), then swap it for the real reply.
    typing_slot = st.empty()
    with typing_slot.container():
        with st.chat_message("assistant"):
            st.markdown(
                '<div class="typing-dots"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )
    bot_msg = get_reply(question)
    typing_slot.empty()

    # 3) show + store the reply (so it is still there on the next rerun)
    render_message(bot_msg)
    st.session_state.messages.append(bot_msg)
