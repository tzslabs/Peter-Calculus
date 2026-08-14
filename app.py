"""DeltaBot's Streamlit interface."""

from __future__ import annotations

import random
import re

import ollama
import streamlit as st

import rag

# The 7B model is a little slower, but it handles the actual calculus much more
# reliably than the smaller models we tried.
MODEL_NAME = "qwen2.5:7b"
MAX_TOKENS = 1400
HISTORY_TURNS = 6
ASSISTANT_AVATAR = ":material/change_history:"

SAMPLE_QUESTIONS = [
    "What is the integral of 1/x dx?",
    "How do I solve derivatives using the chain rule?",
    "Explain the fundamental theorem of calculus",
    "How do I find the derivative of sin(x)?",
    "What is integration by parts?",
    "Explain the concept of limits",
    "How do I solve u-substitution problems?",
    "How do I find critical points?",
    "What is the second derivative test?",
    "How do I solve related rates problems?",
    "What is the mean value theorem?",
    "What is partial fraction decomposition?",
    "How do I solve optimization problems?",
    "What is L'Hopital's rule?",
    "What is the squeeze theorem?",
    "What is trigonometric substitution?",
    "How do I solve implicit differentiation?",
    "How do I evaluate improper integrals?",
    "What is the ratio test for series?",
    "What is Taylor series expansion?",
    "How do I find arc length of curves?",
    "How do I solve volume of revolution problems?",
    "What is partial differentiation?",
    "How do I solve Lagrange multipliers?",
    "What is the divergence theorem?",
    "How do I evaluate line integrals?",
]

st.set_page_config(
    page_title="DeltaBot",
    page_icon="Δ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root { --delta-accent: #77d6b3; }
        .stApp { background: #0d1117; }
        [data-testid="stMainBlockContainer"],
        .block-container {
            max-width: none;
            margin-left: 0;
            margin-right: auto;
            padding: 2.2rem 2rem 7rem !important;
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .75rem; }
        [data-testid="stSidebar"] hr { margin: .35rem 0; }
        [data-testid="stSidebarHeader"] {
            height: 0;
            margin-bottom: 0;
        }
        [data-testid="stLogoSpacer"] { display: none; }
        [data-testid="stSidebarCollapseButton"] {
            position: absolute;
            top: 1rem;
            right: 0;
            bottom: auto;
            z-index: 1000;
            margin: 0;
            transform: none;
            visibility: visible !important;
        }
        [data-testid="stSidebarCollapseButton"] button {
            min-width: 2rem;
            min-height: 3rem;
            padding: .45rem .2rem;
            background: #30343e;
            border: 1px solid rgba(255,255,255,.16);
            border-right: 0;
            border-radius: .65rem 0 0 .65rem;
        }
        [data-testid="stSidebarCollapseButton"] button:hover {
            background: #3a3f4b;
            border-color: rgba(119,214,179,.55);
        }
        .delta-header { margin-bottom: 1.8rem; }
        .delta-heading {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: .65rem;
        }
        .delta-label {
            color: var(--delta-accent);
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .delta-title {
            font-size: clamp(2.2rem, 5vw, 4.2rem);
            font-weight: 760;
            letter-spacing: -.055em;
            line-height: 1;
            margin: 0;
        }
        .delta-rule {
            background: linear-gradient(90deg, var(--delta-accent), transparent);
            height: 1px;
            width: 100%;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 14px;
            padding: .35rem .75rem;
            margin-bottom: .8rem;
        }
        [data-testid="stChatInput"] { border-radius: 12px; }
        div.stButton > button { text-align: left; border-radius: 9px; }
        .sidebar-mark { font-size: 1.4rem; font-weight: 750; letter-spacing: -.03em; }
        .sidebar-note { color: #929aa5; font-size: .82rem; line-height: 1.45; }
        .sidebar-about { padding: .05rem 0; }
        .sidebar-about-title {
            font-size: 1.15rem;
            font-weight: 650;
            line-height: 1.3;
            margin-bottom: .35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_index():
    return rag.build_index()


def check_ollama() -> str | None:
    """Return an error message if Ollama is not usable, otherwise None."""
    try:
        installed = {m.model for m in ollama.list().models}
    except Exception:
        return (
            "Could not reach the Ollama server. Start it with `ollama serve`, "
            "then reload this page."
        )
    # An untagged name matches any tag; a tagged name must match exactly.
    matches = (
        MODEL_NAME in installed
        or (":" not in MODEL_NAME
            and any(n.split(":")[0] == MODEL_NAME for n in installed))
    )
    if not matches:
        return f"Model `{MODEL_NAME}` is not installed. Run `ollama pull {MODEL_NAME}`."
    return None


def stream_answer(messages):
    """Yield response chunks from Ollama so the answer renders as it is generated."""
    for part in ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        options={"temperature": 0.3, "num_predict": MAX_TOKENS, "top_p": 0.9},
    ):
        yield part["message"]["content"]


def normalize_math(text: str) -> str:
    """Convert common model math delimiters into the form Streamlit renders."""
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\1$$", text, flags=re.DOTALL)
    return re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", text, flags=re.DOTALL)


def render_stream(messages) -> str:
    """Stream an answer while keeping equations renderable as they arrive."""
    answer = ""
    surface = st.empty()
    for chunk in stream_answer(messages):
        answer += chunk
        surface.markdown(normalize_math(answer) + " ▌")
    answer = normalize_math(answer)
    surface.markdown(answer)
    return answer


def render_sources(docs):
    if not docs:
        return
    with st.expander(f"Sources ({len(docs)} passages retrieved)"):
        for i, doc in enumerate(docs, start=1):
            st.caption(f"**[{i}]** {rag.citation(doc)}")
            st.text(doc.page_content.strip()[:600])


def main():
    st.markdown(
        '<div class="delta-header"><div class="delta-heading">'
        '<div class="delta-title">Δ DeltaBot</div>'
        '<div class="delta-label">Calculus tutor</div></div>'
        '<div class="delta-rule"></div></div>',
        unsafe_allow_html=True,
    )

    ollama_error = check_ollama()
    if ollama_error:
        st.error(ollama_error)
        st.stop()

    try:
        with st.spinner("Indexing course material (first run embeds the PDFs)..."):
            store, stats = load_index()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.markdown('<div class="sidebar-mark">Δ DeltaBot</div>', unsafe_allow_html=True)
        st.caption("Course index and session controls")
        st.divider()

        st.markdown(
            '<div class="sidebar-about"><div class="sidebar-about-title">About</div>'
            '<div class="sidebar-note">DeltaBot answers calculus questions using '
            'the textbooks in the local index. Each answer includes the source '
            'passages it used so you can check the work.</div></div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader("Library")
        col_a, col_b = st.columns(2)
        col_a.metric("Pages", f"{stats['pages']:,}")
        col_b.metric("Passages", f"{stats['chunks']:,}")
        st.markdown(
            f"**{stats['pdfs']} documents** indexed  \n"
            f"Model: `{MODEL_NAME}`  \n"
            f"Retrieval: {rag.DEFAULT_K} passages per question"
        )

        with st.expander("Index details"):
            st.markdown(
                f"Embedding model: `{rag.EMBEDDING_MODEL}`  \n"
                f"Chunk size: {stats['chunk_size']} characters  \n"
                f"Chunk overlap: {stats['chunk_overlap']} characters"
            )

        st.divider()
        st.subheader("Starting points")
        if "sample_questions" not in st.session_state:
            st.session_state.sample_questions = random.sample(SAMPLE_QUESTIONS, 5)
        for question in st.session_state.sample_questions:
            if st.button(question, key=question, use_container_width=True):
                st.session_state.pending_question = question

        st.divider()
        message_count = len(st.session_state.get("messages", [])) // 2
        st.caption(f"{message_count} question{'s' if message_count != 1 else ''} this session")
        if st.button("Clear conversation", use_container_width=True, disabled=not message_count):
            st.session_state.messages = []
            st.rerun()
        st.markdown(
            '<div class="sidebar-note">Use the arrow at the top of the sidebar '
            'to collapse it and give the solution more room.</div>',
            unsafe_allow_html=True,
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(normalize_math(message["content"]))
            render_sources(message.get("docs"))

    user_input = st.chat_input("Ask me any calculus question...")
    if not user_input:
        user_input = st.session_state.pop("pending_question", None)

    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Searching the course material..."):
            docs = rag.retrieve(store, user_input)

        # Only the current question needs fresh source text. Keeping old source
        # passages out of the prompt leaves more room for the actual solution.
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1][-HISTORY_TURNS:]
        ]
        messages = [
            {"role": "system", "content": rag.SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": rag.build_user_message(user_input, docs)},
        ]

        try:
            answer = render_stream(messages)
        except Exception as exc:
            answer = f"Sorry, generation failed: {exc}"
            st.error(answer)
            docs = []

        render_sources(docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "docs": docs}
    )


if __name__ == "__main__":
    main()
