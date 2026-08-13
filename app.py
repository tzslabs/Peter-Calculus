"""Peter Calculus — a retrieval-augmented calculus tutor running fully locally."""

from __future__ import annotations

import random

import ollama
import streamlit as st

import rag

# qwen2.5:7b is markedly more reliable on the actual mathematics than the
# smaller llama3.2, which produced confidently wrong derivatives. It is also
# verbose, so it needs a generous token budget to reach its final answer.
MODEL_NAME = "qwen2.5:7b"
MAX_TOKENS = 1400
HISTORY_TURNS = 6

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

st.set_page_config(page_title="Peter Calculus", page_icon="∞", layout="centered")


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


def render_sources(docs):
    if not docs:
        return
    with st.expander(f"Sources ({len(docs)} passages retrieved)"):
        for i, doc in enumerate(docs, start=1):
            st.caption(f"**[{i}]** {rag.citation(doc)}")
            st.text(doc.page_content.strip()[:600])


def main():
    st.title("∞ Peter Calculus")
    st.caption("A calculus tutor grounded in your course PDFs. Runs entirely on your machine.")

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
        st.subheader("Index")
        st.markdown(
            f"- **{stats['pdfs']}** PDFs, **{stats['pages']:,}** pages\n"
            f"- **{stats['chunks']:,}** chunks "
            f"({stats['chunk_size']} chars, {stats['chunk_overlap']} overlap)\n"
            f"- Embeddings: `{rag.EMBEDDING_MODEL}`\n"
            f"- Model: `{MODEL_NAME}` via Ollama\n"
            f"- Retrieval: top-{rag.DEFAULT_K} by cosine similarity"
        )

        st.subheader("Try a question")
        if "sample_questions" not in st.session_state:
            st.session_state.sample_questions = random.sample(SAMPLE_QUESTIONS, 5)
        for question in st.session_state.sample_questions:
            if st.button(question, key=question, use_container_width=True):
                st.session_state.pending_question = question

        if st.session_state.get("messages"):
            st.divider()
            if st.button("Clear chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("docs"))

    user_input = st.chat_input("Ask me any calculus question...")
    if not user_input:
        user_input = st.session_state.pop("pending_question", None)

    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching the course material..."):
            docs = rag.retrieve(store, user_input)

        # Prior turns go in as plain conversation; only the current question
        # carries retrieved context, so the window stays small.
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
            answer = st.write_stream(stream_answer(messages))
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
