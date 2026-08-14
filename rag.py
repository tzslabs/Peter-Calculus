"""Retrieval pipeline for DeltaBot.

The retrieval code stays separate from Streamlit so it is also useful from
scripts and tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SOURCES_DIR = Path("sources")
INDEX_DIR = Path(".chroma")
FINGERPRINT_FILE = INDEX_DIR / "fingerprint.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_K = 5

SYSTEM_PROMPT = """You are a calculus tutor working with a bright but confused student.

Guidelines:
- Give clear, step-by-step explanations rather than bare answers.
- Use plain language; introduce jargon only after explaining it.
- Where it helps, ground an abstract idea in an analogy or concrete example.
- If the reference material does not cover the question and you are unsure, say
  so plainly instead of inventing a result.
- Format all mathematics as LaTeX: $...$ for inline, $$...$$ for display.
  For example $\\int f(x)\\,dx$, $\\frac{dy}{dx}$, $\\lim_{x \\to a} f(x)$.
- Never put LaTeX inside a Markdown code block. Do not use bare square brackets
  or plain-text approximations for equations.
"""


def load_pages(sources_dir: Path = SOURCES_DIR) -> list[Document]:
    """Load every PDF in `sources_dir` as one Document per page.

    Page-level granularity is what lets retrieved chunks carry a citation back
    to a specific page.
    """
    pages: list[Document] = []
    for pdf_path in sorted(sources_dir.glob("*.pdf")):
        for page in PyMuPDFLoader(str(pdf_path)).load():
            # PDF page numbers start at zero here, so save the number people see.
            page.metadata["source_name"] = pdf_path.stem
            page.metadata["page_label"] = page.metadata.get("page", 0) + 1
            pages.append(page)
    return pages


def split_pages(
    pages: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split pages into overlapping chunks, preserving page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    # A few textbook pages produce only a page number or a blank fragment.
    return [c for c in chunks if len(c.page_content.strip()) > 40]


def fingerprint(sources_dir: Path = SOURCES_DIR, **params) -> str:
    """Hash the corpus and chunking params so the index rebuilds when they change."""
    parts = [
        f"{p.name}:{p.stat().st_size}" for p in sorted(sources_dir.glob("*.pdf"))
    ]
    parts.append(json.dumps(params, sort_keys=True))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(
    sources_dir: Path = SOURCES_DIR,
    index_dir: Path = INDEX_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force: bool = False,
) -> tuple[Chroma, dict]:
    """Return a Chroma index over the PDFs, rebuilding only when inputs change.

    Returns the store plus a stats dict describing the corpus, which the UI
    surfaces so it is obvious how much text is actually indexed.
    """
    if not sources_dir.exists() or not any(sources_dir.glob("*.pdf")):
        raise FileNotFoundError(
            f"No PDFs found in {sources_dir}/. Add calculus PDFs and restart."
        )

    current = fingerprint(
        sources_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding=EMBEDDING_MODEL,
    )
    embeddings = get_embeddings()

    if not force and FINGERPRINT_FILE.exists():
        cached = json.loads(FINGERPRINT_FILE.read_text())
        if cached.get("fingerprint") == current:
            store = Chroma(
                persist_directory=str(index_dir),
                embedding_function=embeddings,
            )
            return store, cached["stats"]

    pages = load_pages(sources_dir)
    chunks = split_pages(pages, chunk_size, chunk_overlap)

    # Starting fresh prevents passages from removed PDFs hanging around.
    store = Chroma(
        persist_directory=str(index_dir),
        embedding_function=embeddings,
    )
    store.reset_collection()
    store.add_documents(chunks)

    stats = {
        "pdfs": len({c.metadata.get("source_name") for c in chunks}),
        "pages": len(pages),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_FILE.write_text(
        json.dumps({"fingerprint": current, "stats": stats}, indent=2)
    )
    return store, stats


def retrieve(store: Chroma, query: str, k: int = DEFAULT_K) -> list[Document]:
    return store.as_retriever(search_kwargs={"k": k}).invoke(query)


def citation(doc: Document) -> str:
    name = doc.metadata.get("source_name", "source")
    return f"{name}, p.{doc.metadata.get('page_label', '?')}"


def format_context(docs: list[Document]) -> str:
    """Render retrieved chunks with citation labels the model can reference."""
    return "\n\n".join(
        f"[{i}] ({citation(doc)})\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs, start=1)
    )


def build_user_message(question: str, docs: list[Document]) -> str:
    if not docs:
        return question
    return (
        f"Reference material from the course PDFs:\n\n{format_context(docs)}\n\n"
        f"Using the material above where it is relevant, answer the student's "
        f"question. Cite the passages you rely on as [1], [2], etc.\n\n"
        f"Question: {question}"
    )
