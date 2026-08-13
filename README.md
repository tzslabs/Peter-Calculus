# Peter Calculus

A retrieval-augmented calculus tutor that runs entirely on your own machine — local
embeddings, a local vector store, and a local LLM. No API keys, no cloud costs.

Ask a calculus question and the app retrieves the most relevant passages from the
indexed textbooks, feeds them to the model as grounding, and shows you which pages
the answer came from.

## Stack

| Layer | Choice |
|---|---|
| Interface | Streamlit (streaming chat) |
| Generation | Llama 3.2 via [Ollama](https://ollama.com) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, CPU) |
| Vector store | Chroma, persisted to `.chroma/` |
| PDF parsing | PyMuPDF via LangChain's `PyMuPDFLoader` |

## Corpus

4 PDFs → **2,430 pages → 4,601 chunks**, embedded in ~76s on a CPU.

- OpenStax *Calculus* Volumes 1–3 (CC BY-NC-SA 4.0) — fetched by `fetch_sources.py`
- Paul Dawkins' *Calculus Cheat Sheet* — small, ships with the repo

## How retrieval works

PDFs are loaded **one document per page**, then split with
`RecursiveCharacterTextSplitter` into 1000-character chunks with 150 characters of
overlap. Page-level loading is what lets each chunk carry a `source, page` citation
through to the UI.

The index is keyed by a fingerprint of the corpus (filenames + sizes) and the
chunking parameters, so it is built once and reused; changing a PDF or a chunk
setting triggers an automatic rebuild rather than silently serving stale vectors.

At query time the top 5 chunks by cosine similarity are formatted with citation
markers and passed to the model, which is instructed to cite the passages it uses.
Conversation history is passed as prior chat turns, and only the current question
carries retrieved context — which keeps the context window from growing with the
conversation.

## Setup

```bash
# 1. Python deps
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Ollama + model
brew install ollama
ollama serve            # leave running in its own terminal
ollama pull llama3.2

# 3. Textbook PDFs (~170 MB, gitignored)
python fetch_sources.py

# 4. Run
streamlit run app.py
```

The first launch parses and embeds the corpus, which takes a few minutes on CPU.
Later launches load the persisted index and start immediately.

## Layout

```
app.py            Streamlit UI — chat, sidebar, source display
rag.py            Ingestion, chunking, indexing, retrieval, prompt construction
fetch_sources.py  Downloads the OpenStax volumes
sources/          Course PDFs
```

`rag.py` deliberately imports no Streamlit, so the retrieval pipeline can be
imported and evaluated from a plain script.

## Notes

- To trade quality for speed, `ollama pull llama3.2:1b` and set `MODEL_NAME` in
  `app.py`.
- Delete `.chroma/` to force a clean rebuild of the index.
