# Peter Calculus

Free, local calculus tutor chatbot built with Streamlit + Ollama + ChromaDB.

## Quick Start
```bash
# 1) Install Python deps
pip install -r requirements.txt

# 2) Install Ollama (macOS)
brew install ollama
brew services start ollama

# 3) Download a model
ollama pull llama3.2

# 4) (Optional) add PDFs to sources/
# Place your calculus PDFs in the sources/ folder

# 5) Run
streamlit run app_new.py
```

## Notes
- Everything runs locally after the model is downloaded.
- To use a smaller/faster model, update `model_name` in `app_new.py` (e.g. `llama3.2:1b`).
