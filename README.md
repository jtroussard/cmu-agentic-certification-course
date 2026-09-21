# Mail Organizer Pro

An agentic, local-first document processing pipeline that classifies, extracts metadata from, and routes PDF mail/bills into a canonical directory taxonomy using local LLMs and RAG memory.

---

## Features

- **Multi-Agent Pipeline**: Deterministic pre-classification, ReAct extraction, and taxonomy verification.
- **RAG Memory**: SQLite + `sqlite-vec` vector store that learns from human corrections to auto-route future documents.
- **Address & Confidence Gating**: Sends documents with missing account numbers or ungrounded property addresses to Human Review.
- **Human-in-the-Loop (HITL) GUI**: Streamlit dashboard for document upload, review queue staging, and archive history.
- **Deterministic Archiving**: Strict filename sanitization (`YYYY-MM-DD_Entity_Desc_Key.pdf`) and SHA-256 deduplication.

---

## Getting Started

### 1. Setup Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For Apple Silicon Metal acceleration:
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python

# For CPU-only:
# pip install llama-cpp-python
```

### 2. Seed Directory Taxonomy
```bash
python scripts/seed_dirs.py
```

### 3. Launch Application
```bash
streamlit run app.py
```
App deploys locally at: **http://localhost:8501**

---

## Utility Commands

- **Run Test Suite**:
  ```bash
  pytest
  ```
- **Reset Database & Clean Slate**:
  ```bash
  python scripts/purge_document.py "%" --clean-vector-store --clean-custom-dirs
  python scripts/seed_dirs.py
  ```
