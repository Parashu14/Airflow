# RAG Pipeline

An educational **Retrieval-Augmented Generation** pipeline built from scratch in Python. No API keys, no GPU, no model downloads — just Python and your documents.

```text
load documents → chunk → embed → index → retrieve → generate answer
```

---

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

# 2. Install the only required dependency
pip install pypdf

# 3. Run the demo — builds an index and asks a question
python -m rag_learning.cli demo
```

That's it. The demo uses sample documents in `data/source_docs/` and answers using **extractive sentence selection** — no external services needed.

---

## CLI Usage

Three commands available:

| Command | Description |
|---|---|
| `demo` | Build the index and ask a default question |
| `ingest` | Build (or rebuild) the vector index from source documents |
| `ask` | Ask a question against an existing index |

### `demo`

```bash
python -m rag_learning.cli demo
python -m rag_learning.cli demo --question "What is chunking?" --llm ollama
```

### `ingest`

```bash
python -m rag_learning.cli ingest
python -m rag_learning.cli ingest --chunk-size 120 --chunk-overlap 30
python -m rag_learning.cli ingest --source docs/ --index my_index.json
```

Flags: `--source`, `--index`, `--chunk-size`, `--chunk-overlap`

### `ask`

```bash
python -m rag_learning.cli ask "How does RAG work?"
python -m rag_learning.cli ask "What is gradient descent?" --show-context
python -m rag_learning.cli ask "What is overfitting?" --top-k 6 --llm ollama
```

Flags: `--index`, `--top-k`, `--llm` (`extractive` or `ollama`), `--show-context`

---

## Web UI (FastAPI + Gradio)

A chat interface and REST API are available with additional dependencies:

```bash
pip install fastapi uvicorn gradio httpx

# Start both servers
python -m rag_learning.serve
```

| Service | URL | Purpose |
|---|---|---|
| Gradio UI | http://127.0.0.1:7860 | Chat interface + index management |
| FastAPI | http://127.0.0.1:8000 | REST API |
| Swagger docs | http://127.0.0.1:8000/docs | API documentation |

Start servers individually:

```bash
python -m rag_learning.serve --api-only     # FastAPI only
python -m rag_learning.serve --gradio-only  # Gradio only
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/status` | Index status and configuration |
| `POST` | `/api/ask` | Ask a question |
| `POST` | `/api/ingest` | Rebuild the vector index |

---

## MLflow Tracking

Log index builds and queries as MLflow experiments:

```bash
pip install mlflow

python -m rag_learning.cli \
    --mlflow-tracking-uri sqlite:///mlflow.db \
    --experiment-name rag_pipeline \
    ingest

# View the dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Each `build_index` run logs: parameters (chunk size, overlap, dimensions), metrics (document count, chunk count, duration), the index artifact, and the pipeline as an MLflow pyfunc model.

Each `ask` run logs: query params, search metrics (top score, duration), and the answer text.

---

## Project Structure

```text
├── data/
│   ├── source_docs/         # Knowledge base (.md, .txt, .pdf)
│   └── index/               # Generated vector index (gitignored)
│
├── rag_learning/            # Python package
│   ├── cli.py               # Command-line interface
│   ├── pipeline.py          # build_index() and ask() orchestration
│   ├── document_loader.py   # File → Document objects
│   ├── chunking.py          # Document → Chunk objects (word-based)
│   ├── embeddings.py        # Text → Vector (HashingEmbedder)
│   ├── vector_store.py      # Vector storage + dot-product search
│   ├── llm.py               # Answer generation (extractive / ollama)
│   ├── schema.py            # Data dataclasses
│   ├── tracking.py          # Optional MLflow adapter
│   ├── mlflow_model.py      # MLflow pyfunc model wrapper
│   ├── api.py               # FastAPI REST server
│   ├── gradio_app.py        # Gradio chat UI
│   ├── serve.py             # Combined launcher
│   └── web_schemas.py       # Pydantic request/response models
│
├── tests/
│   ├── test_chunking.py
│   └── test_retrieval.py
│
├── docs/
│   └── ALTERNATIVES.md      # Upgrade paths and tool comparisons
│
├── requirements.txt
└── .env.example
```

---

## Running Tests

```bash
python -m unittest
```

Or run individual test files:

```bash
python -m unittest tests.test_chunking
python -m unittest tests.test_retrieval
```

---

## Next Steps

- **[Code Package Guide](rag_learning/README.md)** — Deep dive into each module's design and rationale
- **[Data Guide](data/README.md)** — Adding your own documents
- **[Tests Guide](tests/README.md)** — Writing tests for the pipeline
- **[Alternatives](docs/ALTERNATIVES.md)** — Upgrade paths: LangChain, LlamaIndex, FAISS, neural embeddings, etc.
- **[`.env.example`](.env.example)** — Environment configuration (Ollama, MLflow)

---

## License

[Apache 2.0](LICENSE)
