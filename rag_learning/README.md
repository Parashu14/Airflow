# Code Package Guide

This folder contains the Python package for the RAG learning project.

The code is intentionally split by pipeline stage. That makes it easier to learn because each file answers one question:

```text
document_loader.py -> How do files become text?
chunking.py        -> How does long text become searchable pieces?
embeddings.py      -> How does text become vectors?
vector_store.py    -> How do we store and search vectors?
llm.py             -> How do retrieved chunks become an answer?
pipeline.py        -> How are the pieces connected?
cli.py             -> How does a user run the pipeline?
```

## Why This Package Uses Small Files

Real RAG apps can become confusing quickly because loading, chunking, embedding, retrieval, prompting, and generation all affect answer quality.

This project keeps each concern separate so you can change one layer at a time. For example, you can replace the embedder without rewriting the document loader.

## `schema.py`

Defines the shared data shapes used by the rest of the package.

### `Document`

Represents text loaded from a source file.

It stores:

- `text`: the extracted text
- `metadata`: details like file name, file type, and page number

Why this exists:

RAG needs both content and provenance. The text is used for retrieval. The metadata is used for citations.

Alternative:

You could use plain dictionaries everywhere, but dataclasses make the code easier to read and help you understand what each stage expects.

### `Chunk`

Represents a smaller piece of a document.

Why this exists:

Retrieval works better on focused text than on whole documents. A chunk keeps its text plus source metadata.

### `SearchResult`

Represents one retrieved chunk and its similarity score.

Why this exists:

The answer step should know not only which text was retrieved, but also how strongly it matched the question.

### `RagAnswer`

Represents the final pipeline output.

It stores:

- answer text
- source list
- generator name
- retrieved results

Why this exists:

Keeping the final answer structured makes it easy to build a UI later.

## `document_loader.py`

Turns files from `data/source_docs/` into `Document` objects.

### `load_documents(source_dir)`

Reads all supported files from a folder.

Supported files:

- `.md`
- `.txt`
- `.pdf`

Why this function:

Every RAG pipeline starts with ingestion. This function gives the rest of the pipeline a clean list of `Document` objects, regardless of the original file type.

### `_load_text_file(path, root)`

Reads markdown or text files as UTF-8 text.

Why this function:

Markdown and plain text are the easiest formats to support first. They are great for learning because they do not require OCR or layout parsing.

### `_load_pdf(path, root)`

Reads PDF pages with `pypdf` and creates one `Document` per page.

Why this function:

PDFs are common in real RAG use cases. Keeping page numbers in metadata allows page-level citations.

### `_relative_source(path, root)`

Stores a readable source path instead of a full machine path.

Why this function:

Answers should cite `rag_course_notes.md`, not a long local Windows path.

## `chunking.py`

Splits documents into overlapping word chunks.

### `chunk_documents(documents, chunk_size=180, chunk_overlap=40)`

Creates searchable chunks from documents.

Why this function:

Most documents are too broad for direct retrieval. Chunking gives the search step smaller evidence units.

Why word-based chunking:

It is simple, transparent, and easy to debug. For a learning project, seeing exactly where chunks begin and end matters more than having the most advanced splitter.

Why overlap:

Important meaning can sit between two chunks. Overlap keeps neighboring context together.

### `_make_chunk_id(text, metadata)`

Creates a stable id for each chunk.

Why this function:

Vector stores need a way to identify chunks. A stable id also helps with debugging and future updates.

## `embeddings.py`

Converts text into numeric vectors.

### `HashingEmbedder`

A small local embedder that uses hashed word and bigram features.

Why this class:

The embedder has configuration, especially `dimensions`, so a class is clearer than many loose functions.

Why this tool:

It runs without API keys, downloads, model files, GPU setup, or external services. That makes the first lesson reliable.

Important limitation:

This is not a neural semantic embedder. It is good enough to learn the RAG shape, but it will miss many meaning-based matches.

### `embed(text)`

Turns one string into one vector.

Why this function:

During retrieval, the user question must be embedded the same way as document chunks.

### `embed_many(texts)`

Embeds many strings.

Why this function:

During indexing, we embed all chunks together.

### `_features(text)`

Extracts tokens and bigrams.

Why this function:

Single words catch exact matches. Bigrams catch small phrases like `gradient descent` and `vector search`.

### `_stable_hash(value)`

Maps a feature to a repeatable numeric value.

Why this function:

Python's built-in `hash()` can change between runs. A vector index needs stable values.

### `_l2_normalize(vector)`

Normalizes vector length.

Why this function:

Cosine similarity works best when vectors are normalized.

## `vector_store.py`

Stores chunk vectors and retrieves the most similar chunks.

### `VectorStore`

A JSON-backed local vector store.

Why this class:

The vector store has state: records, vectors, and an embedder. A class keeps those together.

Why JSON:

It is transparent. You can open `data/index/rag_index.json` and see what the pipeline saved.

Important limitation:

JSON search is fine for learning and small collections. It is not built for millions of chunks.

### `from_chunks(chunks, embedder=None)`

Builds a vector store from chunk objects.

Why this function:

Indexing means converting chunks into vectors and storing them with metadata.

### `search(query, top_k=4)`

Embeds the user question and returns the closest chunks.

Why this function:

This is the retrieval step in RAG.

Why `top_k`:

The generator needs enough context to answer, but not so much that it gets distracted.

### `save(path)` and `load(path)`

Persist and reload the index.

Why these functions:

You should not have to reprocess all documents every time you ask a question.

### `_dot_product(left, right)`

Computes similarity between normalized vectors.

Why this function:

For normalized vectors, dot product is cosine similarity.

## `llm.py`

Turns retrieved chunks into an answer.

### `answer_question(question, results, llm=None)`

Chooses a generator and returns a `RagAnswer`.

Available modes:

- `extractive`: local sentence selection
- `ollama`: optional local LLM through Ollama

Why this function:

RAG generation should be a separate step from retrieval. That lets us compare different answer generators using the same retrieved context.

### `build_grounded_prompt(question, results)`

Builds the prompt for a real LLM.

Why this function:

The prompt is where we tell the model to answer only from retrieved context and cite sources.

### `_answer_with_ollama(question, results)`

Sends the grounded prompt to a local Ollama model.

Why this function:

Ollama lets you test LLM generation locally without a cloud API key.

### `_answer_extractively(question, results)`

Builds an answer by selecting relevant sentences from retrieved chunks.

Why this function:

It works everywhere and makes retrieval behavior easy to inspect. If the retrieved chunks are weak, the answer will be weak too, which is useful for learning.

### `_format_source(result)`

Formats metadata into a source citation.

Why this function:

RAG answers should be verifiable. Source formatting keeps citations consistent.

## `pipeline.py`

Connects the package modules into the two main workflows.

### `build_index(...)`

Runs:

```text
load documents -> chunk documents -> embed chunks -> save vector index
```

Why this function:

Indexing is usually done before users ask questions.

### `ask(...)`

Runs:

```text
load vector index -> retrieve chunks -> answer question
```

Why this function:

Question answering should reuse the existing index.

## `cli.py`

Provides the command-line interface.

### `main()`

Defines commands:

- `demo`
- `ingest`
- `ask`

Why this function:

A CLI is the fastest way to learn and test the pipeline before building a web app.

### `_print_answer(...)`

Displays answer, generator, and sources.

Why this function:

Separating display from pipeline logic keeps the core RAG code reusable.

### `_print_context(...)`

Displays retrieved chunks and scores when `--show-context` is used.

Why this function:

Inspecting retrieved context is one of the best ways to debug RAG quality.

## `__init__.py`

Marks `rag_learning` as a Python package.

Why this file:

It allows commands like:

```powershell
python -m rag_learning.cli demo
```

## `tracking.py`

A thin optional wrapper around MLflow.

Why this module:

Tracking index builds and queries helps you compare experiments. Every function is a no-op when MLflow is not installed, so the pipeline runs without it.

### `enabled()`

Checks whether MLflow is available.

### `start_run(run_name, experiment_name, tags, nested)`

Wraps `mlflow.start_run()`. Starts a new run or a nested child run.

### `log_params(params)`

Logs key-value parameters (chunk size, overlap, etc.).

### `log_metrics(metrics)`

Logs numeric metrics (document count, duration, scores).

### `log_artifact(path)`

Logs a file as an artifact.

### `log_text(text, artifact_path)`

Logs a string as a text artifact.

### `end_run(status)`

Ends the active MLflow run.

### `set_tracking_uri(uri)`

Sets the MLflow tracking server URI.

## `mlflow_model.py`

Wraps the RAG pipeline as an MLflow pyfunc model for serving and deployment.

Why this module:

Packaging the pipeline as a model allows deployment via `mlflow models serve` and integration with MLflow's model registry.

### `RagPipelineModel`

An MLflow `PythonModel` that loads the vector store on startup and serves predictions.

- `__init__(index_path, llm)` — stores the index location
- `load_context(context)` — pre-loads the vector store into memory
- `predict(context, model_input)` — accepts a list of questions, returns answers with sources

### `log_rag_pipeline(index_path, llm, artifact_path)`

Logs the `RagPipelineModel` to the current MLflow run. Called automatically by `build_index()` when MLflow is enabled.

## `api.py`

A FastAPI server that exposes the pipeline as REST endpoints.

Why this module:

A REST API decouples the pipeline logic from any specific frontend. It also provides Swagger documentation at `/docs`.

### Endpoints

- `GET /health` — health check
- `GET /api/status` — returns index status, document and chunk counts, and available source files
- `POST /api/ask` — accepts `AskRequest`, returns `AskResponse` with answer, sources, and per-chunk scores
- `POST /api/ingest` — accepts `IngestRequest`, rebuilds the index, returns document/chunk count and duration

Why these endpoints:

The ask-and-ingest pattern matches the two pipeline workflows. The status endpoint helps frontends display pipeline state.

## `gradio_app.py`

A Gradio chat UI that calls the FastAPI backend.

Why this module:

A visual interface makes it easier to iterate and test. The three-tab layout separates chatting from configuration and monitoring.

### Tabs

- **Chat** — conversation history, question input, top-k slider, generator selector, context toggle
- **Index Management** — source directory, chunk size/overlap sliders, rebuild button
- **Status** — pipeline configuration table and list of available source files

Why three tabs:

Chat is the primary user interaction. Index Management controls data ingestion. Status helps with debugging and understanding the current state.

## `serve.py`

Unified launcher that starts both the FastAPI and Gradio servers.

Why this module:

Running two servers should be a single command for development. The script supports `--api-only`, `--gradio-only`, and custom host/port flags.

### `main()`

Parses CLI arguments and starts the servers:

- `--api-host` / `--api-port` (default: 127.0.0.1:8000)
- `--gradio-host` / `--gradio-port` (default: 127.0.0.1:7860)
- `--api-only` / `--gradio-only`

## `web_schemas.py`

Pydantic models for FastAPI request and response serialization.

Why this module:

Pydantic models provide validation, automatic OpenAPI schema generation, and clear documentation for API consumers.

### Models

- `AskRequest` — question, top_k, llm
- `AskResponse` — answer, sources, used_llm, results list
- `SearchResultItem` — chunk_id, text, score, source
- `IngestRequest` — source_dir, index_path, chunk_size, chunk_overlap
- `IngestResponse` — document_count, chunk_count, duration_seconds, index_path
- `StatusResponse` — index_exists, index_path, source_dir, document_count, chunk_count, last_modified, source_files

## How To Read The Code

Start with the core pipeline flow:

1. `cli.py` — entry point
2. `pipeline.py` — orchestrates everything
3. `document_loader.py` — file ingestion
4. `chunking.py` — text splitting
5. `embeddings.py` — vector creation
6. `vector_store.py` — storage and search
7. `llm.py` — answer generation
8. `schema.py` — shared data types

Then explore the optional extras:

9. `tracking.py` — experiment tracking
10. `mlflow_model.py` — model packaging
11. `api.py` — REST API
12. `web_schemas.py` — API data models
13. `gradio_app.py` — web UI
14. `serve.py` — combined launcher

