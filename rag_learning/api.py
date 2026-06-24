from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rag_learning.document_loader import load_documents
from rag_learning.pipeline import DEFAULT_INDEX_PATH, DEFAULT_SOURCE_DIR, ask as pipeline_ask, build_index
from rag_learning.web_schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    SearchResultItem,
    StatusResponse,
)

app = FastAPI(
    title="RAG Pipeline API",
    description="REST API for the educational RAG pipeline",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    source_dir = Path(DEFAULT_SOURCE_DIR)
    index_path = Path(DEFAULT_INDEX_PATH)

    source_files: list[str] = []
    if source_dir.exists():
        source_files = sorted(
            str(p.relative_to(source_dir))
            for p in source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf"}
        )

    index_exists = index_path.exists()
    document_count: int | None = None
    chunk_count: int | None = None
    last_modified: str | None = None

    if index_exists:
        stat = index_path.stat()
        last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        from rag_learning.vector_store import VectorStore

        store = VectorStore.load(index_path)
        chunk_count = len(store.records)
        documents = load_documents(source_dir)
        document_count = len(documents)

    return StatusResponse(
        index_exists=index_exists,
        index_path=str(index_path),
        source_dir=str(source_dir),
        document_count=document_count,
        chunk_count=chunk_count,
        last_modified=last_modified,
        source_files=source_files,
    )


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if not DEFAULT_INDEX_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="No index found. Run 'ingest' first to build the index.",
        )

    result = pipeline_ask(
        question=request.question,
        top_k=request.top_k,
        llm=request.llm,
    )

    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        used_llm=result.used_llm,
        results=[
            SearchResultItem(
                chunk_id=r.chunk.chunk_id,
                text=r.chunk.text,
                score=round(r.score, 4),
                source=_format_source_for_api(r.chunk.metadata),
            )
            for r in result.results
        ],
    )


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    source_dir = Path(request.source_dir) if request.source_dir else DEFAULT_SOURCE_DIR
    index_path = Path(request.index_path) if request.index_path else DEFAULT_INDEX_PATH

    start = time.perf_counter()
    doc_count, chunk_count = build_index(
        source_dir=source_dir,
        index_path=index_path,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )
    elapsed = round(time.perf_counter() - start, 3)

    return IngestResponse(
        document_count=doc_count,
        chunk_count=chunk_count,
        duration_seconds=elapsed,
        index_path=str(index_path),
    )


def _format_source_for_api(metadata: dict) -> str:
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page")
    chunk_index = metadata.get("chunk_index")
    parts = [source]
    if page is not None:
        parts.append(f"page {page}")
    if chunk_index is not None:
        parts.append(f"chunk {chunk_index}")
    return ", ".join(parts)
