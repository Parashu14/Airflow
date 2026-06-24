from __future__ import annotations

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    top_k: int = 4
    llm: str | None = None


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    used_llm: str
    results: list[SearchResultItem]


class IngestRequest(BaseModel):
    source_dir: str | None = None
    index_path: str | None = None
    chunk_size: int = 180
    chunk_overlap: int = 40


class IngestResponse(BaseModel):
    document_count: int
    chunk_count: int
    duration_seconds: float
    index_path: str


class StatusResponse(BaseModel):
    index_exists: bool
    index_path: str
    source_dir: str
    document_count: int | None = None
    chunk_count: int | None = None
    last_modified: str | None = None
    source_files: list[str] = []
