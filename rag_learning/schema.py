from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    """Plain text plus where it came from."""

    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    """A smaller piece of a document that can be searched."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RagAnswer:
    """Final response returned by the RAG pipeline."""

    answer: str
    sources: list[str]
    used_llm: str
    results: list[SearchResult]

