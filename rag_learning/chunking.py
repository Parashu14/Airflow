from __future__ import annotations

import hashlib

from rag_learning.schema import Chunk, Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 180,
    chunk_overlap: int = 40,
) -> list[Chunk]:
    """Split documents into overlapping word chunks.

    Why: retrieval works best when each searchable unit is focused. Overlap
    keeps boundary sentences from being separated too harshly.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    for document in documents:
        words = document.text.split()
        if not words:
            continue

        step = chunk_size - chunk_overlap
        for chunk_index, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                continue

            text = " ".join(chunk_words)
            metadata = {
                **document.metadata,
                "chunk_index": chunk_index,
                "start_word": start,
                "end_word": start + len(chunk_words),
            }
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(text, metadata),
                    text=text,
                    metadata=metadata,
                )
            )

            if start + chunk_size >= len(words):
                break

    return chunks


def _make_chunk_id(text: str, metadata: dict[str, object]) -> str:
    source = str(metadata.get("source", "unknown"))
    page = str(metadata.get("page", ""))
    index = str(metadata.get("chunk_index", ""))
    raw = f"{source}:{page}:{index}:{text[:120]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

