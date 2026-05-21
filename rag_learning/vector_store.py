from __future__ import annotations

import json
from pathlib import Path

from rag_learning.embeddings import HashingEmbedder
from rag_learning.schema import Chunk, SearchResult

INDEX_VERSION = 1


class VectorStore:
    """Small JSON-backed vector store.

    Why: vector databases are just a scalable version of this idea: store
    chunk vectors, embed the query, compare vectors, return the closest chunks.
    """

    def __init__(
        self,
        records: list[dict[str, object]],
        embedder: HashingEmbedder | None = None,
    ) -> None:
        self.records = records
        self.embedder = embedder or HashingEmbedder()

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        embedder: HashingEmbedder | None = None,
    ) -> "VectorStore":
        embedder = embedder or HashingEmbedder()
        vectors = embedder.embed_many([chunk.text for chunk in chunks])
        records = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "vector": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        return cls(records=records, embedder=embedder)

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_vector = self.embedder.embed(query)
        scored: list[tuple[float, dict[str, object]]] = []
        for record in self.records:
            score = _dot_product(query_vector, record["vector"])  # type: ignore[arg-type]
            scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[SearchResult] = []
        for score, record in scored[:top_k]:
            results.append(
                SearchResult(
                    chunk=Chunk(
                        chunk_id=str(record["chunk_id"]),
                        text=str(record["text"]),
                        metadata=dict(record["metadata"]),  # type: ignore[arg-type]
                    ),
                    score=score,
                )
            )
        return results

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "embedding": {
                "name": "hashing",
                "dimensions": self.embedder.dimensions,
            },
            "records": self.records,
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Index file does not exist: {source}")

        payload = json.loads(source.read_text(encoding="utf-8"))
        embedding = payload.get("embedding", {})
        dimensions = int(embedding.get("dimensions", 512))
        return cls(
            records=list(payload["records"]),
            embedder=HashingEmbedder(dimensions=dimensions),
        )


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))

