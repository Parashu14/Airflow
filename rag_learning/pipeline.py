from __future__ import annotations

import time
from pathlib import Path

from rag_learning.chunking import chunk_documents
from rag_learning.document_loader import load_documents
from rag_learning.llm import answer_question
from rag_learning.schema import RagAnswer
from rag_learning.vector_store import VectorStore
from rag_learning import tracking

DEFAULT_SOURCE_DIR = Path("data/source_docs")
DEFAULT_INDEX_PATH = Path("data/index/rag_index.json")


def build_index(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    chunk_size: int = 180,
    chunk_overlap: int = 40,
    mlflow_tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> tuple[int, int]:
    """Load documents, chunk them, embed chunks, and save the index."""

    start_time = time.perf_counter()

    if mlflow_tracking_uri:
        tracking.set_tracking_uri(mlflow_tracking_uri)

    tracking.start_run(
        run_name="build_index",
        experiment_name=experiment_name or "rag_pipeline",
        tags={"pipeline": "build_index"},
    )

    documents = load_documents(source_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    store = VectorStore.from_chunks(chunks)
    store.save(index_path)

    elapsed = time.perf_counter() - start_time

    tracking.log_params({
        "source_dir": str(source_dir),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_dimensions": store.embedder.dimensions,
        "index_path": str(index_path),
    })
    tracking.log_metrics({
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "build_duration_seconds": round(elapsed, 3),
    })
    tracking.log_artifact(index_path)

    if tracking.enabled():
        try:
            from rag_learning.mlflow_model import log_rag_pipeline

            log_rag_pipeline(index_path=index_path)
        except ImportError:
            pass

    tracking.end_run()
    return len(documents), len(chunks)


def ask(
    question: str,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    top_k: int = 4,
    llm: str | None = None,
    mlflow_tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> RagAnswer:
    """Retrieve relevant chunks and answer the user's question."""

    start_time = time.perf_counter()

    if mlflow_tracking_uri:
        tracking.set_tracking_uri(mlflow_tracking_uri)

    tracking.start_run(
        run_name="ask",
        experiment_name=experiment_name or "rag_pipeline",
        tags={"pipeline": "ask"},
        nested=True,
    )

    store = VectorStore.load(index_path)
    results = store.search(question, top_k=top_k)
    answer = answer_question(question, results, llm=llm)

    elapsed = time.perf_counter() - start_time

    top_score = round(results[0].score, 4) if results else 0.0
    mean_score = (
        round(sum(r.score for r in results) / len(results), 4) if results else 0.0
    )

    tracking.log_params({
        "question": question,
        "top_k": top_k,
        "llm": answer.used_llm,
        "index_path": str(index_path),
    })
    tracking.log_metrics({
        "search_duration_seconds": round(elapsed, 3),
        "top_score": top_score,
        "mean_score": mean_score,
        "num_results": len(results),
    })
    tracking.log_text(answer.answer, "answer.txt")

    tracking.end_run()
    return answer

