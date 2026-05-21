from __future__ import annotations

from pathlib import Path

from rag_learning.chunking import chunk_documents
from rag_learning.document_loader import load_documents
from rag_learning.llm import answer_question
from rag_learning.schema import RagAnswer
from rag_learning.vector_store import VectorStore

DEFAULT_SOURCE_DIR = Path("data/source_docs")
DEFAULT_INDEX_PATH = Path("data/index/rag_index.json")


def build_index(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    chunk_size: int = 180,
    chunk_overlap: int = 40,
) -> tuple[int, int]:
    """Load documents, chunk them, embed chunks, and save the index."""

    documents = load_documents(source_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    store = VectorStore.from_chunks(chunks)
    store.save(index_path)
    return len(documents), len(chunks)


def ask(
    question: str,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    top_k: int = 4,
    llm: str | None = None,
) -> RagAnswer:
    """Retrieve relevant chunks and answer the user's question."""

    store = VectorStore.load(index_path)
    results = store.search(question, top_k=top_k)
    return answer_question(question, results, llm=llm)

