from __future__ import annotations

import argparse
from pathlib import Path

from rag_learning.pipeline import DEFAULT_INDEX_PATH, DEFAULT_SOURCE_DIR, ask, build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn RAG by running a small pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build a vector index.")
    ingest_parser.add_argument("--source", default=str(DEFAULT_SOURCE_DIR))
    ingest_parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    ingest_parser.add_argument("--chunk-size", type=int, default=180)
    ingest_parser.add_argument("--chunk-overlap", type=int, default=40)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the index.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    ask_parser.add_argument("--top-k", type=int, default=4)
    ask_parser.add_argument("--llm", choices=["extractive", "ollama"], default=None)
    ask_parser.add_argument("--show-context", action="store_true")

    demo_parser = subparsers.add_parser("demo", help="Build the sample index and ask a question.")
    demo_parser.add_argument(
        "--question",
        default="What are the main steps in a RAG pipeline?",
    )
    demo_parser.add_argument("--llm", choices=["extractive", "ollama"], default=None)

    args = parser.parse_args()

    if args.command == "ingest":
        document_count, chunk_count = build_index(
            source_dir=Path(args.source),
            index_path=Path(args.index),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"Indexed {document_count} document parts into {chunk_count} chunks.")
        print(f"Index saved to {args.index}")
        return

    if args.command == "ask":
        answer = ask(
            question=args.question,
            index_path=Path(args.index),
            top_k=args.top_k,
            llm=args.llm,
        )
        _print_answer(answer.answer, answer.sources, answer.used_llm)
        if args.show_context:
            _print_context(answer.results)
        return

    if args.command == "demo":
        document_count, chunk_count = build_index()
        print(f"Demo index built from {document_count} document parts and {chunk_count} chunks.")
        answer = ask(args.question, llm=args.llm)
        _print_answer(answer.answer, answer.sources, answer.used_llm)
        return


def _print_answer(answer: str, sources: list[str], used_llm: str) -> None:
    print("\nAnswer")
    print("------")
    print(answer)
    print(f"\nGenerator: {used_llm}")
    print("\nSources")
    print("-------")
    for source in sources:
        print(f"- {source}")


def _print_context(results) -> None:  # type: ignore[no-untyped-def]
    print("\nRetrieved Context")
    print("-----------------")
    for index, result in enumerate(results, start=1):
        print(f"\n[{index}] score={result.score:.3f}")
        print(result.chunk.text)


if __name__ == "__main__":
    main()

