from __future__ import annotations

from pathlib import Path

from rag_learning.schema import Document

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_documents(source_dir: str | Path) -> list[Document]:
    """Load supported files from a folder.

    Why: RAG starts by turning files into text. We also keep metadata so the
    final answer can cite the file and page it came from.
    """

    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path, root))
        else:
            documents.append(_load_text_file(path, root))

    if not documents:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"No supported files found in {root}. Expected: {supported}")

    return documents


def _load_text_file(path: Path, root: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return Document(
        text=text,
        metadata={
            "source": _relative_source(path, root),
            "file_type": path.suffix.lower().lstrip("."),
        },
    )


def _load_pdf(path: Path, root: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires pypdf. Install it with: pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "source": _relative_source(path, root),
                    "file_type": "pdf",
                    "page": page_number,
                },
            )
        )

    return documents


def _relative_source(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name

