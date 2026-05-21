# Data Folder Guide

This folder contains the documents your RAG system searches.

## `source_docs/`

Put your knowledge files here.

Current sample files:

- `rag_course_notes.md`
- `machine_learning_notes.md`
- `rag_quality_notes.md`

Why this folder exists:

RAG needs a knowledge base. The files in this folder are the private or custom knowledge the assistant should use.

Supported file types:

- `.md`
- `.txt`
- `.pdf`

Best beginner use case:

Start with markdown or text notes because they are easy to inspect and debug.

Best real-world use case:

Use PDFs when working with policies, reports, contracts, papers, manuals, or course material. For scanned PDFs, you will need OCR later.

## `index/`

This folder is generated when you run:

```powershell
python -m rag_learning.cli ingest
```

It contains `rag_index.json`, which stores:

- chunk text
- chunk metadata
- chunk vectors

Why this folder is ignored by git:

Indexes are generated artifacts. You can rebuild them from the source documents.

## Learning Exercise

Add a new `.md` file to `source_docs/`, then run:

```powershell
python -m rag_learning.cli ingest
python -m rag_learning.cli ask "Ask something from your new file"
```

Use `--show-context` to see which chunks were retrieved:

```powershell
python -m rag_learning.cli ask "Ask something from your new file" --show-context
```

