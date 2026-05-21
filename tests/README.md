# Tests Guide

This folder contains small tests for the RAG learning pipeline.

Run tests with:

```powershell
python -m unittest
```

## `test_chunking.py`

Tests the chunking layer.

What it checks:

- chunks overlap correctly
- invalid chunk settings raise errors

Why this matters:

Chunking directly affects retrieval quality. A small bug here can make the retriever miss useful evidence.

## `test_retrieval.py`

Tests the retrieval layer.

What it checks:

- a question about gradient descent retrieves the machine learning chunk

Why this matters:

RAG quality depends heavily on retrieval. Before improving the LLM, first make sure the right context is being found.

## Why These Tests Are Small

This project is meant for learning. The tests focus on the most important behavior:

- Are chunks created predictably?
- Can search return a relevant chunk?

As you upgrade the project, add tests for:

- PDF loading
- metadata citations
- empty document handling
- top-k retrieval behavior
- answer fallback when context is missing
- prompt formatting

