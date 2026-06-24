from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from rag_learning.schema import RagAnswer, SearchResult

QUESTION_TERMS = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = frozenset({
    "what", "is", "the", "and", "are", "how", "why", "where", "when",
    "who", "which", "does", "do", "did", "can", "will", "would", "could",
    "should", "has", "have", "had", "been", "its", "was", "were", "may",
    "might", "shall", "need", "dare", "ought", "used", "use", "this",
    "that", "these", "those", "with", "without", "for", "of", "in", "on",
    "to", "from", "by", "at", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "such", "each",
    "every", "both", "all", "some", "any", "no", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "as", "also",
    "if", "then", "else", "over", "under", "get", "got", "let", "put",
    "set", "does", "done", "make", "made", "take", "took",
})


def answer_question(
    question: str,
    results: list[SearchResult],
    llm: str | None = None,
) -> RagAnswer:
    """Generate an answer from retrieved chunks.

    Why: this is the "G" in RAG. The generator should receive only the
    evidence we retrieved, plus clear instructions to stay grounded.
    """

    selected_llm = (llm or os.getenv("RAG_LLM", "extractive")).strip().lower()
    sources = [_format_source(result) for result in results]

    if selected_llm == "ollama":
        try:
            answer = _answer_with_ollama(question, results)
            return RagAnswer(
                answer=answer,
                sources=sources,
                used_llm=f"ollama:{os.getenv('OLLAMA_MODEL', 'llama3.2')}",
                results=results,
            )
        except RuntimeError as exc:
            fallback = _answer_extractively(question, results)
            note = f"Ollama was requested but unavailable: {exc}"
            return RagAnswer(
                answer=f"{fallback}\n\nNote: {note}",
                sources=sources,
                used_llm="extractive-fallback",
                results=results,
            )

    return RagAnswer(
        answer=_answer_extractively(question, results),
        sources=sources,
        used_llm="extractive",
        results=results,
    )


def build_grounded_prompt(question: str, results: list[SearchResult]) -> str:
    """Build the prompt used by a real LLM."""

    context = "\n\n".join(
        f"[{index}] Source: {_format_source(result)}\n{result.chunk.text}"
        for index, result in enumerate(results, start=1)
    )
    return f"""You are a careful RAG assistant.
Answer the question using only the context below.
If the context does not contain the answer, say you do not know.
Mention the most relevant source numbers in your answer.

Context:
{context}

Question:
{question}
"""


def _answer_with_ollama(question: str, results: list[SearchResult]) -> str:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    prompt = build_grounded_prompt(question, results)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You answer only from retrieved context and cite sources.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    request = urllib.request.Request(
        url=f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(str(exc)) from exc

    message = data.get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return str(content).strip()


MIN_TERM_COVERAGE = 0.3


def _significant_terms(text: str) -> set[str]:
    return {
        t.lower()
        for t in QUESTION_TERMS.findall(text)
        if len(t) > 2 and t.lower() not in STOP_WORDS
    }


def _answer_extractively(question: str, results: list[SearchResult]) -> str:
    """Simple local answerer used when no LLM is configured.

    It chooses sentences from retrieved chunks instead of inventing new text.
    This is less fluent than an LLM, but it is very useful for learning because
    you can see exactly how retrieval affects the final answer.
    """

    question_terms = _significant_terms(question)

    if not question_terms:
        return "I do not know from the provided context."

    all_chunk_terms: set[str] = set()
    for result in results:
        all_chunk_terms.update(_significant_terms(result.chunk.text))

    matched = len(question_terms & all_chunk_terms)
    coverage = matched / len(question_terms)
    if coverage < MIN_TERM_COVERAGE:
        return (
            "I do not know from the provided context. "
            f"(Only {matched}/{len(question_terms)} question terms appear in the retrieved content.)"
        )

    scored_sentences: list[tuple[float, str]] = []

    for result in results:
        for sentence in SENTENCE_BOUNDARY.split(result.chunk.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_terms = _significant_terms(sentence)
            overlap = len(question_terms & sentence_terms)
            if overlap > 0:
                scored_sentences.append((overlap + result.score, sentence))

    if not scored_sentences:
        return "I do not know from the provided context."

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence in scored_sentences:
        if sentence in seen:
            continue
        selected.append(sentence)
        seen.add(sentence)
        if len(selected) == 3:
            break

    return " ".join(selected)


def _format_source(result: SearchResult) -> str:
    metadata = result.chunk.metadata
    source = str(metadata.get("source", "unknown"))
    page = metadata.get("page")
    chunk_index = metadata.get("chunk_index")

    parts = [source]
    if page is not None:
        parts.append(f"page {page}")
    if chunk_index is not None:
        parts.append(f"chunk {chunk_index}")
    return ", ".join(parts)

