# RAG Quality Notes

Good retrieval is often more important than a larger language model. If the retriever brings irrelevant context, the generator will struggle to produce a reliable answer.

Top-k controls how many chunks are sent to the generator. A small top-k may miss needed evidence. A large top-k may include distracting context and increase cost.

Reranking is a second retrieval step. After the vector search finds candidate chunks, a reranker scores those chunks more carefully and keeps the best ones.

Hybrid search combines keyword search and vector search. Keyword search is strong for exact terms, names, codes, and rare phrases. Vector search is strong for related meaning.

RAG evaluation should include answer correctness, citation quality, retrieval relevance, faithfulness to context, and the system's ability to say it does not know.

