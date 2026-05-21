# RAG Course Notes

Retrieval-Augmented Generation, or RAG, is a pattern for answering questions with private or changing knowledge. The system first retrieves relevant passages from documents, then gives those passages to a language model so the answer can be grounded in evidence.

A basic RAG pipeline has six steps: load documents, split documents into chunks, create embeddings for each chunk, store those vectors in a vector index, retrieve relevant chunks for a user question, and generate an answer using the retrieved context.

Chunking matters because large documents usually contain many topics. Smaller chunks make retrieval more focused. Chunk overlap keeps nearby context together when an idea crosses a chunk boundary.

Embeddings turn text into numeric vectors. During retrieval, the question is embedded with the same method as the chunks. The system compares the question vector to stored chunk vectors and returns the closest matches.

Grounded prompting reduces hallucination. A good RAG prompt tells the model to answer only from the retrieved context and to say it does not know when the answer is missing.

Metadata is important in RAG systems. Each chunk should remember its source file, page number when available, and chunk number. This allows the final answer to include citations and helps users verify the response.

