import unittest

from rag_learning.chunking import chunk_documents
from rag_learning.schema import Document
from rag_learning.vector_store import VectorStore


class RetrievalTests(unittest.TestCase):
    def test_search_returns_relevant_chunk(self):
        documents = [
            Document(
                text="Gradient descent updates model parameters to reduce loss.",
                metadata={"source": "ml.txt"},
            ),
            Document(
                text="Chunk overlap keeps context around boundaries in RAG.",
                metadata={"source": "rag.txt"},
            ),
        ]
        chunks = chunk_documents(documents, chunk_size=20, chunk_overlap=0)
        store = VectorStore.from_chunks(chunks)

        results = store.search("How does gradient descent reduce loss?", top_k=1)

        self.assertEqual(results[0].chunk.metadata["source"], "ml.txt")


if __name__ == "__main__":
    unittest.main()

