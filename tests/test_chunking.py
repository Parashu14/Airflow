import unittest

from rag_learning.chunking import chunk_documents
from rag_learning.schema import Document


class ChunkingTests(unittest.TestCase):
    def test_chunks_overlap(self):
        document = Document(
            text="one two three four five six seven eight nine ten",
            metadata={"source": "test.txt"},
        )

        chunks = chunk_documents([document], chunk_size=5, chunk_overlap=2)

        self.assertEqual(chunks[0].text, "one two three four five")
        self.assertEqual(chunks[1].text, "four five six seven eight")
        self.assertEqual(chunks[0].metadata["source"], "test.txt")

    def test_overlap_must_be_smaller_than_chunk_size(self):
        document = Document(text="one two three", metadata={"source": "test.txt"})

        with self.assertRaises(ValueError):
            chunk_documents([document], chunk_size=3, chunk_overlap=3)


if __name__ == "__main__":
    unittest.main()

