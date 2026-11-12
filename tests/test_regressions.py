import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qdrant_client import QdrantClient

from app.memory import ChatMemory
from app.rag.vector_store import VectorStore


class FakeEmbedder:
    dimension = 3

    def embed(self, text):
        return [1.0, 0.0, 0.0]


class RegressionTests(unittest.TestCase):
    def test_legacy_history_is_preserved_and_sessions_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "history.db")
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "CREATE TABLE messages (id INTEGER PRIMARY KEY, "
                    "role TEXT, content TEXT, timestamp TEXT)"
                )
                conn.execute(
                    "INSERT INTO messages VALUES (1, 'user', 'old message', '2025-01-01')"
                )
            with patch("app.memory.DATABASE_PATH", path):
                memory = ChatMemory()
                try:
                    legacy = memory.get_history()[0]
                    self.assertTrue(legacy["message_id"])
                    self.assertEqual(legacy["content"], "old message")
                    memory.add_message("user", "new session", "other")
                    self.assertEqual(len(memory.get_history()), 1)
                    self.assertEqual(len(memory.get_recent("other")), 1)
                finally:
                    memory.close()

    def test_vector_round_trip_uses_embedding_dimension(self):
        client = QdrantClient(":memory:")
        with patch("app.rag.vector_store.QdrantClient", return_value=client), patch(
            "app.rag.vector_store.EmbeddingModel", FakeEmbedder
        ):
            store = VectorStore()
            try:
                store.add_memory({
                    "message_id": "12345678-1234-5678-1234-567812345678",
                    "content": "remember this",
                    "role": "user",
                    "timestamp": "2026-01-01",
                    "session_id": "default",
                })
                results = store.search("remember")
                self.assertEqual(results[0]["text"], "remember this")
                self.assertAlmostEqual(results[0]["score"], 1.0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
