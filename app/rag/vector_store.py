"""Vector indexes; original messages and canonical answers live in SQLite."""

from qdrant_client import QdrantClient, models

from app.config import QDRANT_URL, QDRANT_COLLECTION, CACHE_COLLECTION, EMBEDDING_MODEL
from app.rag.embeddings import EmbeddingModel


class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL, timeout=5)
        self.collection = QDRANT_COLLECTION
        self._embedder = None
        self._ready: set[str] = set()
        self.index_key = f"{QDRANT_URL}|{self.collection}|{EMBEDDING_MODEL}"

    @property
    def embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel()
        return self._embedder

    def _ensure(self, collection: str) -> None:
        if collection in self._ready:
            return
        # Check connectivity before loading/downloading a model.
        exists = self.client.collection_exists(collection)
        dimension = self.embedder.dimension
        if not exists:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
        else:
            vectors = self.client.get_collection(collection).config.params.vectors
            if not isinstance(vectors, models.VectorParams) or vectors.size != dimension:
                raise ValueError("Embedding dimension mismatch; use a new Qdrant collection")
        self._ready.add(collection)

    def close(self) -> None:
        self.client.close()

    def add_memory(self, message: dict) -> None:
        self._ensure(self.collection)
        self.client.upsert(
            collection_name=self.collection, wait=True,
            points=[models.PointStruct(
                id=message["message_id"], vector=self.embedder.embed(message["content"]),
                payload={"message_id": message["message_id"],
                         "turn_id": message.get("turn_id"), "text": message["content"],
                         "role": message["role"], "timestamp": message["timestamp"],
                         "session_id": message["session_id"], "trusted": False},
            )],
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        self._ensure(self.collection)
        response = self.client.query_points(
            collection_name=self.collection, query=self.embedder.embed(query),
            limit=limit, with_payload=True,
        )
        return [{**(point.payload or {}), "message_id": str(point.id), "score": point.score}
                for point in response.points]

    def add_cache(self, record: dict) -> None:
        self._ensure(CACHE_COLLECTION)
        self.client.upsert(
            collection_name=CACHE_COLLECTION, wait=True,
            points=[models.PointStruct(
                id=record["cache_id"], vector=self.embedder.embed(record["question"]),
                payload={"cache_id": record["cache_id"], "namespace": record["namespace"]},
            )],
        )

    def search_cache(self, question: str) -> list[dict]:
        self._ensure(CACHE_COLLECTION)
        response = self.client.query_points(
            collection_name=CACHE_COLLECTION, query=self.embedder.embed(question),
            limit=5, with_payload=True,
        )
        return [{"cache_id": str(point.id), "score": point.score} for point in response.points]
