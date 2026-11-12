import re

from app.cache import is_sensitive
from app import config
from app.memory import ChatMemory
from app.rag.vector_store import VectorStore


def anchors(text: str) -> set[str]:
    stop = {"what", "which", "where", "when", "who", "was", "were", "have", "has",
            "had", "the", "and", "for", "from", "that", "this", "with", "you", "your",
            "used", "using", "building", "previously", "previous", "yesterday", "today",
            "still", "supported", "support", "mentioned", "remember", "about", "did"}
    return {word for word in re.findall(r"[a-z0-9][a-z0-9.+-]*", text.lower())
            if len(word) > 2 and word not in stop}


class MemoryRetriever:
    def __init__(self, min_score: float = 0.60, memory: ChatMemory | None = None) -> None:
        self._store: VectorStore | None = None
        self.min_score = min_score
        self.memory = memory

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore()
        return self._store

    def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        try:
            results = self.store.search(query=query, limit=limit * 3)
        except Exception:
            if config.DEBUG:
                print("[Memory: vector retrieval unavailable]")
            return []
        memories, seen = [], set()
        for result in results:
            if result["score"] < self.min_score:
                continue
            # Hydrate from SQLite; deleted/orphaned/stale vector payloads are ignored.
            turn = self.memory.get_turn(result["message_id"]) if self.memory else []
            if not turn:
                continue
            user_text = " ".join(m["content"] for m in turn if m["role"] == "user")
            if not anchors(query).intersection(anchors(user_text)):
                continue
            if any(is_sensitive(m["content"]) for m in turn):
                continue
            turn_id = turn[0]["turn_id"]
            if turn_id in seen:
                continue
            seen.add(turn_id)
            memories.append({"message_id": result["message_id"], "turn_id": turn_id,
                             "timestamp": turn[0]["timestamp"], "session_id": turn[0]["session_id"],
                             "score": result["score"], "messages": turn,
                             "text": "\n".join(f"{m['role']}: {m['content']}" for m in turn)})
            if len(memories) == limit:
                break
        if config.DEBUG:
            print(f"[Memory: {len(memories)} relevant results]")
            for result in memories:
                print(f"[Memory score: {result['score']:.3f}]")
        return memories

    def index_messages(self, messages: list[dict]) -> None:
        for message in messages:
            if is_sensitive(message["content"]):
                continue
            self.store.add_memory(message)
            if self.memory:
                self.memory.mark_indexed(message["message_id"], self.store.index_key)

    def backfill(self, batch_size: int = 100) -> dict[str, int]:
        """Retry durable pending rows. Stable point UUIDs make crash retries idempotent."""
        counts = {"indexed": 0, "skipped": 0, "failed": 0}
        if not self.memory:
            return counts
        store = self.store
        messages = self.memory.pending_messages(store.index_key, batch_size)
        for message in messages:
            try:
                if is_sensitive(message["content"]):
                    self.memory.mark_indexed(message["message_id"], store.index_key)
                    counts["skipped"] += 1
                    continue
                self.index_messages([message])
                counts["indexed"] += 1
            except Exception:
                counts["failed"] += 1
                break
        return counts

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
