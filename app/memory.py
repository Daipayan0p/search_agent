
import sqlite3
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

from app.config import DATABASE_PATH


class ChatMemory:
    def __init__(self, database_path: str | None = None) -> None:
        database_path = database_path or DATABASE_PATH
        Path(database_path).parent.mkdir(
            parents=True, exist_ok=True
        )

        self.conn = sqlite3.connect(database_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE,
                session_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        # Support databases from the earlier version.
        columns = {
            row["name"]
            for row in self.conn.execute(
                "PRAGMA table_info(messages)"
            )
        }

        if "message_id" not in columns:
            self.conn.execute(
                "ALTER TABLE messages ADD COLUMN message_id TEXT"
            )

        if "session_id" not in columns:
            self.conn.execute(
                "ALTER TABLE messages ADD COLUMN session_id TEXT"
            )

        if "turn_id" not in columns:
            self.conn.execute("ALTER TABLE messages ADD COLUMN turn_id TEXT")

        self.conn.execute(
            "UPDATE messages SET session_id = 'default' WHERE session_id IS NULL"
        )
        for row in self.conn.execute(
            "SELECT id FROM messages WHERE message_id IS NULL"
        ).fetchall():
            self.conn.execute(
                "UPDATE messages SET message_id = ? WHERE id = ?",
                (str(uuid4()), row["id"])
            )

        # Pair legacy messages only within their original session and ordering.
        turns: dict[str, str] = {}
        for row in self.conn.execute("SELECT * FROM messages ORDER BY id").fetchall():
            session = row["session_id"]
            if row["turn_id"]:
                turns[session] = row["turn_id"]
                if row["role"] in ("model", "assistant"):
                    turns.pop(session, None)
                continue
            if row["role"] == "user" or session not in turns:
                turns[session] = row["message_id"]
            self.conn.execute("UPDATE messages SET turn_id = ? WHERE id = ?",
                              (turns[session], row["id"]))
            if row["role"] in ("model", "assistant"):
                turns.pop(session, None)

        self.conn.executescript("""
            CREATE INDEX IF NOT EXISTS messages_turn ON messages(turn_id);
            CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
            CREATE TABLE IF NOT EXISTS answer_cache (
                cache_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                eligibility_key TEXT NOT NULL,
                namespace TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                UNIQUE(question, namespace)
            );
            CREATE TABLE IF NOT EXISTS memory_index (
                message_id TEXT NOT NULL,
                index_key TEXT NOT NULL,
                PRIMARY KEY(message_id, index_key)
            );
            CREATE TABLE IF NOT EXISTS answer_cache_aliases (
                question TEXT NOT NULL,
                namespace TEXT NOT NULL,
                cache_id TEXT NOT NULL REFERENCES answer_cache(cache_id) ON DELETE CASCADE,
                PRIMARY KEY(question, namespace)
            );
            CREATE INDEX IF NOT EXISTS cache_expiry ON answer_cache(namespace, expires_at);
        """)

        self.conn.commit()

    def add_message(
        self, role: str, content: str, session_id: str = "default"
    ) -> dict:
        message_id = str(uuid4())
        timestamp = datetime.now(
            timezone.utc
        ).isoformat()
        turn_id = message_id
        if role in ("model", "assistant"):
            previous = self.conn.execute(
                "SELECT role, turn_id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if previous and previous["role"] == "user":
                turn_id = previous["turn_id"]

        self.conn.execute(
            """
            INSERT INTO messages
            (message_id, session_id, role, content, timestamp, turn_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                message_id, session_id,
                role, content, timestamp, turn_id
            )
        )
        self.conn.commit()

        return {
            "message_id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": timestamp,
            "turn_id": turn_id
        }

    def get_recent(self, session_id: str = "default", limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit)
        ).fetchall()

        return [
            dict(row) for row in reversed(rows)
        ]

    def get_history(self, session_id: str = "default") -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY id
            """,
            (session_id,)
        ).fetchall()

        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()

    def add_exchange(self, question: str, answer: str, session_id: str) -> list[dict]:
        """Commit the whole turn before attempting any external indexing."""
        turn_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        messages = [dict(message_id=str(uuid4()), turn_id=turn_id,
                         session_id=session_id, role=role, content=content,
                         timestamp=timestamp)
                    for role, content in (("user", question), ("model", answer))]
        with self.conn:
            self.conn.executemany(
                "INSERT INTO messages (message_id, turn_id, session_id, role, content, timestamp) "
                "VALUES (:message_id, :turn_id, :session_id, :role, :content, :timestamp)",
                messages,
            )
        return messages

    def get_turn(self, message_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE turn_id = "
            "(SELECT turn_id FROM messages WHERE message_id = ?) ORDER BY id",
            (message_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def pending_messages(self, index_key: str, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT m.* FROM messages m WHERE NOT EXISTS "
            "(SELECT 1 FROM memory_index i WHERE i.message_id = m.message_id "
            "AND i.index_key = ?) ORDER BY m.id LIMIT ?", (index_key, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_indexed(self, message_id: str, index_key: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO memory_index VALUES (?, ?)",
                              (message_id, index_key))

    def cache_exact(self, question: str, namespace: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM answer_cache WHERE question = ? AND namespace = ?",
            (question, namespace),
        ).fetchone()
        if row is None:
            row = self.conn.execute(
                "SELECT c.* FROM answer_cache c JOIN answer_cache_aliases a "
                "ON a.cache_id = c.cache_id WHERE a.question = ? AND a.namespace = ?",
                (question, namespace),
            ).fetchone()
        return dict(row) if row else None

    def cache_alias(self, question: str, namespace: str, cache_id: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO answer_cache_aliases VALUES (?, ?, ?) "
                "ON CONFLICT(question, namespace) DO UPDATE SET cache_id=excluded.cache_id",
                (question, namespace, cache_id),
            )

    def has_live_cache(self, namespace: str, now: float) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM answer_cache WHERE namespace = ? AND expires_at > ? LIMIT 1",
            (namespace, now),
        ).fetchone() is not None

    def cache_record(self, cache_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM answer_cache WHERE cache_id = ?",
                                (cache_id,)).fetchone()
        return dict(row) if row else None

    def save_cache(self, question: str, answer: str, key: str, namespace: str,
                   now: float, ttl: int) -> dict:
        with self.conn:
            self.conn.execute(
                "INSERT INTO answer_cache VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(question, namespace) DO UPDATE SET "
                "answer=excluded.answer, eligibility_key=excluded.eligibility_key, "
                "created_at=excluded.created_at, expires_at=excluded.expires_at",
                (str(uuid4()), question, answer, key, namespace, now, now + ttl),
            )
        record = self.cache_exact(question, namespace)
        assert record is not None
        return record

    def invalidate_cache(self, cache_id: str | None = None) -> None:
        with self.conn:
            if cache_id is None:
                self.conn.execute("DELETE FROM answer_cache")
            else:
                self.conn.execute("DELETE FROM answer_cache WHERE cache_id = ?", (cache_id,))
