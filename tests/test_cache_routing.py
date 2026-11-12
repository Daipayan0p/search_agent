"""Offline contract tests: no real LLM, web search, downloads, or Qdrant server."""

from unittest.mock import Mock, patch

import pytest
from qdrant_client import QdrantClient

from app import config
from app.agent import SearchMindAgent
from app.cache import AnswerCache, question_key
from app.memory import ChatMemory
from app.rag.retriever import MemoryRetriever
from app.rag.vector_store import VectorStore


class FakeEmbedder:
    dimension = 3

    def embed(self, text: str) -> list[float]:
        # Deliberately identical vectors: the policy must guard even score=1 hits.
        return [1.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def no_external_apis():
    with patch("app.agent.genai.Client", side_effect=AssertionError("Unexpected Gemini call")), patch(
        "app.agent.web_search", side_effect=AssertionError("Unexpected Tavily call")
    ), patch("app.rag.embeddings._load_model", side_effect=AssertionError("Unexpected model download")):
        yield


@pytest.fixture
def memory(tmp_path):
    value = ChatMemory(str(tmp_path / "memory.db"))
    yield value
    value.close()


@pytest.fixture
def store():
    client = QdrantClient(":memory:")
    with patch("app.rag.vector_store.QdrantClient", return_value=client), patch(
        "app.rag.vector_store.EmbeddingModel", FakeEmbedder
    ):
        value = VectorStore()
        yield value
        value.close()


@pytest.fixture
def agent(memory, store):
    retriever = MemoryRetriever(memory=memory)
    retriever._store = store
    value = SearchMindAgent(memory=memory, retriever=retriever)
    value._generate = Mock(return_value="Python is a high-level programming language.")
    return value


def test_exact_hit_has_no_gemini_tavily_embedding_or_vector_calls(agent, capsys):
    agent.ask("Is Python a high-level language?")
    agent._generate.reset_mock()
    with patch.object(agent.retriever, "retrieve", side_effect=AssertionError), patch.object(
        agent.retriever.store, "search_cache", side_effect=AssertionError
    ), patch.object(agent.retriever.store, "add_memory", side_effect=AssertionError), patch.object(
        agent.retriever.store.embedder, "embed", side_effect=AssertionError
    ):
        assert agent.ask("  IS  Python a HIGH-level   language? ") == "Python is a high-level programming language."
    agent._generate.assert_not_called()
    assert len(agent.history()) == 4
    assert "[SearchMind route: cache]" in capsys.readouterr().out


def test_exact_miss_generates_one_answer_without_router_call(agent, capsys):
    assert agent.cache.lookup("What is Python?") is None
    agent.ask("What is Python?")
    assert agent._generate.call_count == 1
    assert "[SearchMind route: direct]" in capsys.readouterr().out


def test_safe_semantic_hit_requires_same_template_subject(agent):
    agent.ask("What is Python?")
    agent._generate.reset_mock()
    assert agent.ask("Define Python") == "Python is a high-level programming language."
    assert agent.cache.last_hit == "semantic"
    agent._generate.assert_not_called()


def test_semantic_hit_creates_exact_alias_without_duplicating_canonical_record(agent):
    agent.ask("What is Python?")
    agent.ask("Define Python")
    assert agent.cache.last_hit == "semantic"
    with patch.object(agent.retriever.store, "search_cache", side_effect=AssertionError):
        agent.ask("Define Python")
        assert agent.cache.last_hit == "exact"
    assert agent.memory.conn.execute("SELECT count(*) FROM answer_cache").fetchone()[0] == 1
    agent.cache.invalidate()
    assert agent.memory.conn.execute("SELECT count(*) FROM answer_cache_aliases").fetchone()[0] == 0
    assert agent.cache.lookup("Define Python") is None


def test_semantic_threshold_is_enforced_even_for_equivalent_questions(agent):
    agent.ask("What is Python?")
    record = agent.memory.cache_exact("what is python?", agent.cache.namespace)
    with patch.object(agent.retriever.store, "search_cache", return_value=[
        {"cache_id": record["cache_id"], "score": config.CACHE_SIMILARITY_THRESHOLD - 0.01}
    ]):
        assert agent.cache.lookup("Define Python") is None


def test_disabled_cache_never_reads_or_writes_vectors(agent, monkeypatch):
    monkeypatch.setattr(config, "CACHE_ENABLED", False)
    with patch.object(agent.retriever.store, "search_cache", side_effect=AssertionError), patch.object(
        agent.retriever.store, "add_cache", side_effect=AssertionError
    ):
        agent.cache.save("What is Python?", "Python is a programming language.", "direct")
        assert agent.cache.lookup("What is Python?") is None


@pytest.mark.parametrize("question", [
    "What is a list in Python?", "Is Python a compiled language?",
    "Is Python an interpreted language?", "What is photosynthesis?",
    "Is Python not a high-level language?", "What is Python used for?",
])
def test_similar_but_different_questions_never_reuse_answer(agent, question):
    agent.cache.save("Is Python a high-level language?", "Yes, Python is high-level.", "direct")
    assert agent.cache.lookup(question) is None


def test_expiration_and_ttl_reduction(memory, store, monkeypatch):
    now = [100.0]
    cache = AnswerCache(memory, lambda: store, clock=lambda: now[0])
    monkeypatch.setattr(config, "CACHE_TTL_SECONDS", 60)
    cache.save("What is Python?", "Python is a programming language.", "direct")
    assert cache.lookup("What is Python?")
    now[0] = 161.0
    assert cache.lookup("What is Python?") is None
    now[0] = 110.0
    monkeypatch.setattr(config, "CACHE_TTL_SECONDS", 5)
    assert cache.lookup("Define Python") is None


@pytest.mark.parametrize("question", [
    "What is the latest Python version?", "What is Python today?", "Weather tomorrow?",
    "What happened yesterday?", "Python stock price", "Current news",
])
def test_time_sensitive_queries_bypass_all_caches(agent, question):
    with patch.object(agent.retriever.store, "search_cache", side_effect=AssertionError):
        assert question_key(question) is None
        assert agent.cache.lookup(question) is None


@pytest.mark.parametrize("question", [
    "What about it?", "Explain that", "What was my project?", "What did I ask before?",
    "What is my Python password?", "What is Python in my project?",
])
def test_context_dependent_queries_bypass_cache(agent, question):
    assert question_key(question) is None
    assert agent.cache.lookup(question) is None


def test_retrieval_across_sessions_hydrates_user_and_assistant(memory, store):
    messages = memory.add_exchange("My project uses Kubernetes 1.25.",
                                   "You mentioned Kubernetes 1.25.", "old-session")
    retriever = MemoryRetriever(memory=memory)
    retriever._store = store
    retriever.index_messages(messages)
    results = retriever.retrieve("What Kubernetes version did I previously use?")
    assert len(results) == 1  # Both matching points collapse to one original turn.
    assert results[0]["session_id"] == "old-session"
    assert {m["role"] for m in results[0]["messages"]} == {"user", "model"}
    assert results[0]["messages"][0]["message_id"] == messages[0]["message_id"]
    assert results[0]["timestamp"] == messages[0]["timestamp"]


def test_unrelated_high_score_and_orphaned_points_ignored(memory, store):
    messages = memory.add_exchange("My project uses Python.", "Python is useful.", "old")
    retriever = MemoryRetriever(memory=memory)
    retriever._store = store
    retriever.index_messages(messages)
    assert retriever.retrieve("Which bicycle did I buy?") == []
    memory.conn.execute("DELETE FROM messages")
    memory.conn.commit()
    assert retriever.retrieve("My Python project") == []


def test_hybrid_uses_original_user_version_in_tavily_query(agent, capsys):
    messages = agent.memory.add_exchange("My project uses Kubernetes 1.25.",
                                         "Unverified claim: Kubernetes 9.99.", "old")
    agent.retriever.index_messages(messages)
    with patch("app.agent.web_search", return_value={"results": [{"url": "https://example.test"}]}) as web:
        agent.ask("Is the Kubernetes version I previously used still supported?")
    query = web.call_args.args[0]
    assert "1.25" in query
    assert "9.99" not in query
    assert "[SearchMind route: hybrid]" in capsys.readouterr().out
    assert agent._generate.call_count == 1
    assert agent.memory.conn.execute("SELECT count(*) FROM answer_cache").fetchone()[0] == 0


def test_historical_relative_date_routes_to_memory_not_web(agent, capsys):
    agent.ask("What project was I building yesterday?")
    assert "[SearchMind route: memory]" in capsys.readouterr().out


def test_latest_version_routes_web(agent, capsys):
    with patch("app.agent.web_search", return_value={"results": []}) as web:
        agent.ask("What is the latest Python version?")
        web.assert_called_once()
    assert "[SearchMind route: web]" in capsys.readouterr().out
    assert agent.cache.lookup("What is the latest Python version?") is None


@pytest.mark.parametrize("question", [
    "Tell me the latest Python version", "I want to know the latest Python version",
])
def test_courtesy_language_does_not_trigger_personal_memory_search(agent, question):
    with patch.object(agent.retriever, "retrieve", side_effect=AssertionError), patch(
        "app.agent.web_search", return_value={"results": []}
    ):
        agent.ask(question)
    assert agent._generate.call_count == 1


def test_qdrant_failure_does_not_lose_sqlite_turns(agent):
    with patch.object(agent.retriever.store, "search", side_effect=ConnectionError("secret-error")), patch.object(
        agent.retriever.store, "add_memory", side_effect=ConnectionError("secret-error")
    ):
        agent.ask("What Python project was I building yesterday?")
        assert len(agent.history()) == 2
        assert len(agent.memory.pending_messages(agent.retriever.store.index_key)) == 2
        assert agent.retriever.retrieve("My Python project") == []


def test_cache_qdrant_failure_keeps_exact_answer(agent):
    with patch.object(agent.retriever.store, "add_cache", side_effect=ConnectionError):
        agent.cache.save("What is Python?", "Python is a programming language.", "direct")
    with patch.object(agent.retriever.store, "search_cache", side_effect=ConnectionError):
        assert agent.cache.lookup("Define Python") is None
        assert agent.cache.lookup("What is Python?") == "Python is a programming language."


def test_sqlite_restart_preserves_history_and_exact_cache(tmp_path):
    path = str(tmp_path / "restart.db")
    unavailable = Mock(side_effect=AssertionError("Must not load vectors"))
    first = ChatMemory(path)
    cache = AnswerCache(first, unavailable)
    first.add_exchange("What is Python?", "Python is a programming language.", "session")
    cache_record = first.save_cache("what is python?", "Python is a programming language.",
                                    "definition:python", cache.namespace, cache.clock(), 600)
    first.close()
    second = ChatMemory(path)
    try:
        assert len(second.get_history("session")) == 2
        assert second.cache_record(cache_record["cache_id"])
        assert AnswerCache(second, unavailable).lookup(" WHAT IS   PYTHON? ")
    finally:
        second.close()


def test_startup_and_exact_hit_never_construct_vector_store(memory):
    unavailable = Mock(side_effect=AssertionError("Must not initialize Qdrant"))
    cache = AnswerCache(memory, unavailable)
    memory.save_cache("what is python?", "Python is a programming language.",
                      "definition:python", cache.namespace, cache.clock(), 600)
    with patch("app.rag.retriever.VectorStore", unavailable):
        agent = SearchMindAgent(memory=memory)
        assert agent.ask("What is Python?") == "Python is a programming language."
    unavailable.assert_not_called()


def test_qdrant_initialization_failure_is_recoverable(memory):
    with patch("app.rag.retriever.VectorStore", side_effect=ConnectionError):
        agent = SearchMindAgent(memory=memory)
        agent._generate = Mock(return_value="No relevant history is available.")
        agent.ask("What was my Python project?")
        assert len(agent.history()) == 2


def test_invalidation_rejects_stale_semantic_vectors(agent):
    agent.ask("What is Python?")
    record = agent.memory.cache_exact("what is python?", agent.cache.namespace)
    agent.cache.invalidate(record["cache_id"])
    assert agent.cache.lookup("What is Python?") is None
    assert agent.cache.lookup("Define Python") is None


def test_namespace_change_invalidates_both_paths(agent, monkeypatch):
    agent.ask("What is Python?")
    monkeypatch.setattr(config, "CACHE_VERSION", "different")
    new_cache = AnswerCache(agent.memory, lambda: agent.retriever.store)
    assert new_cache.lookup("What is Python?") is None
    assert new_cache.lookup("Define Python") is None


def test_backfill_is_idempotent_and_retries_partial_failure(memory, store):
    messages = memory.add_exchange("My project is Python.", "An unverified assistant answer.", "old")
    retriever = MemoryRetriever(memory=memory)
    retriever._store = store
    original = store.add_memory
    calls = [0]

    def fail_second(message):
        calls[0] += 1
        if calls[0] == 2:
            raise ConnectionError
        original(message)

    with patch.object(store, "add_memory", side_effect=fail_second):
        assert retriever.backfill() == {"indexed": 1, "failed": 1, "skipped": 0}
    assert len(memory.get_history("old")) == 2
    assert retriever.backfill()["indexed"] == 1
    assert retriever.backfill()["indexed"] == 0
    assert store.client.count(store.collection).count == len(messages)


@pytest.mark.parametrize("answer", [
    "I cannot verify this with insufficient evidence.", "Your API key is secret.",
    "Email alice@example.com.", "Password: abcdefg", "Maybe; I don't know.",
    "Bearer abcdef12345", "This is the latest release.", "My name is Alice.",
    "I don't have enough information to answer.", "I am unable to verify that.",
])
def test_sensitive_or_insufficient_answers_not_cached(agent, answer):
    agent.cache.save("What is Python?", answer, "direct")
    assert agent.memory.conn.execute("SELECT count(*) FROM answer_cache").fetchone()[0] == 0


def test_context_is_omitted_from_cacheable_generation(agent):
    agent.memory.add_exchange("My private project context", "old assistant context", "default")
    agent.ask("What is Python?")
    prompt = agent._generate.call_args.args[0]
    assert "My private project context" not in prompt
    assert "old assistant context" not in prompt


def test_debug_logs_never_include_question_answer_or_exception(agent, monkeypatch, capsys):
    monkeypatch.setattr(config, "DEBUG", True)
    with patch.object(agent.retriever.store, "add_memory", side_effect=ConnectionError("hidden-secret")):
        agent.ask("What is Python?")
    output = capsys.readouterr().out
    assert "hidden-secret" not in output
    assert "What is Python" not in output
    assert "Python is a high-level" not in output
    assert "[Response time:" in output


def test_follow_up_uses_recent_history_and_never_caches(agent):
    agent.memory.add_exchange("Tell me about Python", "It is a language.", "default")
    agent._generate.side_effect = ["direct", "More details about Python."]
    agent.ask("What about it?")
    assert "Tell me about Python" in agent._generate.call_args.args[0]
    assert agent.memory.conn.execute("SELECT count(*) FROM answer_cache").fetchone()[0] == 0


def test_legacy_pairing_preserves_ids_and_survives_repeated_migration(tmp_path):
    import sqlite3

    path = str(tmp_path / "legacy.db")
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, message_id TEXT, "
                     "session_id TEXT, role TEXT, content TEXT, timestamp TEXT)")
        conn.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)", [
            (1, "11111111-1111-1111-1111-111111111111", "a", "user", "Python project", "2025-01-01"),
            (2, "22222222-2222-2222-2222-222222222222", "b", "user", "Other project", "2025-01-02"),
            (3, "33333333-3333-3333-3333-333333333333", "a", "model", "Answer A", "2025-01-03"),
            (4, "44444444-4444-4444-4444-444444444444", "b", "model", "Answer B", "2025-01-04"),
        ])
    first = ChatMemory(path)
    original = first.get_turn("11111111-1111-1111-1111-111111111111")
    assert [m["content"] for m in original] == ["Python project", "Answer A"]
    assert {m["session_id"] for m in original} == {"a"}
    first.close()
    second = ChatMemory(path)
    try:
        assert second.get_turn("11111111-1111-1111-1111-111111111111") == original
        assert len(second.get_history("b")) == 2
    finally:
        second.close()
