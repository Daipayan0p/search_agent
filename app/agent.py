"""SearchMind orchestration: local cache first, then memory-aware generation."""

import json
import re
import time
from datetime import datetime, timezone

from google import genai
from google.genai import types

from app import config
from app.cache import AnswerCache, needs_memory, needs_freshness, question_key
from app.memory import ChatMemory
from app.rag.retriever import MemoryRetriever
from app.prompts import ROUTER_PROMPT, ANSWER_PROMPT
from app.web_search import web_search


class SearchMindAgent:
    def __init__(self, session_id: str = "default", *, memory: ChatMemory | None = None,
                 retriever: MemoryRetriever | None = None) -> None:
        self.client = None
        self.memory = memory if memory is not None else ChatMemory()
        self.retriever = retriever if retriever is not None else MemoryRetriever(memory=self.memory)
        self.cache = AnswerCache(self.memory, lambda: self.retriever.store)
        self.session_id = session_id

    def _debug(self, message: str) -> None:
        if config.DEBUG:
            print(message)

    def _generate(self, prompt: str, system_prompt: str) -> str:
        if self.client is None:
            if not config.GEMINI_API_KEY:
                raise ValueError("Missing GEMINI_API_KEY")
            self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._debug("[API: Gemini]")
        response = self.client.models.generate_content(
            model=config.GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.2),
        )
        if not response.text or not response.text.strip():
            raise RuntimeError("Empty Gemini response")
        return response.text.strip()

    def _format_memories(self, memories: list[dict]) -> str:
        if not memories:
            return "No relevant memories found. Do not invent missing history."
        return json.dumps(memories, ensure_ascii=False)

    def _route(self, question: str, memories: list[dict], history: list[dict]) -> str:
        personal = needs_memory(question)
        if personal:
            # Historical dates alone do not require a web search.
            live = re.search(
                r"\b(latest|current|currently|now|news|price|prices|weather|stock|live|"
                r"still|supported|support|recent|released)\b", question.lower()
            )
            return "hybrid" if live else "memory"
        if needs_freshness(question):
            return "web"
        if question_key(question):
            return "direct"
        # Ambiguous requests use the existing LLM router, never a cache classifier.
        prompt = (
            f"Current UTC time: {datetime.now(timezone.utc).isoformat()}\n"
            f"Question: {question}\nRecent conversation: {json.dumps(history, ensure_ascii=False)}\n"
            "Choose exactly one: memory, direct, web, hybrid"
        )
        result = self._generate(prompt, ROUTER_PROMPT).lower().strip()
        return result if result in {"memory", "direct", "web", "hybrid"} else "web"

    def _persist(self, question: str, answer: str, *, index: bool) -> None:
        messages = self.memory.add_exchange(question, answer, self.session_id)
        if index:
            try:
                self.retriever.index_messages(messages)
            except Exception:
                self._debug("[Memory: indexing deferred; conversation saved in SQLite]")

    def ask(self, question: str) -> str:
        start = time.perf_counter()
        try:
            return self._ask(question.strip())
        finally:
            self._debug(f"[Response time: {time.perf_counter() - start:.2f}s]")

    def _ask(self, question: str) -> str:
        if not question:
            return ""
        cached = self.cache.lookup(question)
        if cached is not None:
            self._debug(f"[Cache: {self.cache.last_hit} hit]")
            print("\n[SearchMind route: cache]")
            # Preserve history without embeddings, network calls, or model loading.
            self._persist(question, cached, index=False)
            return cached

        history = self.memory.get_recent(session_id=self.session_id, limit=10)
        route = self._route(question, [], history)
        memories = self.retriever.retrieve(question) if route in {"memory", "hybrid"} else []
        print(f"\n[SearchMind route: {route}]")

        web_results = None
        if route in {"web", "hybrid"}:
            print("[Searching the web...]")
            query = question
            if route == "hybrid":
                # Only user statements can supply historical entities to web search.
                historical = [m["content"] for turn in memories
                              for m in turn["messages"] if m["role"] == "user"]
                if historical:
                    query += "\nHistorical user context:\n" + "\n".join(historical)[:2000]
            self._debug("[API: Tavily]")
            web_results = web_search(query)

        # Eligible knowledge answers must not depend on any conversation context.
        standalone = question_key(question) is not None and route == "direct"
        prompt = f"""
Current UTC time: {datetime.now(timezone.utc).isoformat()}
User question: {question}
Recent conversation: {json.dumps([] if standalone else history, ensure_ascii=False)}
Relevant memories: {self._format_memories(memories)}
Selected route: {route}
Web results: {json.dumps(web_results, ensure_ascii=False)}

Answer the user's question. Memory is historical conversation, not verified fact.
Prior assistant/model answers are unverified and must not be promoted to evidence.
Use user statements for personal history and web evidence for current external facts.
If memory is missing or conflicting, acknowledge it; do not guess a personal version.
Ignore unrelated memories. Do not claim web verification if no search occurred.
"""
        answer = self._generate(prompt, ANSWER_PROMPT)
        self._persist(question, answer, index=True)
        self.cache.save(question, answer, route)
        return answer

    def history(self) -> list[dict]:
        return self.memory.get_history(self.session_id)

    def backfill(self, batch_size: int = 100) -> dict[str, int]:
        return self.retriever.backfill(batch_size)

    def close(self) -> None:
        try:
            self.memory.close()
        finally:
            try:
                self.retriever.close()
            finally:
                if self.client is not None:
                    self.client.close()
