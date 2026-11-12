"""Conservative cache policy and SQLite-authoritative answer caching.

Semantic similarity only proposes candidates. A closed set of equivalent question
templates must also agree, including the entire subject. Unknown wording is a miss.
"""

import hashlib
import re
import time
from typing import TYPE_CHECKING, Callable

from app import config

if TYPE_CHECKING:
    from app.memory import ChatMemory
    from app.rag.vector_store import VectorStore


def normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def is_sensitive(text: str) -> bool:
    return bool(re.search(
        r"(?i)(api[ _-]?key|password|passwd|secret|credential|access[ _-]?token|"
        r"bearer\s+\S+|private[ _-]?key|AIza[\w-]{20,}|sk-[\w-]{16,}|"
        r"AKIA[A-Z0-9]{16}|[\w.+-]+@[\w.-]+\.[a-z]{2,}|"
        r"\b\d(?:[ -]?\d){9,18}\b|eyJ[\w-]+\.[\w-]+\.[\w-]+|"
        r"https?://[^\s/]+:[^\s@]+@|"
        r"-----BEGIN .*PRIVATE KEY-----)", text
    ))


def needs_memory(question: str) -> bool:
    text = normalize_question(question)
    text = re.sub(r"\b(?:tell|show|give|explain to) me\b", "", text)
    text = re.sub(r"\bi (?:want|would like|need) to (?:know|learn|understand)\b", "", text)
    return bool(re.search(
        r"\b(i|me|my|mine|we|our|ours|conversations?|"
        r"remember|mentioned|told|discussed|earlier|last time)\b",
        text,
    ))


def needs_freshness(question: str) -> bool:
    return bool(re.search(
        r"\b(today|yesterday|tomorrow|latest|current|currently|now|recent|"
        r"news|price|prices|weather|stock|stocks|live|supported|support|"
        r"still|release|released|version|versions|this week|this month|this year)\b",
        normalize_question(question),
    ))


# Intentionally narrow: additions require stable subjects and equivalence tests.
SUBJECTS = {
    "python", "the python programming language", "a list in python",
    "a tuple in python", "a dictionary in python", "a python list",
    "a python tuple", "a python dictionary", "recursion", "a binary search tree",
    "binary search", "a prime number", "photosynthesis", "gravity",
    "the pythagorean theorem", "an algorithm", "a programming language",
}


def question_key(question: str) -> str | None:
    """Return a proven template key, never a bag-of-words equivalence guess."""
    text = normalize_question(question).removesuffix("?").removesuffix(".")
    if is_sensitive(text) or needs_memory(text) or needs_freshness(text):
        return None
    definition = re.fullmatch(r"(?:what is|define|explain) (.+)", text)
    if definition and definition[1] in SUBJECTS:
        return "definition:" + definition[1]
    # Keep predicates separate: 'compiled' and 'interpreted' cannot share answers.
    predicate = re.fullmatch(
        r"is python (a high[- ]level (?:programming )?language|"
        r"an interpreted (?:programming )?language|a compiled (?:programming )?language)",
        text,
    )
    if predicate:
        return "python:" + predicate[1].replace("high-level", "high level").replace("programming ", "")
    return None


def safe_answer(answer: str) -> bool:
    if (not answer.strip() or is_sensitive(answer) or needs_freshness(answer)
            or re.search(r"(?i)\b(my|your|our|ours|mine)\b", answer)):
        return False
    return not re.search(
        r"(?i)(not sure|uncertain|cannot (?:verify|confirm|determine)|"
        r"can't (?:verify|confirm|determine)|insufficient|not enough (?:information|evidence)|"
        r"(?:don't|do not) have (?:enough |sufficient |access to )?(?:information|evidence|context)|"
        r"unable to (?:verify|confirm|determine)|no (?:reliable |relevant )?(?:evidence|context)|"
        r"don't know|do not know|may be|might be|maybe|i think|i believe|unverified|"
        r"as of|currently|today|https?://)",
        answer,
    )


class AnswerCache:
    def __init__(self, memory: "ChatMemory", store: Callable[[], "VectorStore"],
                 clock: Callable[[], float] = time.time) -> None:
        self.memory = memory
        self.store = store
        self.clock = clock
        self.namespace = hashlib.sha256(
            f"{config.CACHE_VERSION}:{config.GEMINI_MODEL}:{config.EMBEDDING_MODEL}".encode()
        ).hexdigest()
        self.last_hit: str | None = None

    def _valid(self, record: dict | None, key: str) -> bool:
        return bool(record and record["namespace"] == self.namespace
                    and record["expires_at"] > self.clock()
                    and record["created_at"] + config.CACHE_TTL_SECONDS > self.clock()
                    and record["eligibility_key"] == key
                    and question_key(record["question"]) == key
                    and safe_answer(record["answer"]))

    def lookup(self, question: str) -> str | None:
        self.last_hit = None
        key = question_key(question)
        if not config.CACHE_ENABLED or not key:
            return None
        record = self.memory.cache_exact(normalize_question(question), self.namespace)
        if self._valid(record, key):
            self.last_hit = "exact"
            return record["answer"]
        if not self.memory.has_live_cache(self.namespace, self.clock()):
            return None
        try:
            candidates = self.store().search_cache(question)
            for candidate in candidates:
                if candidate["score"] < config.CACHE_SIMILARITY_THRESHOLD:
                    continue
                # Never trust the Qdrant payload as the answer or validity authority.
                record = self.memory.cache_record(candidate["cache_id"])
                if self._valid(record, key):
                    self.memory.cache_alias(normalize_question(question), self.namespace,
                                            record["cache_id"])
                    self.last_hit = "semantic"
                    return record["answer"]
        except Exception:
            # Cache failures must never block answering. Do not log payloads/errors.
            if config.DEBUG:
                print("[Cache: semantic lookup unavailable]")
        return None

    def save(self, question: str, answer: str, route: str) -> None:
        key = question_key(question)
        if not config.CACHE_ENABLED or route != "direct" or not key or not safe_answer(answer):
            return
        record = self.memory.save_cache(
            normalize_question(question), answer, key, self.namespace,
            self.clock(), config.CACHE_TTL_SECONDS,
        )
        try:
            self.store().add_cache(record)
        except Exception:
            if config.DEBUG:
                print("[Cache: semantic indexing deferred; exact entry saved]")

    def invalidate(self, cache_id: str | None = None) -> None:
        # Stale vector points become unusable immediately, even during an outage.
        self.memory.invalidate_cache(cache_id)
