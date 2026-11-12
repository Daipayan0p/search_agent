# SearchMind

A Python CLI research assistant using Gemini, Tavily, SQLite, Sentence Transformers,
and Qdrant. It reuses safe answers before making model calls and retrieves personal
conversation history only when the question calls for it.

## Setup and run

Python 3.11+ is recommended. With `uv`:

```sh
uv venv
uv pip install -r requirements.txt
uv run --no-project python -m app.main
```

Or activate your virtual environment and use
`python -m pip install -r requirements.txt`, then `python -m app.main`.

Set `GEMINI_API_KEY` and `TAVILY_API_KEY` in the project-root `.env` file. Set
`GEMINI_MODEL` to a model available to your Gemini account if the existing default
is unavailable. Keys are checked only when the corresponding service is needed;
an exact cache hit can work offline without keys.

Run a Qdrant server locally (requires Docker), or set `QDRANT_URL` to your server:

```sh
docker run --name searchmind-qdrant -p 6333:6333 -v searchmind-qdrant:/qdrant/storage qdrant/qdrant
```

The local embedding model downloads on its first use and is shared for the life of
the process. Exact cache hits never load it. Qdrant failures degrade to SQLite
history and uncached generation; Gemini/Tavily still require their own services.

CLI commands: `history`, `new`, `backfill`, `clear-cache`, `exit` (or `quit`).
`new` creates another session for the same local user; relevant long-term memories
can be retrieved across these sessions.

## Architecture and workflow

- SQLite is the source of truth for original messages, complete conversation turns,
  canonical cached answers, semantic-question aliases, and indexing progress.
- Qdrant has separate memory and answer-cache collections. Memory points reference
  SQLite message IDs; cache points reference SQLite canonical cache IDs.
- Sentence Transformers creates local embeddings, sharing one loaded model.
- Gemini handles answers and ambiguous routing. Recognized routes skip a separate
  router call. Tavily remains the only web-search provider.

On each request:

1. Normalize whitespace and case. Apply the local cache eligibility policy.
2. Look for a live exact SQLite answer, including previously accepted aliases.
   A hit prints `[SearchMind route: cache]`, records the turn, and returns without
   Gemini, Tavily, Qdrant, or embeddings.
3. For eligible misses with existing cache records, search Qdrant's answer cache.
   Require the configured cosine threshold **and** the same approved question
   template, complete subject, and predicate. Re-check SQLite validity before
   returning. Save the accepted wording as an exact alias of that same record.
4. Choose a route. Retrieve long-term memory only for `memory` or `hybrid`.
   Hydrate complete turns from SQLite and filter by score and topic overlap.
5. Search Tavily for `web` or `hybrid`; generate one answer with Gemini. Commit the
   conversation to SQLite before attempting vector indexing. Cache eligible direct
   answers only when they pass the answer safety checks.

| Question | Route on first request | Repeated request |
| --- | --- | --- |
| What is Python? | `direct` | `cache` if the answer passes safety checks |
| What project was I building yesterday? | `memory` | `memory` |
| What is the latest Python version? | `web` | `web` |
| Is the Kubernetes version I previously used still supported? | `hybrid` | `hybrid` |

For hybrid requests, relevant **user statements** identify historical versions in
Tavily's query. Prior assistant answers remain unverified conversation and are not
used to supply historical facts to the search query. If history is absent or
conflicting, the answer prompt requires acknowledging that limitation.

## Cache safety and configuration

Eligibility is deliberately a small allowlist in `app/cache.py`: definitions of
specific stable subjects and a few explicit Python language-property questions.
Unknown questions bypass caching; there is no paid eligibility-classification call.
Even a similarity of 1.0 cannot override subject/predicate checks. For example,
`What is Python?` and `Define Python` can share an answer, but Python lists,
Python tuples, compiled languages, and interpreted languages stay separate.

Personal/history questions, follow-ups, relative dates, current facts, extra
qualifiers, negation outside the approved templates, and unknown subjects are not
eligible. Eligible generation omits conversation context and retrieved memory.
Answers with recognized credentials, personal details, uncertainty, insufficient
evidence, current-information language, or URLs are rejected. Web, memory, and
hybrid answers are never cached.

| Setting | Default | Purpose |
| --- | --- | --- |
| `GEMINI_MODEL` | Existing `gemini-3.5-flash-lite` | Gemini model; must be available to your account |
| `DATABASE_PATH` | Project `data/searchmind.db` | Persistent local database |
| `QDRANT_URL` | `http://localhost:6333` | Vector service |
| `QDRANT_COLLECTION` | `searchmind_memory` | Conversation index |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embeddings |
| `CACHE_ENABLED` | `true` | Disable cache reads and writes with `false` |
| `CACHE_COLLECTION` | `searchmind_answer_cache` | Separate semantic cache index |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Minimum cosine similarity, plus equivalence checks |
| `CACHE_TTL_SECONDS` | `604800` (7 days) | Maximum answer lifetime |
| `CACHE_VERSION` | `1` | Increment and restart to invalidate previous cache namespaces |
| `SEARCHMIND_DEBUG` | `false` | Counts, scores, cache hits, API call labels, routes, latency |

Cache namespaces include the model names and cache version. Shortening the TTL also
limits existing entries; extending it does not revive expired entries. `clear-cache`
invalidates all canonical records and aliases immediately. From Python,
`agent.cache.invalidate(cache_id)` invalidates a single record. Orphaned Qdrant
points cannot serve answers because every hit must resolve a live SQLite record.
Expired/orphaned cache points are retained until the collection is recreated; there
is no automatic physical compaction in this version.

## Memory, migrations, and recovery

Migrations add tables/columns and backfill missing IDs/session IDs without deleting
original rows. Legacy adjacent user/assistant messages are paired within their own
sessions. New exchanges share a turn ID and commit atomically. Every point uses its
original message UUID, with session, role, timestamp, and turn metadata.

Both user and assistant messages are searchable, but results are hydrated from
SQLite, deduplicated by turn, checked for topic overlap with the **user** statement,
and supplied as untrusted historical text. Sensitive messages are omitted from the
vector index. Recognized credentials are never printed by debug logging; normal
CLI answers and explicit `history` still display conversation content.

Run `backfill` to index up to 100 pending messages. Repeat until `indexed`, `failed`,
and `skipped` are all zero. Acknowledged Qdrant upserts are marked in SQLite;
failures remain pending. Retries use the same UUIDs, preventing duplicate points
even after a crash between upsert and acknowledgement. Cache-hit turns deliberately
stay pending to keep the immediate response free of embedding/network work.

After changing embedding models, use new **memory and cache collection names** and
backfill. Never mix models in a collection, even if their vector dimensions match.
If an indexed collection is deleted externally, use a new collection name before
backfill so the indexing-progress namespace changes. Backfill repairs conversation
indexing; failed semantic cache writes remain exact-only until regenerated.

## Tests

```sh
uv pip install -r requirements-dev.txt
uv run --no-project python -m pytest -q
```

Tests mock Gemini/Tavily, use an in-memory Qdrant client and deterministic fake
embeddings, and require no API credits, Qdrant server, or model downloads.

## Limitations and tradeoffs

- Precision is prioritized over hit rate. Most general-knowledge wording is not
  cache eligible yet; broadening the allowlist requires equivalence/safety tests.
- This is a single-user local application. Cross-session memory is intentional;
  there is no tenant isolation or authentication layer.
- Safety and routing use conservative rules plus Gemini for ambiguous routing.
  They are not a factual verifier, universal secret detector, or guarantee against
  hallucinations. Clear an incorrect cached answer and bump `CACHE_VERSION` after
  changing policy/prompts. SQLite retains original messages, including user input.
- Topic-overlap filtering can miss paraphrased or very broad recollection queries.
  Relative dates/conflicting history are passed with timestamps for Gemini to
  interpret; the application does not implement a temporal query engine.
- Semantic misses require embedding and vector work; exact hits do not. Unknown
  routes can require a router call and an answer call. Qdrant outages may add
  timeout latency; persistence/backfill provide recovery rather than a distributed
  transaction. No automatic background retry or vector compaction runs.
- Live Gemini model availability, Tavily responses, remote Qdrant connectivity,
  and real embedding quality must be validated in your environment.
