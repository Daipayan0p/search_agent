import os 
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)
QDRANT_URL = os.getenv(
    "QDRANT_URL", "http://localhost:6333"
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION", "searchmind_memory"
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DATABASE_PATH = os.getenv(
    "DATABASE_PATH", str(PROJECT_ROOT / "data" / "searchmind.db")
)
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_COLLECTION = os.getenv("CACHE_COLLECTION", "searchmind_answer_cache")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "604800"))
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
# Change the namespace to invalidate all previous entries without deleting history.
CACHE_VERSION = os.getenv("CACHE_VERSION", "1")
DEBUG = os.getenv("SEARCHMIND_DEBUG", "false").lower() == "true"

if CACHE_TTL_SECONDS <= 0:
    raise ValueError("CACHE_TTL_SECONDS must be positive")
if not 0 <= CACHE_SIMILARITY_THRESHOLD <= 1:
    raise ValueError("CACHE_SIMILARITY_THRESHOLD must be between 0 and 1")
if CACHE_COLLECTION == QDRANT_COLLECTION:
    raise ValueError("CACHE_COLLECTION must be separate from QDRANT_COLLECTION")
