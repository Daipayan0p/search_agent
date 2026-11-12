
from functools import lru_cache

from app.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


class EmbeddingModel:
    def __init__(self):
        self.model = _load_model(EMBEDDING_MODEL)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return vector.tolist()
