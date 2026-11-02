"""Local SentenceTransformers Embedder for Nykaa Support Agent.

Zero API keys, 100% local CPU execution.
Model: all-MiniLM-L6-v2 (384-dimensional dense vectors).
"""

from typing import List
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDER_INSTANCE = None


def get_embedder() -> SentenceTransformer:
    """Singleton provider for local SentenceTransformer model."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = SentenceTransformer(_MODEL_NAME)
    return _EMBEDDER_INSTANCE


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generates normalized vector embeddings for a list of texts."""
    model = get_embedder()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_text(text: str) -> List[float]:
    """Generates a single normalized vector embedding."""
    return embed_texts([text])[0]

