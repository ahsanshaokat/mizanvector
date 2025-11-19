"""MizanVector: scale-aware similarity & vector search framework."""

from .config import MizanConfig
from .metrics import (
    cosine_similarity,
    euclidean_distance,
    mizan_distance,
    mizan_similarity,
    dot_product,
)
from .store_pgvector import MizanPgVectorStore
from .store_memory import MizanMemoryStore
from .embeddings import HFEmbedder
from . import rerank
from . import losses

__all__ = [
    "MizanConfig",
    "cosine_similarity",
    "euclidean_distance",
    "mizan_distance",
    "mizan_similarity",
    "dot_product",
    "MizanPgVectorStore",
    "MizanMemoryStore",
    "HFEmbedder",
    "rerank",
    "losses",
]
