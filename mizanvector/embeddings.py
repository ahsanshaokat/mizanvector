"""Embedding helpers (wrappers around external models)."""

from __future__ import annotations
from typing import Iterable, List

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]


class HFEmbedder:
    """Thin wrapper around a SentenceTransformer text embedding model."""

    def __init__(self, model_name: str = "intfloat/e5-base-v2") -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for HFEmbedder. "
                "Install with: pip install sentence-transformers"
            )

        # IMPORTANT: disable ALL normalization inside the model
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str]) -> List[list[float]]:
        # IMPORTANT: keep normalization disabled here too
        embs = self.model.encode(
            list(texts),
            show_progress_bar=False
        )
        return embs.tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
