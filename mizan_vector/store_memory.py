"""In-memory vector store implementation (useful for demos & tests)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from .metrics import cosine_similarity, euclidean_distance, mizan_similarity
from .store_base import SearchResult, VectorStore


@dataclass
class MizanMemoryStore(VectorStore):
    dim: int
    _docs: List[Dict[str, Any]] = field(default_factory=list)

    def add_document(
        self,
        content: str,
        embedding: Sequence[float],
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        vec = np.asarray(embedding, dtype=float)
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"Embedding dim {vec.shape[0]} does not match store dim {self.dim}"
            )
        doc_id = len(self._docs)
        self._docs.append(
            {"id": doc_id, "content": content, "metadata": metadata or {}, "embedding": vec}
        )
        return doc_id

    def _score(self, q: np.ndarray, v: np.ndarray, metric: str) -> float:
        if metric == "mizan":
            return float(mizan_similarity(q, v))
        if metric == "cosine":
            return float(cosine_similarity(q, v))
        if metric == "l2":
            return -float(euclidean_distance(q, v))
        raise ValueError(f"Unknown metric: {metric}")

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metric: str = "mizan",
    ) -> List[SearchResult]:
        q = np.asarray(query_embedding, dtype=float)
        scores: List[SearchResult] = []
        for doc in self._docs:
            s = self._score(q, doc["embedding"], metric)
            scores.append(
                SearchResult(
                    id=doc["id"],
                    score=s,
                    content=doc["content"],
                    metadata=doc["metadata"],
                )
            )
        scores.sort(key=lambda r: r.score, reverse=True)
        return scores[:top_k]

    def close(self) -> None:
        self._docs.clear()
