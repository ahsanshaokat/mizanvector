"""Abstract base interfaces for vector stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence


Document = Dict[str, Any]


@dataclass
class SearchResult:
    id: Any
    score: float
    content: str
    metadata: Dict[str, Any]


class VectorStore(Protocol):
    """Protocol for pluggable vector store backends."""

    dim: int

    def add_document(
        self,
        content: str,
        embedding: Sequence[float],
        metadata: Dict[str, Any] | None = None,
    ) -> Any:
        ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metric: str = "mizan",
    ) -> List[SearchResult]:
        ...

    def close(self) -> None:
        ...
