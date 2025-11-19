"""Mizan-based re-ranking helpers for arbitrary candidate lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

import numpy as np

from .metrics import mizan_similarity


@dataclass
class RankedItem:
    id: Any
    score: float
    payload: Any


def mizan_rerank(
    query_embedding: Sequence[float],
    candidates: List[RankedItem],
    embedding_getter: Callable[[Any], Sequence[float]],
) -> List[RankedItem]:
    q = np.asarray(query_embedding, dtype=float)
    rescored: List[RankedItem] = []
    for item in candidates:
        emb = np.asarray(embedding_getter(item.payload), dtype=float)
        s = float(mizan_similarity(q, emb))
        rescored.append(RankedItem(id=item.id, score=s, payload=item.payload))
    rescored.sort(key=lambda r: r.score, reverse=True)
    return rescored
