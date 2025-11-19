"""Vector similarity and distance metrics, including Mizan variants."""

from typing import Sequence
import numpy as np


def _to_vec(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def cosine_similarity(v1: Sequence[float], v2: Sequence[float], eps: float = 1e-8) -> float:
    v1 = _to_vec(v1)
    v2 = _to_vec(v2)
    num = float(np.dot(v1, v2))
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
    return num / den


def dot_product(v1: Sequence[float], v2: Sequence[float]) -> float:
    v1 = _to_vec(v1)
    v2 = _to_vec(v2)
    return float(np.dot(v1, v2))


def euclidean_distance(v1: Sequence[float], v2: Sequence[float]) -> float:
    v1 = _to_vec(v1)
    v2 = _to_vec(v2)
    return float(np.linalg.norm(v1 - v2))


def mizan_distance(
    v1: Sequence[float], v2: Sequence[float], p: float = 2.0, eps: float = 1e-8
) -> float:
    """Mizan distance between two vectors in [0, 1]."""
    v1 = _to_vec(v1)
    v2 = _to_vec(v2)
    diff = float(np.linalg.norm(v1 - v2))
    num = diff ** p
    den = float(np.linalg.norm(v1) ** p + np.linalg.norm(v2) ** p + eps)
    return float(num / den)


def mizan_similarity(
    v1: Sequence[float], v2: Sequence[float], p: float = 2.0, eps: float = 1e-8
) -> float:
    """Mizan similarity in [0, 1]. Higher = more similar."""
    return 1.0 - mizan_distance(v1, v2, p=p, eps=eps)
