"""Core scalar-level Mizan math utilities."""

from typing import Union

Number = Union[int, float]


def mizan_scalar_similarity(
    x: Number, y: Number, p: float = 2.0, eps: float = 1e-8
) -> float:
    """Scalar Mizan similarity in [0, 1]."""
    num = abs(x - y) ** p
    den = abs(x) ** p + abs(y) ** p + eps
    return 1.0 - float(num / den)


def mizan_scalar_distance(
    x: Number, y: Number, p: float = 2.0, eps: float = 1e-8
) -> float:
    """Scalar Mizan distance in [0, 1]."""
    return 1.0 - mizan_scalar_similarity(x, y, p=p, eps=eps)
