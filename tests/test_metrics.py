from mizan_vector.metrics import (
    cosine_similarity,
    euclidean_distance,
    mizan_distance,
    mizan_similarity,
)


def test_basic_metrics():
    v1 = [1, 2]
    v2 = [2, 4]

    cos = cosine_similarity(v1, v2)
    assert -1.0 <= cos <= 1.0

    assert euclidean_distance(v1, v1) == 0.0

    d = mizan_distance(v1, v2)
    s = mizan_similarity(v1, v2)

    assert 0.0 <= d <= 1.0
    assert 0.0 <= s <= 1.0
    assert abs((1 - d) - s) < 1e-6
