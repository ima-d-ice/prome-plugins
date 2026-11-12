"""Helpers shared by embedding consumers (dimension checks, cosine similarity).

Cosine similarity lives here so every consumer compares vectors the same
way. It is implemented with the standard library only: no numpy, no scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from embeddings.contracts.exceptions import EmbeddingError


def validate_dimensions(vectors: Sequence[Sequence[float]], dimension: int) -> None:
    """Raise :class:`EmbeddingError` when ``vectors`` do not match ``dimension``.

    Checks that every vector has the expected length and that the values are
    finite numbers. NaN/inf silently poison cosine similarity, so they are
    rejected here rather than downstream.
    """
    for index, vector in enumerate(vectors):
        if len(vector) != dimension:
            raise EmbeddingError(
                f"embedding dimension mismatch at vector {index}: "
                f"expected {dimension}, got {len(vector)}"
            )
        for value in vector:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise EmbeddingError(
                    f"embedding vector {index} contains a non-numeric value: {value!r}"
                )
            if not math.isfinite(value):
                raise EmbeddingError(
                    f"embedding vector {index} contains a non-finite value: {value!r}"
                )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two equally sized dense vectors.

    The result is in ``[-1.0, 1.0]``. A zero-length vector has no
    direction, so it is reported as ``0.0`` similarity rather than raising:
    a degenerate embedding means "no information", not a crash.
    """
    if len(left) != len(right):
        raise EmbeddingError(
            f"cannot compare vectors of different dimensions: {len(left)} != {len(right)}"
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)
