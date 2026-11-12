"""Retrieval metrics for the hierarchical evaluation.

The metrics are the standard ones: recall at k, mean reciprocal rank, nDCG at
k and the latency percentiles.  Every metric is computed from a ranked list of
result ids and a set of relevant ids.

Two conventions:

* a *hit* is a relevant id that appears in the ranked list; the rank is
  1-based;
* a metric that cannot be computed (no relevant id at all) is reported as
  ``None`` and excluded from the averages, never as a zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def _clean(ranked: Sequence[str], relevant: Iterable[str]) -> list[str]:
    """Return the relevant ids of ``ranked``, in rank order, without repeats."""
    wanted = set(relevant)
    seen: set[str] = set()
    hits: list[str] = []
    for item in ranked:
        if item in wanted and item not in seen:
            seen.add(item)
            hits.append(item)
    return hits


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """Return the fraction of the relevant ids that appear in the top ``k``.

    Returns ``None`` when there is no relevant id to find.
    """
    relevant = list(relevant)
    if not relevant:
        return None
    hits = _clean(ranked[:k], relevant)
    return len(hits) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """Return whether at least one relevant id appears in the top ``k``."""
    relevant = list(relevant)
    if not relevant:
        return None
    return 1.0 if _clean(ranked[:k], relevant) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float | None:
    """Return the reciprocal rank of the first relevant hit."""
    relevant = list(relevant)
    if not relevant:
        return None
    hits = _clean(ranked, relevant)
    if not hits:
        return 0.0
    return 1.0 / (ranked.index(hits[0]) + 1)


def dcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return the discounted cumulative gain of the top ``k``."""
    relevant = set(relevant)
    gain = 0.0
    for position, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            gain += 1.0 / math.log2(position + 1)
    return gain


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """Return the normalized discounted cumulative gain of the top ``k``.

    Returns ``None`` when there is no relevant id to find.
    """
    relevant = list(relevant)
    if not relevant:
        return None
    ideal = dcg_at_k(list(relevant), relevant, k)
    if ideal == 0.0:
        return None
    return dcg_at_k(ranked, relevant, k) / ideal


@dataclass(frozen=True)
class MetricSummary:
    """The average of one metric over a set of queries."""

    name: str
    value: float | None
    samples: int
    """Number of queries the metric could be computed for."""


def summarize(name: str, values: Iterable[float | None]) -> MetricSummary:
    """Return the average of ``values``, skipping the ones that are ``None``."""
    usable = [value for value in values if value is not None]
    if not usable:
        return MetricSummary(name=name, value=None, samples=0)
    return MetricSummary(
        name=name,
        value=sum(usable) / len(usable),
        samples=len(usable),
    )


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return the given percentile of ``values``.

    Uses the nearest-rank method, so the result is always one of the
    measured values.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = math.ceil(fraction * len(ordered))
    index = min(max(rank - 1, 0), len(ordered) - 1)
    return ordered[index]


__all__ = [
    "MetricSummary",
    "dcg_at_k",
    "hit_at_k",
    "ndcg_at_k",
    "percentile",
    "recall_at_k",
    "reciprocal_rank",
    "summarize",
]
