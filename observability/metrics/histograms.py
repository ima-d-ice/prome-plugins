"""Histogram instruments (OTel-backed, fallback, and no-op modes)."""
from __future__ import annotations

from collections.abc import Mapping

from ._labels import labels_tuple, sanitize_metric_labels


class Histogram:
    """Distribution instrument for durations, sizes, and candidate counts."""

    def __init__(self, instrument=None, *, name: str = "", fallback: bool = False) -> None:
        self._instrument = instrument
        self._name = name
        self._fallback: dict | None = (
            {} if (instrument is None and fallback) else None
        )

    def record(self, value: float, labels: Mapping[str, object] | None = None) -> None:
        clean = sanitize_metric_labels(labels)
        value = float(value)
        if self._instrument is not None:
            try:
                self._instrument.record(value, attributes=clean)
            except Exception:
                pass
            return
        if self._fallback is not None:
            key = labels_tuple(clean)
            bucket = self._fallback.setdefault(key, [0, 0.0, value])
            bucket[0] += 1
            bucket[1] += value
            bucket[2] = min(bucket[2], value)

    def snapshot(self) -> list:
        if self._fallback is None:
            return []
        return [
            {
                "labels": dict(key),
                "count": stats[0],
                "sum": stats[1],
                "min": stats[2],
            }
            for key, stats in self._fallback.items()
        ]
