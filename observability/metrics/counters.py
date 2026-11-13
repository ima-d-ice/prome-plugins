"""Counter instruments (OTel-backed, fallback, and no-op modes)."""
from __future__ import annotations

from collections.abc import Mapping

from ._labels import labels_tuple, sanitize_metric_labels


class Counter:
    """Monotonic additive instrument.

    ``instrument`` is an ``opentelemetry.metrics.Counter`` when the SDK is
    configured; ``fallback=True`` enables the in-memory implementation used
    when the SDK is missing but observability is enabled.
    """

    def __init__(self, instrument=None, *, name: str = "", fallback: bool = False) -> None:
        self._instrument = instrument
        self._name = name
        self._fallback: dict | None = {} if (instrument is None and fallback) else None

    def add(self, amount: float = 1.0, labels: Mapping[str, object] | None = None) -> None:
        clean = sanitize_metric_labels(labels)
        if self._instrument is not None:
            try:
                self._instrument.add(float(amount), attributes=clean)
            except Exception:
                pass
            return
        if self._fallback is not None:
            key = labels_tuple(clean)
            self._fallback[key] = self._fallback.get(key, 0.0) + float(amount)

    def inc(self, labels: Mapping[str, object] | None = None) -> None:
        self.add(1.0, labels)

    def snapshot(self) -> list:
        if self._fallback is None:
            return []
        return [{"labels": dict(key), "value": value} for key, value in self._fallback.items()]


class UpDownCounter(Counter):
    """Bidirectional additive instrument (e.g. active request levels)."""
