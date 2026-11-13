"""Gauge instruments (OTel-backed, fallback, and no-op modes)."""
from __future__ import annotations

from collections.abc import Mapping

from ._labels import labels_tuple, sanitize_metric_labels


class Gauge:
    """Last-value-wins instrument (e.g. queue length, memory usage)."""

    def __init__(self, instrument=None, *, name: str = "", fallback: bool = False) -> None:
        self._instrument = instrument
        self._name = name
        self._fallback: dict | None = {} if (instrument is None and fallback) else None

    def set(self, value: float, labels: Mapping[str, object] | None = None) -> None:
        clean = sanitize_metric_labels(labels)
        if self._instrument is not None:
            try:
                self._instrument.set(float(value), attributes=clean)
            except Exception:
                pass
            return
        if self._fallback is not None:
            self._fallback[labels_tuple(clean)] = float(value)

    def snapshot(self) -> list:
        if self._fallback is None:
            return []
        return [{"labels": dict(key), "value": value} for key, value in self._fallback.items()]
