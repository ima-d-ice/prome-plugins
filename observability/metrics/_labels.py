"""Metric label policy enforcement (cardinality control).

Only keys in ``ALLOWED_LABEL_KEYS`` survive, values are truncated to a
bounded length, and forbidden keys are dropped. ``strict`` mode raises
:class:`CardinalityViolationError` instead of dropping.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

from ..contracts.exceptions import CardinalityViolationError
from ..contracts.models import (
    ALLOWED_LABEL_KEYS,
    FORBIDDEN_LABEL_KEYS,
    MAX_LABEL_VALUE_LENGTH,
)

logger = logging.getLogger("prome_rag.metrics")


def sanitize_metric_labels(
    labels: Mapping[str, object] | None,
    *,
    strict: bool | None = None,
) -> dict[str, str]:
    """Filter ``labels`` down to bounded, non-sensitive dimensions.

    ``strict=None`` reads the active configuration's
    ``strict_cardinality`` flag.
    """
    if strict is None:
        from ..configuration import get_config

        strict = bool(get_config().strict_cardinality)

    cleaned: dict[str, str] = {}
    violations: list[str] = []
    for raw_key, raw_value in (labels or {}).items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        if key in FORBIDDEN_LABEL_KEYS:
            violations.append(key)
            continue
        if key not in ALLOWED_LABEL_KEYS:
            violations.append(key)
            continue
        value = "" if raw_value is None else str(raw_value)
        value = value.strip()
        if not value:
            continue
        if len(value) > MAX_LABEL_VALUE_LENGTH:
            suffix = "...(trunc)"
            value = value[: MAX_LABEL_VALUE_LENGTH - len(suffix)] + suffix
        cleaned[key] = value

    if violations:
        message = f"metric labels rejected by cardinality policy: {sorted(violations)}"
        if strict:
            raise CardinalityViolationError(message)
        logger.debug("%s (dropped)", message)
    return cleaned


def labels_tuple(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


__all__ = [
    "ALLOWED_LABEL_KEYS",
    "FORBIDDEN_LABEL_KEYS",
    "MAX_LABEL_VALUE_LENGTH",
    "labels_tuple",
    "sanitize_metric_labels",
]
