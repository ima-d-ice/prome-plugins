"""High-cardinality safeguards: forbidden labels never reach metrics."""
from __future__ import annotations

import pytest

from observability.contracts.exceptions import CardinalityViolationError
from observability.contracts.models import (
    ALLOWED_LABEL_KEYS,
    FORBIDDEN_LABEL_KEYS,
    MAX_LABEL_VALUE_LENGTH,
)
from observability.metrics._labels import sanitize_metric_labels


def test_allowed_keys_cover_spec_dimensions():
    for key in (
        "component",
        "operation",
        "provider",
        "model",
        "outcome",
        "error_category",
        "retrieval_strategy",
    ):
        assert key in ALLOWED_LABEL_KEYS


def test_forbidden_keys_cover_spec_list():
    for key in (
        "user_id",
        "request_id",
        "trace_id",
        "span_id",
        "chunk_id",
        "document_id",
        "file_name",
        "query",
        "prompt",
        "api_key",
        "metadata",
    ):
        assert key in FORBIDDEN_LABEL_KEYS


def test_every_forbidden_key_is_dropped():
    for key in FORBIDDEN_LABEL_KEYS:
        assert sanitize_metric_labels({key: "sensitive", "component": "a"}) == {
            "component": "a"
        }


def test_mixed_labels_keep_only_bounded():
    cleaned = sanitize_metric_labels(
        {
            "component": "retrieval",
            "operation": "dense",
            "provider": "qdrant",
            "model": "qwen3-embedding",
            "retrieval_strategy": "hybrid",
            "outcome": "ok",
            "error_category": "none",
            "user_id": "123",
            "document_id": "doc-1",
            "filename": "secret.pdf",
            "prompt": "tell me...",
            "arbitrary_metadata": {"k": "v"},
        }
    )
    assert cleaned == {
        "component": "retrieval",
        "operation": "dense",
        "provider": "qdrant",
        "model": "qwen3-embedding",
        "retrieval_strategy": "hybrid",
        "outcome": "ok",
        "error_category": "none",
    }


def test_values_are_never_longer_than_cap():
    cleaned = sanitize_metric_labels({"model": "x" * 10000})
    assert len(cleaned["model"]) <= MAX_LABEL_VALUE_LENGTH


def test_strict_mode_raises_on_forbidden():
    with pytest.raises(CardinalityViolationError):
        sanitize_metric_labels({"trace_id": "t" * 32}, strict=True)


def test_non_string_values_are_stringified():
    cleaned = sanitize_metric_labels({"component": 10, "outcome": True})
    assert cleaned == {"component": "10", "outcome": "True"}


def test_empty_and_none_values_are_dropped():
    cleaned = sanitize_metric_labels({"component": "", "model": None, "operation": "parse"})
    assert cleaned == {"operation": "parse"}
