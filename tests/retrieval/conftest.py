"""Shared fixtures for the retrieval test suites.

The doubles themselves live in :mod:`tests.retrieval.support`, so the
helpers can be imported by the test modules as well as used as fixtures.

No model is downloaded and no inference over real weights is run: every
vector comes from the deterministic test double, so the tests are fast and
reproducible.
"""

from __future__ import annotations

from tests.retrieval.support import (
    DIMENSIONS,
    FakeEmbeddingModel,
    FakeVectorStore,
    RecordingEmbeddingModel,
    collection_config,
    embedding_model,
    fake_store,
    make_chunk,
    make_document,
    make_record,
    provenance_of,
    store,
)

__all__ = [
    "DIMENSIONS",
    "FakeEmbeddingModel",
    "FakeVectorStore",
    "RecordingEmbeddingModel",
    "collection_config",
    "embedding_model",
    "fake_store",
    "make_chunk",
    "make_document",
    "make_record",
    "provenance_of",
    "store",
]
