"""Integration tests of the hierarchical strategy with the real Qwen model.

The real model is optional: the tests skip when ``sentence-transformers`` is
not installed or when the weights are not in the local cache.  They never
download anything.
"""

from __future__ import annotations

import pytest

from retrieval.strategies.hierarchical import (
    HierarchicalConfig,
    HierarchicalIndexer,
    HierarchicalRetrievalStrategy,
)
from tests.retrieval.conftest import make_document


def qwen_model():
    """Return the real Qwen adapter, skipping when it cannot be used."""
    pytest.importorskip("sentence_transformers")
    from embeddings.qwen3 import Qwen3EmbeddingModel

    try:
        model = Qwen3EmbeddingModel()
        model.get_model_info()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"the Qwen3 model is not usable here: {exc}")
    return model


@pytest.fixture()
def corpus():
    """Return two documents with two sections each."""
    return [
        make_document(
            "auth.md",
            [
                (
                    "Authentication",
                    "API keys are rotated every thirty days by the key rotation "
                    "scheduler. A key that is not rotated within the grace period "
                    "is revoked automatically.",
                ),
                (
                    "Authorization",
                    "Roles grant access to the billing console. The auditor role "
                    "can read invoices but cannot issue refunds.",
                ),
            ],
        ),
        make_document(
            "billing.md",
            [
                (
                    "Invoices",
                    "Invoices are issued on the first day of every month. An "
                    "invoice for a partial month is pro-rated by day.",
                ),
                (
                    "Refunds",
                    "Refunds are issued within five business days. A partial "
                    "refund keeps the original invoice number.",
                ),
            ],
        ),
    ]


def test_the_real_model_indexes_and_retrieves(store, corpus):
    """The hierarchy works with the real production encoder."""
    from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig as ChunkerConfig

    model = qwen_model()
    indexer = HierarchicalIndexer(store, model, collection="docs")
    indexer.ensure_collection()
    chunker = HierarchicalChunker(
        ChunkerConfig(parent_size=400, child_size=200, child_overlap=30)
    )
    for document in corpus:
        indexer.index_result(chunker.chunk(document))

    strategy = HierarchicalRetrievalStrategy.create(
        store, model, collection="docs", config=HierarchicalConfig(parent_top_k=4)
    )
    response = strategy.retrieve("how often are API keys rotated?")

    assert response.results
    # The answer comes from the authentication section.
    assert any(
        "Authentication" in (result.metadata.get("heading_path") or [])
        for result in response.results
    )
    for result in response.results:
        assert result.parent_context is not None
        assert result.metadata["parent_id"]


def test_the_real_model_keeps_the_documents_apart(store, corpus):
    """The fine stage does not cross the document boundary."""
    from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig as ChunkerConfig

    model = qwen_model()
    indexer = HierarchicalIndexer(store, model, collection="docs")
    indexer.ensure_collection()
    chunker = HierarchicalChunker(
        ChunkerConfig(parent_size=400, child_size=200, child_overlap=30)
    )
    for document in corpus:
        indexer.index_result(chunker.chunk(document))

    strategy = HierarchicalRetrievalStrategy.create(
        store, model, collection="docs", config=HierarchicalConfig(parent_top_k=2)
    )
    response = strategy.retrieve("how are refunds issued?")

    selected = set(response.diagnostics.selected_parent_ids)
    for result in response.results:
        assert result.metadata["parent_id"] in selected
        assert result.doc_id in {"auth.md", "billing.md"}
