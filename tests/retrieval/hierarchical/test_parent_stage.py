"""Tests of the coarse (parent-level) stage of the hierarchical strategy."""

from __future__ import annotations

import pytest

from retrieval.contracts.exceptions import RetrievalConfigError
from retrieval.dense import DenseRetriever
from retrieval.strategies.hierarchical import (
    HierarchicalIndexer,
    ParentStage,
    parent_filter,
)
from retrieval.strategies.hierarchical.metadata import CHUNK_LEVEL_FIELD
from tests.retrieval.conftest import make_document
from vector_stores.contracts.filters import eq


@pytest.fixture()
def indexed(store, embedding_model):
    """Return an indexed collection with two documents and two parents each."""
    from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig as ChunkerConfig

    documents = [
        make_document(
            "auth.md",
            [
                ("Authentication", "api keys are rotated every thirty days by the scheduler " * 6),
                ("Authorization", "roles grant access to the billing console " * 6),
            ],
        ),
        make_document(
            "billing.md",
            [
                ("Invoices", "invoices are issued on the first of every month " * 6),
                ("Refunds", "refunds are issued within five business days " * 6),
            ],
        ),
    ]
    indexer = HierarchicalIndexer(store, embedding_model, collection="docs")
    indexer.ensure_collection()
    chunker = HierarchicalChunker(
        ChunkerConfig(parent_size=300, child_size=150, child_overlap=20)
    )
    for document in documents:
        indexer.index_result(chunker.chunk(document))
    return store, embedding_model


def test_the_parent_stage_returns_only_parent_records(indexed):
    """A parent search must never return a child record."""
    store, model = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ParentStage(retriever, top_k=10)

    results = stage.search_text("api keys")

    assert results
    for result in results:
        record = store.get("docs", [result.chunk_id])[0]
        assert record.metadata[CHUNK_LEVEL_FIELD] == "parent"


def test_the_parent_stage_uses_the_parent_filter(indexed):
    """The stage searches with the parent-level filter."""
    store, model = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ParentStage(retriever)

    assert stage.filter == parent_filter()


def test_the_parent_stage_applies_a_user_filter(indexed):
    """A user filter is combined with the parent filter."""
    store, model = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ParentStage(retriever, filters=eq("document_id", "auth.md"))

    results = stage.search_text("invoices")

    # The filter excludes every parent of the other document.
    assert all(result.doc_id == "auth.md" for result in results)


def test_the_parent_stage_preserves_the_provenance(indexed):
    """The provenance of a parent survives the round trip through the store."""
    store, model = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ParentStage(retriever, top_k=5)

    results = stage.search_text("api keys")

    assert results
    assert all(result.provenance is not None for result in results)
    assert all(result.provenance.source for result in results)


def test_the_parent_stage_preserves_the_section_context(indexed):
    """The heading path of a parent is part of the result."""
    store, model = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ParentStage(retriever, top_k=10)

    results = stage.search_text("api keys rotation scheduler")

    assert any(result.heading_path for result in results)
    assert all(result.child_count >= 0 for result in results)


def test_the_parent_stage_honours_the_score_threshold(indexed):
    """A score threshold removes the parents that score below it."""
    store, model = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    unfiltered = ParentStage(retriever, top_k=10).search_text("api keys")
    assert unfiltered

    threshold = unfiltered[0].score
    filtered = ParentStage(retriever, top_k=10, score_threshold=threshold).search_text(
        "api keys"
    )

    assert len(filtered) < len(unfiltered)
    assert all(result.score >= threshold for result in filtered)


def test_the_parent_stage_rejects_an_invalid_top_k(indexed):
    """A non-positive limit is a configuration error."""
    store, model = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    with pytest.raises(RetrievalConfigError):
        ParentStage(retriever, top_k=0)


def test_the_parent_stage_reports_an_empty_query(indexed):
    """An empty query is rejected instead of returning everything."""
    store, model = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    with pytest.raises(RetrievalConfigError):
        ParentStage(retriever).search_text("   ")
