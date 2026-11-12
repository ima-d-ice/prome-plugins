"""Tests of the fine (child-level) stage of the hierarchical strategy."""

from __future__ import annotations

import pytest

from retrieval.contracts.exceptions import (
    RetrievalConfigError,
    UnsupportedRetrievalCapabilityError,
)
from retrieval.dense import DenseRetriever
from retrieval.strategies.hierarchical import (
    ChildStage,
    HierarchicalIndexer,
    child_filter,
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
    parents: dict[str, list[str]] = {}
    for document in documents:
        result = chunker.chunk(document)
        indexer.index_result(result)
        for parent in result.parents:
            parents[parent.id] = list(parent.metadata.get("child_ids") or [])
    return store, embedding_model, parents


def test_the_child_stage_searches_only_the_children_of_the_given_parents(indexed):
    """The returned children all belong to the requested parents."""
    store, model, parents = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ChildStage(retriever, top_k=50)

    parent_id = next(iter(parents))
    results = stage.search_text("api keys", [parent_id])

    expected = set(parents[parent_id])
    assert results
    assert {result.chunk_id for result in results} <= expected
    assert all(result.parent_id == parent_id for result in results)


def test_the_child_stage_never_leaks_across_documents(indexed):
    """A child of another document is never returned."""
    store, model, parents = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ChildStage(retriever, top_k=50)

    auth_parent = next(
        parent_id
        for parent_id, children in parents.items()
        if children
        and store.get("docs", [children[0]])[0].chunk.doc_id == "auth.md"
    )
    results = stage.search_text("api keys rotation", [auth_parent])

    assert results
    for result in results:
        record = store.get("docs", [result.chunk_id])[0]
        assert record.chunk.doc_id == "auth.md"
        assert record.metadata[CHUNK_LEVEL_FIELD] == "child"


def test_the_child_stage_uses_the_level_and_parent_filter(indexed):
    """The filter is the conjunction of the level filter and the parent filter."""
    store, model, parents = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ChildStage(retriever)

    parent_id = next(iter(parents))
    assert stage.filter_for([parent_id]) == child_filter([parent_id])


def test_the_child_stage_refuses_to_search_without_a_parent(indexed):
    """Searching without a selected parent would search every child."""
    store, model, _ = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ChildStage(retriever)

    with pytest.raises(RetrievalConfigError):
        stage.search_text("api keys", [])


def test_the_child_stage_applies_a_user_filter(indexed):
    """A user filter is combined with the level and parent filters."""
    store, model, parents = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ChildStage(retriever, top_k=50, filters=eq("document_id", "billing.md"))

    billing_parent = next(
        parent_id
        for parent_id, children in parents.items()
        if children
        and store.get("docs", [children[0]])[0].chunk.doc_id == "billing.md"
    )
    results = stage.search_text("invoices", [billing_parent])

    assert all(result.doc_id == "billing.md" for result in results)


def test_the_child_stage_preserves_the_provenance(indexed):
    """The provenance of a child survives the round trip through the store."""
    store, model, parents = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ChildStage(retriever, top_k=50)

    parent_id = next(iter(parents))
    results = stage.search_text("api keys", [parent_id])

    assert results
    assert all(result.provenance is not None for result in results)


def test_the_child_stage_preserves_the_parent_reference(indexed):
    """Every child keeps the reference to its parent."""
    store, model, parents = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ChildStage(retriever, top_k=50)

    parent_id = next(iter(parents))
    results = stage.search_text("api keys", [parent_id])

    assert results
    assert all(result.parent_id == parent_id for result in results)


def test_the_child_stage_honours_the_score_threshold(indexed):
    """A score threshold removes the children that score below it."""
    store, model, parents = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    parent_id = next(iter(parents))
    unfiltered = ChildStage(retriever, top_k=50).search_text("api keys", [parent_id])
    assert unfiltered

    threshold = unfiltered[0].score
    filtered = ChildStage(
        retriever, top_k=50, score_threshold=threshold
    ).search_text("api keys", [parent_id])

    assert len(filtered) < len(unfiltered)
    assert all(result.score >= threshold for result in filtered)


def test_a_backend_without_filter_support_is_reported(indexed, monkeypatch):
    """A backend that cannot filter must be reported, never worked around."""
    store, model, parents = indexed
    retriever = DenseRetriever(store, model, collection="docs")
    stage = ChildStage(retriever, top_k=50)

    capabilities = store.capabilities
    monkeypatch.setattr(
        type(store),
        "capabilities",
        property(lambda self: type(capabilities)(**{
            **{f.name: getattr(capabilities, f.name) for f in capabilities.__dataclass_fields__.values()},
            "filter_ops": frozenset(),
        })),
    )

    parent_id = next(iter(parents))
    with pytest.raises(UnsupportedRetrievalCapabilityError):
        stage.search_text("api keys", [parent_id])
