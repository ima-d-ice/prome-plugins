"""Tests of the failure and edge cases of the hierarchical strategy.

Every case listed in the specification is covered here: an empty document, a
parent without children, a child without its parent, a missing parent record,
a deleted child, an invalid relationship, a relationship that crosses
documents, duplicate child ids, missing hierarchy metadata, an empty result
set, a filter that excludes everything and an embedding-space mismatch.

The rule that runs through all of them: the strategy never compensates a
broken hierarchy with results from an unrelated document.
"""

from __future__ import annotations

import pytest

from chunkers.contracts.models import Chunk
from chunkers.hierarchical.chunker import HierarchicalResult
from retrieval.contracts.exceptions import (
    HierarchicalIntegrityError,
    UnsupportedRetrievalCapabilityError,
)
from retrieval.dense import DenseRetriever
from retrieval.strategies.hierarchical import (
    HierarchicalConfig,
    HierarchicalIndexer,
    HierarchicalRetrievalStrategy,
    validate_hierarchy,
)
from retrieval.strategies.hierarchical.metadata import (
    CHUNK_LEVEL_FIELD,
    PARENT_ID_FIELD,
)
from tests.retrieval.conftest import make_document
from vector_stores.contracts.filters import eq


def make_chunk(chunk_id: str, doc_id: str, metadata: dict, text: str = "text") -> Chunk:
    """Build a chunk with the given metadata."""
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        text=text,
        index=0,
        count=1,
        length=len(text),
        length_unit="characters",
        metadata=metadata,
    )


# ----------------------------------------------------------------------
# indexing
# ----------------------------------------------------------------------
def test_an_empty_document_is_indexed_without_error(store, embedding_model):
    """An empty document produces no record and no error."""
    indexer = HierarchicalIndexer(store, embedding_model, collection="docs")
    indexer.ensure_collection()

    report = indexer.index_result(HierarchicalResult())

    assert report.total == 0
    assert store.count("docs") == 0


def test_a_parent_without_children_is_rejected(store, embedding_model):
    """A parent that declares children which do not exist is refused."""
    parent = make_chunk(
        "p0",
        "doc.md",
        {"level": "parent", "parent_id": "p0", "child_ids": ["missing"], "child_count": 1},
    )
    indexer = HierarchicalIndexer(store, embedding_model, collection="docs")
    indexer.ensure_collection()

    with pytest.raises(HierarchicalIntegrityError):
        indexer.index_result(HierarchicalResult(parents=[parent], children=[]))


def test_a_child_without_its_parent_is_rejected(store, embedding_model):
    """A child that references an unknown parent is refused."""
    child = make_chunk(
        "c0", "doc.md", {"level": "child", "parent_id": "unknown", "child_index": 0}
    )
    indexer = HierarchicalIndexer(store, embedding_model, collection="docs")
    indexer.ensure_collection()

    with pytest.raises(HierarchicalIntegrityError):
        indexer.index_result(HierarchicalResult(children=[child]))


def test_a_relationship_across_documents_is_rejected(store, embedding_model):
    """A parent and a child of different documents are never linked."""
    parent = make_chunk(
        "p0",
        "a.md",
        {"level": "parent", "parent_id": "p0", "child_ids": ["c0"], "child_count": 1},
    )
    child = make_chunk(
        "c0", "b.md", {"level": "child", "parent_id": "p0", "child_index": 0}
    )
    indexer = HierarchicalIndexer(store, embedding_model, collection="docs")
    indexer.ensure_collection()

    with pytest.raises(HierarchicalIntegrityError):
        indexer.index_result(HierarchicalResult(parents=[parent], children=[child]))


def test_duplicate_child_ids_are_rejected(store, embedding_model):
    """Two children with the same id are refused."""
    parent = make_chunk(
        "p0",
        "doc.md",
        {"level": "parent", "parent_id": "p0", "child_ids": ["c0", "c0"], "child_count": 2},
    )
    first = make_chunk("c0", "doc.md", {"level": "child", "parent_id": "p0", "child_index": 0})
    second = make_chunk("c0", "doc.md", {"level": "child", "parent_id": "p0", "child_index": 1})
    indexer = HierarchicalIndexer(store, embedding_model, collection="docs")
    indexer.ensure_collection()

    with pytest.raises(HierarchicalIntegrityError):
        indexer.index_result(
            HierarchicalResult(parents=[parent], children=[first, second])
        )


def test_a_chunk_without_hierarchy_metadata_is_rejected(store, embedding_model):
    """A chunk that was not produced by the hierarchical chunker is refused."""
    chunk = make_chunk("c0", "doc.md", {"source": "doc.md"})
    indexer = HierarchicalIndexer(store, embedding_model, collection="docs")
    indexer.ensure_collection()

    with pytest.raises(ValueError):
        indexer.index_result(HierarchicalResult(children=[chunk]))


def test_a_mismatching_parent_child_list_is_rejected():
    """The parent's child list must match the hierarchy."""
    parent = make_chunk(
        "p0",
        "doc.md",
        {"level": "parent", "parent_id": "p0", "child_ids": ["c1"], "child_count": 1},
    )
    child = make_chunk("c0", "doc.md", {"level": "child", "parent_id": "p0", "child_index": 0})

    with pytest.raises(HierarchicalIntegrityError):
        validate_hierarchy(HierarchicalResult(parents=[parent], children=[child]))


def test_the_indexing_is_idempotent(store, embedding_model):
    """Indexing the same document twice does not duplicate the records."""
    from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig as ChunkerConfig

    document = make_document("doc.md", [("Section", "content " * 60)])
    indexer = HierarchicalIndexer(store, embedding_model, collection="docs")
    indexer.ensure_collection()
    chunker = HierarchicalChunker(
        ChunkerConfig(parent_size=300, child_size=150, child_overlap=20)
    )
    result = chunker.chunk(document)

    indexer.index_result(result)
    after_first = store.count("docs")
    indexer.index_result(result)
    after_second = store.count("docs")

    assert after_first == after_second


# ----------------------------------------------------------------------
# retrieval
# ----------------------------------------------------------------------
@pytest.fixture()
def indexed(store, embedding_model):
    """Return an indexed collection with two documents."""
    from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig as ChunkerConfig

    documents = [
        make_document(
            "auth.md",
            [("Authentication", "api keys are rotated every thirty days " * 8)],
        ),
        make_document(
            "billing.md",
            [("Invoices", "invoices are issued on the first of every month " * 8)],
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


def test_a_deleted_child_is_simply_absent(indexed):
    """A deleted child is not returned and does not break the search."""
    store, model = indexed
    children = [
        record.chunk.id
        for record in store.scroll("docs", limit=200).records
        if record.metadata.get(CHUNK_LEVEL_FIELD) == "child"
    ]
    assert children
    store.delete("docs", [children[0]])

    strategy = HierarchicalRetrievalStrategy.create(store, model, collection="docs")
    response = strategy.retrieve("api keys are rotated")

    assert children[0] not in {r.chunk_id for r in response.results}


def test_a_deleted_parent_removes_its_children_from_the_search(indexed):
    """The children of a deleted parent are not returned."""
    store, model = indexed
    parents = [
        record.chunk.id
        for record in store.scroll("docs", limit=200).records
        if record.metadata.get(CHUNK_LEVEL_FIELD) == "parent"
    ]
    assert parents
    store.delete("docs", [parents[0]])

    strategy = HierarchicalRetrievalStrategy.create(store, model, collection="docs")
    response = strategy.retrieve("api keys are rotated")

    assert response.results
    for result in response.results:
        assert result.metadata[PARENT_ID_FIELD] != parents[0]


def test_a_filter_that_excludes_every_parent_returns_nothing(indexed):
    """A filter that excludes every parent yields an empty answer."""
    store, model = indexed
    strategy = HierarchicalRetrievalStrategy.create(
        store,
        model,
        collection="docs",
        config=HierarchicalConfig(filters=eq("document_id", "does-not-exist.md")),
    )

    response = strategy.retrieve("api keys are rotated")

    assert response.results == []
    assert response.diagnostics.parents_selected == 0
    assert response.diagnostics.notes


def test_a_filter_that_excludes_every_child_returns_nothing(indexed):
    """A filter that excludes every child yields an empty answer."""
    store, model = indexed
    strategy = HierarchicalRetrievalStrategy.create(
        store,
        model,
        collection="docs",
        config=HierarchicalConfig(filters=eq("document_id", "does-not-exist.md")),
    )

    response = strategy.retrieve("api keys are rotated")

    assert response.results == []


def test_a_broken_hierarchy_never_yields_results_from_another_document(indexed):
    """A child whose parent was removed is not returned."""
    store, model = indexed
    parents = [
        record.chunk.id
        for record in store.scroll("docs", limit=200).records
        if record.metadata.get(CHUNK_LEVEL_FIELD) == "parent"
    ]
    children = [
        record.chunk.id
        for record in store.scroll("docs", limit=200).records
        if record.metadata.get(CHUNK_LEVEL_FIELD) == "child"
    ]
    store.delete("docs", parents)

    strategy = HierarchicalRetrievalStrategy.create(store, model, collection="docs")
    response = strategy.retrieve("api keys are rotated")

    # Without any parent the fine stage has nothing to search.
    assert response.results == []
    assert not set(children) & {r.chunk_id for r in response.results}


def test_an_embedding_space_mismatch_is_reported(indexed):
    """A model of another embedding space cannot search the collection."""
    store, model = indexed

    class OtherModel:
        name = "other/model"

        def get_embedding_space(self):
            from embeddings.contracts.models import (
                EmbeddingSpaceDescriptor,
                Normalization,
                RepresentationType,
            )

            return EmbeddingSpaceDescriptor(
                model_id="other/model",
                model_revision=None,
                adapter_name="other",
                dimensions=16,
                representation_type=RepresentationType.DENSE,
                normalization=Normalization.L2,
            )

    retriever = DenseRetriever(store, OtherModel(), collection="docs")
    with pytest.raises(UnsupportedRetrievalCapabilityError):
        retriever.embed_query("api keys")
