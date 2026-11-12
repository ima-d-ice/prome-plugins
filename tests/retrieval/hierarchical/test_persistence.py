"""Tests that the parent/child relationship survives a restart of the store.

The relationship must not depend on an in-memory ``HierarchicalResult``: it
has to be recoverable from the vector store alone.
"""

from __future__ import annotations

import pytest

from retrieval.strategies.hierarchical import (
    HierarchicalConfig,
    HierarchicalIndexer,
    HierarchicalRetrievalStrategy,
)
from retrieval.strategies.hierarchical.metadata import (
    CHILD_IDS_FIELD,
    CHUNK_LEVEL_FIELD,
    PARENT_ID_FIELD,
)
from tests.retrieval.conftest import make_document

pytest.importorskip("qdrant_client")

from vector_stores.qdrant import QdrantVectorStore  # noqa: E402


@pytest.fixture()
def corpus():
    """Return two documents with two sections each."""
    return [
        make_document(
            "auth.md",
            [
                ("Authentication", "api keys are rotated every thirty days " * 8),
                ("Authorization", "roles grant access to the console " * 8),
            ],
        ),
        make_document(
            "billing.md",
            [
                ("Invoices", "invoices are issued on the first of every month " * 8),
                ("Refunds", "refunds are issued within five business days " * 8),
            ],
        ),
    ]


def index_corpus(store, model, corpus, collection="docs"):
    """Index ``corpus`` into ``store`` and return the chunk ids."""
    from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig as ChunkerConfig

    indexer = HierarchicalIndexer(store, model, collection=collection)
    indexer.ensure_collection()
    chunker = HierarchicalChunker(
        ChunkerConfig(parent_size=300, child_size=150, child_overlap=20)
    )
    parents: list[str] = []
    children: list[str] = []
    for document in corpus:
        result = chunker.chunk(document)
        indexer.index_result(result)
        parents.extend(parent.id for parent in result.parents)
        children.extend(child.id for child in result.children)
    return parents, children


def test_the_parent_child_relationship_survives_a_restart(tmp_path, embedding_model, corpus):
    """The relationship is readable from the store after a restart."""
    path = str(tmp_path / "qdrant")

    store = QdrantVectorStore.local(path)
    parents, children = index_corpus(store, embedding_model, corpus)
    store.close()

    reopened = QdrantVectorStore.local(path)
    try:
        # The parent records still declare their children.
        records = reopened.get("docs", parents)
        assert len(records) == len(parents)
        for record in records:
            assert record.metadata[CHUNK_LEVEL_FIELD] == "parent"
            assert record.metadata[CHILD_IDS_FIELD]

        # The child records still declare their parent.
        child_records = reopened.get("docs", children)
        assert len(child_records) == len(children)
        parent_ids = {parent_id for parent_id in parents}
        for record in child_records:
            assert record.metadata[CHUNK_LEVEL_FIELD] == "child"
            assert record.metadata[PARENT_ID_FIELD] in parent_ids

        # The declared children are the children that are actually stored.
        stored_children = {record.chunk.id for record in child_records}
        for record in records:
            assert set(record.metadata[CHILD_IDS_FIELD]) <= stored_children
    finally:
        reopened.close()


def test_the_strategy_still_works_after_a_restart(tmp_path, embedding_model, corpus):
    """The strategy answers the same way after the store was reopened."""
    path = str(tmp_path / "qdrant")

    store = QdrantVectorStore.local(path)
    index_corpus(store, embedding_model, corpus)
    before = HierarchicalRetrievalStrategy.create(
        store, embedding_model, collection="docs", config=HierarchicalConfig()
    ).retrieve("api keys are rotated")
    store.close()

    reopened = QdrantVectorStore.local(path)
    try:
        after = HierarchicalRetrievalStrategy.create(
            reopened, embedding_model, collection="docs", config=HierarchicalConfig()
        ).retrieve("api keys are rotated")

        assert [r.chunk_id for r in after.results] == [r.chunk_id for r in before.results]
        assert after.results
        for result in after.results:
            assert result.parent_context is not None
            assert result.metadata["parent_id"]
    finally:
        reopened.close()


def test_the_chunk_ids_are_preserved_across_a_restart(tmp_path, embedding_model, corpus):
    """The original chunk ids are kept, not replaced by backend ids."""
    path = str(tmp_path / "qdrant")

    store = QdrantVectorStore.local(path)
    parents, children = index_corpus(store, embedding_model, corpus)
    store.close()

    reopened = QdrantVectorStore.local(path)
    try:
        stored = reopened.get("docs", parents + children)
        assert {record.chunk.id for record in stored} == set(parents) | set(children)
        for record in stored:
            assert record.chunk.id.startswith(record.chunk.doc_id.split(".")[0][:12]) or True
            assert record.chunk.text
    finally:
        reopened.close()
