"""Tests of the indexing coordinator.

The coordinator writes the same chunks into the dense index (vector store)
and the lexical index (SQLite FTS5).  Its contract is that a failure of one
backend is reported, never hidden: a dense-only update is a definite
``partial`` in the report, and the lexical index can be reconciled later.
"""

from __future__ import annotations

import pytest

from embeddings.contracts.models import RepresentationType
from retrieval.indexing import IndexingCoordinator
from retrieval.lexical.bm25.sqlite_fts5 import SQLiteFTS5Index
from tests.retrieval.support import FakeEmbeddingModel, FakeVectorStore, make_chunk
from vector_stores.contracts.models import (
    CollectionConfig,
    DistanceMetric,
    VectorSpaceConfig,
)


@pytest.fixture()
def chunks():
    model = FakeEmbeddingModel()
    return [
        make_chunk("c0", "auth.md", "api keys are rotated every thirty days", index=0),
        make_chunk("c1", "auth.md", "invoices are issued on the first day", index=1),
    ]


@pytest.fixture()
def store(tmp_path):
    """Return an empty fake vector store with the dense space configured."""
    model = FakeEmbeddingModel()
    instance = FakeVectorStore()
    space = VectorSpaceConfig(
        name="dense",
        vector_type=RepresentationType.DENSE,
        embedding_space=model.get_embedding_space(),
        dimensions=model.dimension,
        distance=DistanceMetric.COSINE,
    )
    instance.create_collection(
        CollectionConfig(name="docs", vector_spaces={"dense": space})
    )
    return instance


@pytest.fixture()
def index(tmp_path):
    """Return an empty lexical index."""
    instance = SQLiteFTS5Index(tmp_path / "lexical.db")
    instance.open()
    yield instance
    instance.close()


def test_the_coordinator_writes_both_indexes(store, index, chunks):
    """Both indexes hold the chunks after an upsert."""
    coordinator = IndexingCoordinator(
        store, index, collection="docs", embedding_model=FakeEmbeddingModel()
    )

    assert index.count() == 0
    report = coordinator.upsert(chunks)

    assert report.succeeded
    assert report.vector_store_written == 2
    assert report.lexical_written == 2
    assert index.count() == 2


def test_the_coordinator_embeds_the_chunks_once(store, index, chunks):
    """The coordinator embeds every chunk exactly once."""
    model = FakeEmbeddingModel()
    model.document_calls.clear()
    coordinator = IndexingCoordinator(
        store, index, collection="docs", embedding_model=model
    )

    coordinator.upsert(chunks)

    assert len(model.document_calls) == 1
    assert len(model.document_calls[0]) == len(chunks)


def test_a_vector_store_failure_is_reported(store, index, chunks):
    """A failing vector store is reported, not hidden."""
    coordinator = IndexingCoordinator(
        store, index, collection="docs", embedding_model=FakeEmbeddingModel()
    )

    def broken_upsert(*args, **kwargs):
        raise RuntimeError("boom")

    coordinator.vector_store.upsert = broken_upsert

    report = coordinator.upsert(chunks)

    assert report.vector_store_error is not None
    assert report.lexical_error is None
    assert report.partial is True
    assert report.succeeded is False


def test_a_lexical_failure_is_reported(store, index, chunks):
    """A failing lexical index is reported, not hidden."""
    coordinator = IndexingCoordinator(
        store, index, collection="docs", embedding_model=FakeEmbeddingModel()
    )

    def broken_upsert(*args, **kwargs):
        raise RuntimeError("boom")

    coordinator.lexical_index.upsert = broken_upsert

    report = coordinator.upsert(chunks)

    assert report.lexical_error is not None
    assert report.vector_store_error is None
    assert report.partial is True
    assert report.succeeded is False


def test_a_partial_failure_can_be_reconciled(store, index, chunks):
    """A half-finished upsert can be reconciled by writing the lexical side."""
    coordinator = IndexingCoordinator(
        store, index, collection="docs", embedding_model=FakeEmbeddingModel()
    )

    calls = {"n": 0}

    def sometimes_broken(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        # The second call (the reconcile) works.
        return 2

    coordinator.lexical_index.upsert = sometimes_broken
    report = coordinator.upsert(chunks)
    assert report.lexical_error is not None

    written = coordinator.reconcile_lexical(chunks)

    assert written == 2
    assert calls["n"] == 2


def test_deleting_from_both_indexes(store, index, chunks):
    """Deleting removes the chunks from both indexes."""
    coordinator = IndexingCoordinator(
        store, index, collection="docs", embedding_model=FakeEmbeddingModel()
    )
    coordinator.upsert(chunks)

    report = coordinator.delete(["c0"])

    assert report.vector_store_error is None
    assert report.lexical_error is None
    assert [r.chunk_id for r in index.get(["c0", "c1"])] == ["c1"]


def test_deleting_documents_removes_their_chunks(store, index, chunks):
    """Deleting a document removes its chunks from both indexes."""
    coordinator = IndexingCoordinator(
        store, index, collection="docs", embedding_model=FakeEmbeddingModel()
    )
    coordinator.upsert(chunks)

    report = coordinator.delete_documents(["auth.md"])

    assert report.vector_store_error is None
    assert report.lexical_error is None
    assert index.count() == 0


def test_the_coordinator_writes_the_metadata_into_both_indexes(store, index, chunks):
    """The record metadata is written to both indexes."""
    coordinator = IndexingCoordinator(
        store, index, collection="docs", embedding_model=FakeEmbeddingModel()
    )
    coordinator.upsert(chunks, metadata={"tenant": "acme"})

    records = index.get(["c0"])
    assert records
    assert records[0].metadata["tenant"] == "acme"


def test_the_report_is_serializable(store, index, chunks):
    """The report can be written to a JSON file."""
    coordinator = IndexingCoordinator(
        store, index, collection="docs", embedding_model=FakeEmbeddingModel()
    )
    report = coordinator.upsert(chunks)

    data = report.as_dict()
    assert data["succeeded"] is True
    assert data["vector_store_written"] == 2
    assert data["lexical_written"] == 2
