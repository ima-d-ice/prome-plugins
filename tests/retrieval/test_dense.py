"""Tests of the dense retrieval channel.

The tests use the deterministic fake model and the in-memory fake store, so
no weight is downloaded and no inference over real weights is run.
"""

from __future__ import annotations

import pytest

from retrieval.contracts.exceptions import (
    RetrievalConfigError,
    UnsupportedRetrievalCapabilityError,
)
from retrieval.contracts.models import RetrievalQuery
from retrieval.dense.qwen import QWEN_MODEL_ID, QwenDenseRetriever
from retrieval.dense.retriever import DenseRetriever, filter_fields, required_filter_ops
from vector_stores.contracts.filters import FilterOp, and_, eq, is_in
from vector_stores.contracts.models import ScoreDirection

from tests.retrieval.support import FakeEmbeddingModel, make_chunk, make_record


@pytest.fixture()
def dense(fake_store, collection_config, embedding_model):
    """Return a dense retriever over a fake store with three documents."""
    fake_store.create_collection(collection_config)
    chunks = [
        make_chunk("c0", "a.md", "api keys are rotated every thirty days", index=0),
        make_chunk("c1", "a.md", "roles grant access to the console", index=1),
        make_chunk("c2", "b.md", "invoices are issued every month", index=0),
    ]
    for chunk in chunks:
        fake_store.upsert("docs", [make_record(chunk, embedding_model)])
    return QwenDenseRetriever(
        fake_store, embedding_model, collection="docs", expected_model_id=None
    )


def test_the_query_is_embedded_with_the_injected_model(dense, embedding_model):
    """The query is embedded with the model that was injected."""
    dense.retrieve(RetrievalQuery(text="api keys rotated"))

    assert embedding_model.query_calls == ["api keys rotated"]


def test_the_dense_channel_returns_the_matching_passages(dense):
    """The passages that share the terms of the query rank first."""
    response = dense.retrieve(RetrievalQuery(text="api keys rotated"))

    assert response.results
    assert response.results[0].chunk_id == "c0"
    assert response.strategy == "dense-qwen"


def test_the_chunk_identity_is_preserved(dense):
    """The original chunk id is returned, never a backend one."""
    response = dense.retrieve(RetrievalQuery(text="invoices"))

    assert {item.chunk_id for item in response.results} <= {"c0", "c1", "c2"}


def test_the_provenance_is_preserved(dense):
    """The page and block provenance survives the search."""
    response = dense.retrieve(RetrievalQuery(text="api keys"))

    assert response.results
    provenance = response.results[0].provenance
    assert provenance is not None
    assert provenance.source
    assert provenance.pages == (1,)


def test_the_metadata_filter_is_applied(dense):
    """A filtered search only returns the passages that pass the filter."""
    response = dense.retrieve(
        RetrievalQuery(text="api keys invoices", filters=eq("doc_id", "b.md"))
    )

    assert response.results
    assert all(item.doc_id == "b.md" for item in response.results)


def test_a_filter_that_excludes_everything_returns_nothing(dense):
    """A filter that matches nothing returns nothing."""
    response = dense.retrieve(
        RetrievalQuery(text="api keys", filters=eq("doc_id", "missing.md"))
    )

    assert response.results == []


def test_an_unsupported_filter_operation_is_reported(fake_store, collection_config, embedding_model):
    """A backend that cannot express a filter is reported, not worked around."""
    from tests.retrieval.support import FakeVectorStore

    store = FakeVectorStore(filter_ops=frozenset())
    store.create_collection(collection_config)
    retriever = DenseRetriever(store, embedding_model, collection="docs")

    with pytest.raises(UnsupportedRetrievalCapabilityError):
        retriever.search([0.0] * 16, limit=5, filter=eq("doc_id", "a.md"))


def test_an_embedding_space_mismatch_is_reported(dense, fake_store, collection_config):
    """A model of another embedding space cannot search the collection."""
    other = FakeEmbeddingModel(model_id="test/other-model")
    mismatched = DenseRetriever(
        fake_store, other, collection="docs", space="dense"
    )

    with pytest.raises(UnsupportedRetrievalCapabilityError):
        mismatched.embed_query("api keys")


def test_a_collection_of_another_model_is_reported(fake_store, collection_config, embedding_model):
    """A collection built for another encoder is reported."""
    fake_store.create_collection(collection_config)
    retriever = QwenDenseRetriever(
        fake_store, embedding_model, collection="docs", expected_model_id=QWEN_MODEL_ID
    )

    with pytest.raises(UnsupportedRetrievalCapabilityError):
        retriever.embed_query("api keys")


def test_an_empty_query_is_rejected(dense):
    """An empty query raises instead of returning the whole collection."""
    with pytest.raises(RetrievalConfigError):
        dense.retrieve(RetrievalQuery(text="   "))


def test_the_score_direction_is_recorded(dense):
    """The score of the dense channel is a similarity: bigger is better."""
    response = dense.retrieve(RetrievalQuery(text="api keys"))

    assert response.results[0].score_direction is ScoreDirection.HIGHER_IS_BETTER
    assert response.results[0].score_interpretation == "cosine"


def test_the_ordering_is_deterministic(dense):
    """Two identical queries return the same order."""
    first = dense.retrieve(RetrievalQuery(text="api keys roles"))
    second = dense.retrieve(RetrievalQuery(text="api keys roles"))

    assert [i.chunk_id for i in first.results] == [i.chunk_id for i in second.results]
    assert [i.score for i in first.results] == [i.score for i in second.results]


def test_the_candidate_limit_is_configurable(dense):
    """The channel returns at most the requested number of candidates."""
    response = dense.retrieve(RetrievalQuery(text="api keys roles invoices", limit=10, dense_limit=1))

    assert len(response.results) == 1


def test_the_channel_reports_its_capabilities(dense):
    """The dense channel reports what it can do."""
    capabilities = dense.capabilities

    assert capabilities.dense is True
    assert capabilities.lexical is False
    assert capabilities.supports_filtering


def test_the_required_filter_operations_are_collected():
    """Only the field operations are collected."""
    tree = and_(eq("chunk_level", "child"), is_in("parent_id", ["p0"]))

    assert required_filter_ops(tree) == {FilterOp.EQ, FilterOp.IN}
    assert filter_fields(tree) == {"chunk_level", "parent_id"}
    assert required_filter_ops(None) == set()
