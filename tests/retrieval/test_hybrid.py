"""Tests of the hybrid retrieval strategy."""

from __future__ import annotations

import pytest

from embeddings.contracts.models import RepresentationType
from retrieval.contracts.exceptions import RetrievalConfigError
from retrieval.contracts.models import RetrievalQuery
from retrieval.contracts.protocols import Retriever
from retrieval.lexical.bm25.sqlite_fts5 import BM25Retriever, LexicalRecord, SQLiteFTS5Index
from retrieval.strategies.hybrid import (
    PARTIAL,
    STRICT,
    ChannelFailure,
    HybridConfig,
    HybridRetrievalStrategy,
)
from tests.retrieval.support import (
    FakeEmbeddingModel,
    FakeVectorStore,
    make_chunk,
)
from vector_stores.contracts.models import (
    CollectionConfig,
    DistanceMetric,
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)


def _build_store(store, model, chunks):
    """Index the chunks into the fake store."""
    space = VectorSpaceConfig(
        name="dense",
        vector_type=RepresentationType.DENSE,
        embedding_space=model.get_embedding_space(),
        dimensions=model.dimension,
        distance=DistanceMetric.COSINE,
    )
    store.create_collection(
        CollectionConfig(name="docs", vector_spaces={"dense": space})
    )
    for chunk in chunks:
        store.upsert(
            "docs",
            [
                StoredRecord(
                    chunk=chunk,
                    vectors={
                        "dense": SpaceVectors(
                            space="dense",
                            embedding_space=model.get_embedding_space(),
                            dense=model.vector(chunk.text),
                        )
                    },
                )
            ],
        )


def _build_index(index_path, model, chunks):
    """Write the chunks into the lexical index."""
    index = SQLiteFTS5Index(index_path)
    index.open()
    index.upsert([LexicalRecord.from_chunk(chunk) for chunk in chunks])
    return index


@pytest.fixture()
def corpus():
    """Return a two-document corpus and its embedding model."""
    model = FakeEmbeddingModel()
    chunks = [
        make_chunk(
            "c0", "auth.md",
            "api keys are rotated every thirty days by the rotation scheduler",
            index=0, count=2,
        ),
        make_chunk(
            "c1", "auth.md",
            "the rotation writes the new key to the key store and publishes "
            "it to the discovery endpoint",
            index=1, count=2,
        ),
        make_chunk(
            "c2", "billing.md",
            "invoices are issued on the first day of every month",
            index=0, count=2,
        ),
        make_chunk(
            "c3", "billing.md",
            "a refund is issued within five business days of an approved "
            "credit note",
            index=1, count=2,
        ),
    ]
    return model, chunks


@pytest.fixture()
def hybrid(tmp_path, corpus):
    """Return a hybrid strategy over an indexed corpus."""
    from retrieval.dense.qwen import QwenDenseRetriever

    model, chunks = corpus
    store = FakeVectorStore()
    _build_store(store, model, chunks)
    index = _build_index(tmp_path / "lexical.db", model, chunks)
    dense = QwenDenseRetriever(store, model, collection="docs", expected_model_id=None)
    lexical = BM25Retriever(index)
    strategy = HybridRetrievalStrategy(dense, lexical)
    return strategy, index, store, model


def test_the_hybrid_strategy_embeds_the_query_once(hybrid):
    """The dense channel embeds the query once; the lexical channel does not."""
    strategy, _, _, model = hybrid

    strategy.retrieve(RetrievalQuery(text="rotation", limit=3))

    assert len(model.query_calls) == 1
    assert model.query_calls == ["rotation"]


def test_the_hybrid_strategy_returns_the_requested_number_of_results(hybrid):
    """The final result limit is respected."""
    strategy, _, _, _ = hybrid

    response = strategy.retrieve(RetrievalQuery(text="rotation", limit=2))

    assert len(response.results) <= 2


def test_the_hybrid_strategy_is_a_retriever(hybrid):
    """The hybrid strategy satisfies the retriever protocol."""
    strategy, _, _, _ = hybrid

    assert isinstance(strategy, Retriever)


def test_the_fused_items_record_their_channels(hybrid):
    """The fused items record the channels that found them."""
    strategy, _, _, _ = hybrid

    response = strategy.retrieve(RetrievalQuery(text="rotation", limit=5))

    for item in response.results:
        assert set(item.methods) <= {"dense-qwen", "bm25"}
        for channel in item.methods:
            assert channel in item.channel_ranks
            assert channel in item.channel_scores


def test_a_chunk_found_by_both_channels_outranks_one_found_by_a_single_channel(hybrid):
    """A chunk both channels returned outranks a chunk only one returned."""
    strategy, _, _, _ = hybrid

    response = strategy.retrieve(RetrievalQuery(text="rotation", limit=5))

    assert response.results[0].chunk_id in {"c0", "c1"}


def test_the_candidate_limits_are_independent_of_the_result_limit(hybrid):
    """The candidate limits are configurable independently of the final limit."""
    strategy, _, _, _ = hybrid

    response = strategy.retrieve(
        RetrievalQuery(
            text="rotation", limit=1, dense_limit=2, lexical_limit=2
        )
    )

    assert len(response.results) == 1


def test_a_filter_excludes_passages_from_both_channels(hybrid):
    """A metadata filter excludes the same passages from both channels."""
    from vector_stores.contracts.filters import eq

    strategy, _, _, _ = hybrid

    response = strategy.retrieve(
        RetrievalQuery(text="rotation", limit=5, filters=eq("doc_id", "billing.md"))
    )

    assert all(item.doc_id == "billing.md" for item in response.results)


def test_the_strict_policy_raises_when_a_channel_fails(hybrid, monkeypatch):
    """The strict policy fails loudly when a channel fails."""
    strategy, _, _, _ = hybrid

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(strategy.dense, "retrieve", broken)

    with pytest.raises(ChannelFailure) as caught:
        strategy.retrieve(RetrievalQuery(text="rotation", limit=5))

    assert "dense-qwen" in str(caught.value)


def test_the_partial_policy_keeps_the_surviving_channel(hybrid, monkeypatch):
    """The partial policy continues with the surviving channel and reports it."""
    strategy, _, _, _ = hybrid

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(strategy.dense, "retrieve", broken)

    partial = HybridRetrievalStrategy(
        strategy.dense,
        strategy.lexical,
        config=HybridConfig(failure_policy=PARTIAL),
    )

    response = partial.retrieve(RetrievalQuery(text="rotation", limit=5))

    assert response.results
    assert "dense-qwen" in response.diagnostics.channel_failures
    assert response.diagnostics.channel_failures["dense-qwen"].startswith("RuntimeError")
    assert response.results[0].methods == ("bm25",)


def test_the_diagnostics_count_the_candidates_of_each_channel(hybrid):
    """The diagnostics count how many candidates each channel produced."""
    strategy, _, _, _ = hybrid

    response = strategy.retrieve(RetrievalQuery(text="rotation", limit=5))

    assert response.diagnostics.channel_candidates["dense-qwen"] >= 1
    assert response.diagnostics.channel_candidates["bm25"] >= 1


def test_an_empty_query_is_rejected(hybrid):
    """An empty query raises instead of returning the whole corpus."""
    strategy, _, _, _ = hybrid

    with pytest.raises(RetrievalConfigError):
        strategy.retrieve(RetrievalQuery(text="   "))


def test_the_default_policy_is_strict():
    """The default failure policy is strict."""
    assert STRICT == "strict"
    assert PARTIAL == "partial"
    assert HybridConfig().failure_policy == STRICT
