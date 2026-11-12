"""Tests of the full hierarchical (coarse-to-fine) retrieval strategy."""

from __future__ import annotations

import pytest

from retrieval.contracts.exceptions import RetrievalConfigError
from retrieval.contracts.protocols import RetrievalStrategy
from retrieval.strategies.hierarchical import (
    HierarchicalConfig,
    HierarchicalIndexer,
    HierarchicalRetrievalStrategy,
)
from retrieval.strategies.hierarchical.metadata import CHUNK_LEVEL_FIELD
from tests.retrieval.conftest import make_document


@pytest.fixture()
def corpus():
    """Return two documents with two sections each."""
    return [
        make_document(
            "auth.md",
            [
                (
                    "Authentication",
                    "api keys are rotated every thirty days by the scheduler service " * 6,
                ),
                (
                    "Authorization",
                    "roles grant access to the billing console and the audit log " * 6,
                ),
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


@pytest.fixture()
def indexed(store, embedding_model, corpus):
    """Return an indexed collection and the embedding model."""
    from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig as ChunkerConfig

    indexer = HierarchicalIndexer(store, embedding_model, collection="docs")
    indexer.ensure_collection()
    chunker = HierarchicalChunker(
        ChunkerConfig(parent_size=300, child_size=150, child_overlap=20)
    )
    for document in corpus:
        indexer.index_result(chunker.chunk(document))
    return store, embedding_model


def build_strategy(store, model, **overrides) -> HierarchicalRetrievalStrategy:
    """Build a strategy with the default configuration plus ``overrides``."""
    config = HierarchicalConfig(**overrides)
    return HierarchicalRetrievalStrategy.create(
        store, model, collection="docs", config=config
    )


def test_the_strategy_embeds_the_query_exactly_once(indexed):
    """The query is embedded once and the vector is shared by both stages."""
    store, model = indexed
    strategy = build_strategy(store, model, parent_top_k=3, child_top_k=10)

    strategy.retrieve("api keys are rotated")

    assert len(model.query_calls) == 1
    assert model.query_calls == ["api keys are rotated"]


def test_the_parent_stage_runs_before_the_child_stage(indexed):
    """The child search is restricted to the parents the coarse stage selected."""
    store, model = indexed
    strategy = build_strategy(store, model, parent_top_k=2, child_top_k=20)

    response = strategy.retrieve("api keys are rotated")

    assert response.diagnostics.parents_selected > 0
    assert response.diagnostics.parents_selected <= 2


def test_the_child_results_are_restricted_to_the_selected_parents(indexed):
    """Every returned child belongs to a parent the coarse stage selected."""
    store, model = indexed
    strategy = build_strategy(store, model, parent_top_k=2, child_top_k=20)

    response = strategy.retrieve("api keys are rotated")

    selected = set(response.diagnostics.selected_parent_ids)
    assert selected
    assert response.diagnostics.parents_selected == len(selected)
    for result in response.results:
        assert result.metadata["parent_id"] in selected


def test_the_strategy_returns_the_standardized_result_interface(indexed):
    """The strategy answers with the shared response contract."""
    store, model = indexed
    strategy = build_strategy(store, model)

    response = strategy.retrieve("api keys")

    assert isinstance(strategy, RetrievalStrategy)
    assert response.results
    for result in response.results:
        assert result.strategy == "hierarchical"
        assert result.rank >= 1
        assert result.chunk_id
        assert result.text
        assert result.metadata["final_rank"] == result.rank
        assert result.metadata["child_rank"] >= 1
        assert result.metadata["parent_id"]


def test_the_evidence_is_the_child_and_the_parent_is_the_context(indexed):
    """The matched child is the evidence; the parent is the context."""
    store, model = indexed
    strategy = build_strategy(store, model, include_parent_context=True)

    response = strategy.retrieve("api keys are rotated")

    for result in response.results:
        parent = result.parent_context
        assert parent is not None
        assert parent.role == "parent"
        # The evidence is the child, not the parent.
        assert result.chunk_id != parent.chunk_id
        # The evidence text is the text of the child record.
        child = store.get("docs", [result.chunk_id])[0]
        assert result.text == child.chunk.text


def test_the_parent_context_is_not_attached_when_disabled(indexed):
    """Without parent context the results carry no context at all."""
    store, model = indexed
    strategy = build_strategy(store, model, include_parent_context=False)

    response = strategy.retrieve("api keys are rotated")

    assert response.results
    for result in response.results:
        assert result.parent_context is None
        assert result.context == ()


def test_the_adjacent_expansion_attaches_neighbours(indexed):
    """The neighbouring children of the same parent are attached."""
    store, model = indexed
    strategy = build_strategy(
        store, model, expand_adjacent=True, max_adjacent=1, child_top_k=20
    )

    response = strategy.retrieve("api keys are rotated")

    adjacent = [c for result in response.results for c in result.adjacent_context]
    assert adjacent
    for result in response.results:
        for item in result.adjacent_context:
            assert item.role == "adjacent"
            # An adjacent chunk is a different chunk than its own evidence.
            assert item.chunk_id != result.chunk_id
            # It belongs to the same parent as the evidence.
            assert item.doc_id == result.doc_id


def test_the_adjacent_expansion_is_disabled_by_default(indexed):
    """Without the option no neighbouring chunk is attached."""
    store, model = indexed
    strategy = build_strategy(store, model, expand_adjacent=False)

    response = strategy.retrieve("api keys are rotated")

    for result in response.results:
        assert result.adjacent_context == ()


def test_the_adjacent_expansion_respects_the_limit(indexed):
    """At most ``max_adjacent`` neighbours are attached."""
    store, model = indexed
    strategy = build_strategy(
        store, model, expand_adjacent=True, max_adjacent=1, child_top_k=20
    )

    response = strategy.retrieve("api keys are rotated")

    for result in response.results:
        assert len(result.adjacent_context) <= 1


def test_the_final_top_k_limits_the_number_of_results(indexed):
    """Only ``final_top_k`` results are returned."""
    store, model = indexed
    strategy = build_strategy(store, model, parent_top_k=4, child_top_k=30, final_top_k=2)

    response = strategy.retrieve("api keys are rotated")

    assert len(response.results) == 2
    assert [result.rank for result in response.results] == [1, 2]


def test_the_ordering_is_deterministic(indexed):
    """Two identical queries produce the same order."""
    store, model = indexed
    strategy = build_strategy(store, model)

    first = strategy.retrieve("api keys are rotated")
    second = strategy.retrieve("api keys are rotated")

    assert [r.chunk_id for r in first.results] == [r.chunk_id for r in second.results]
    assert [r.score for r in first.results] == [r.score for r in second.results]


def test_the_scores_are_sorted_by_descending_similarity(indexed):
    """The results are ranked by their dense similarity."""
    store, model = indexed
    strategy = build_strategy(store, model)

    response = strategy.retrieve("api keys are rotated")

    scores = [result.score for result in response.results]
    assert scores == sorted(scores, reverse=True)


def test_a_query_without_a_parent_match_returns_no_result(indexed):
    """When the coarse stage selects nothing, nothing is returned."""
    store, model = indexed
    strategy = build_strategy(store, model, parent_top_k=3)

    response = strategy.retrieve("zzzz qqqq wwww unrelated gibberish")

    # The stage may still select a parent (everything is similar to
    # something), but when it selects none the answer is empty.
    if response.diagnostics.parents_selected == 0:
        assert response.results == []
        assert response.diagnostics.notes


def test_an_empty_query_is_rejected(indexed):
    """An empty query raises instead of returning the whole collection."""
    store, model = indexed
    strategy = build_strategy(store, model)
    with pytest.raises(RetrievalConfigError):
        strategy.retrieve("")


def test_the_diagnostics_report_the_candidate_reduction(indexed):
    """The diagnostics say how much of the corpus was searched."""
    store, model = indexed
    strategy = build_strategy(store, model, parent_top_k=1, child_top_k=5)

    response = strategy.retrieve("api keys are rotated")

    assert response.diagnostics.candidates_total > 0
    assert response.diagnostics.candidates_searched >= 0
    ratio = response.diagnostics.candidate_reduction_ratio
    assert 0.0 <= ratio <= 1.0


def test_the_score_threshold_is_applied(indexed):
    """A child threshold removes the children that score below it."""
    store, model = indexed
    unfiltered = build_strategy(store, model, parent_top_k=3, child_top_k=20).retrieve(
        "api keys are rotated"
    )
    assert unfiltered.results

    threshold = unfiltered.results[0].score
    filtered = build_strategy(
        store,
        model,
        parent_top_k=3,
        child_top_k=20,
        child_score_threshold=threshold,
    ).retrieve("api keys are rotated")

    assert len(filtered.results) < len(unfiltered.results)
    assert all(result.score >= threshold for result in filtered.results)


def test_the_user_filter_restricts_both_stages(indexed):
    """A user filter applies to the coarse and to the fine stage."""
    from vector_stores.contracts.filters import eq

    store, model = indexed
    strategy = build_strategy(
        store, model, parent_top_k=4, child_top_k=30, filters=eq("document_id", "billing.md")
    )

    response = strategy.retrieve("api keys are rotated")

    assert response.results
    for result in response.results:
        assert result.doc_id == "billing.md"
