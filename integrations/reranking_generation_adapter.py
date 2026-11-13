"""Reranking → Generation adapter.

Converts ``RerankedItem`` (from the reranking ecosystem) into
``GenerationRequest.context_items`` (the structure expected by the generation
protocol).

This adapter is the critical bridge between re-ranked candidates and the
generation provider's context budget.  It preserves the complete provenance
chain and maintains the rerank order (best-first), so that generation citations
can resolve to the original retrieved sources.
"""

from __future__ import annotations

from typing import Any

from reranking.contracts.models import RerankedItem, RerankCandidate

from integrations.retrieval_reranking_adapter import reranked_items_to_context_items


def reranked_to_generation_context(
    reranked_items: tuple[RerankedItem, ...],
) -> tuple[dict[str, Any], ...]:
    """Convert ``RerankedItem`` into ``GenerationRequest.context_items``.

    This is the primary adapter for feeding reranked evidence into a generation
    provider.

    Each ``RerankedItem`` carries a ``candidate`` which is a ``RerankCandidate``.
    The ``RerankCandidate`` already preserves ``candidate_id`` (chunk_id),
    ``text``, ``metadata``, ``source_provenance``, and ``retrieval_result``.

    This function maps those into the ``dict[str, Any]`` entries that
    ``GenerationRequest.context_items`` expects.  The dicts typically contain
    at minimum: ``chunk_id``, ``doc_id``, ``text``, ``metadata``, and ``provenance``.

    The returned tuple preserves the rerank order (index 0 = most relevant),
    which generation providers use for context budgeting and citation priority.

    Returns:
        ``tuple[dict[str, Any], ...]`` compatible with
        ``GenerationRequest.context_items``.

    Example::
        from reranking.qwen3.reranker import Qwen3Reranker
        from integrations.reranking_generation_adapter import reranked_to_generation_context

        reranker = Qwen3Reranker()
        request = RerankRequest(
            query="What is quantum computing?",
            candidates=[RerankCandidate(candidate_id="ch-1", text="...")],
        )
        response = reranker.rerank(request)

        # Convert to generation context
        context = reranked_to_generation_context(response.reranked_items)
        # context is tuple[dict, ...] compatible with GenerationRequest.context_items

        # Use in generation
        gen_req = GenerationRequest(
            user_input="Tell me about quantum computing",
            context_items=context,
        )
        response = ollama_provider.generate(gen_req)
    """
    return reranked_items_to_context_items(reranked_items)