"""Prome RAG Reranking Ecosystem.

Modular, plug-and-play reranking component for reordering retrieved candidates.

The only production reranker is :class:`~reranking.qwen3.reranker.Qwen3Reranker`,
which uses the Qwen/Qwen3-Reranker-0.6B model via Sentence Transformers'
CrossEncoder interface.

Typical usage:

    from reranking import Qwen3Reranker

    reranker = Qwen3Reranker()
    request = RerankRequest(
        query="What is quantum computing?",
        candidates=[chunk1, chunk2, ...],
    )
    response = reranker.rerank(request)
    reranked = response.reranked_items
"""

from __future__ import annotations

from reranking.contracts.protocols import Reranker
from reranking.qwen3.reranker import Qwen3Reranker

__all__ = [
    "Qwen3Reranker",
    "Reranker",
]