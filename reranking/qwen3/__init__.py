"""Qwen3-Reranker-0.6B reranking adapter.

Wraps Sentence Transformers' CrossEncoder interface for query--candidate
relevance scoring using the Qwen/Qwen3-Reranker-0.6B model.

Typical usage:

    from reranking.qwen3.reranker import Qwen3Reranker

    reranker = Qwen3Reranker(device="cpu")
    ...
"""

from __future__ import annotations

from reranking.qwen3.reranker import Qwen3Reranker

__all__ = [
    "Qwen3Reranker",
]