"""The dense channel of the retrieval engine, backed by Qwen3-Embedding-0.6B.

This package is the *production* dense channel of the ecosystem.  It binds the
generic :class:`~retrieval.dense.retriever.DenseRetriever` to the Qwen
encoder without ever loading it: the model is injected, so the weights are
never downloaded and the adapter stays a lazy import.

The only thing this package adds over the generic dense retriever is the
knowledge of *which* model is expected, so a collection that was built with
another encoder is reported before a query is embedded.
"""

from __future__ import annotations

from retrieval.dense.qwen.channel import (
    QWEN_MODEL_ID,
    QwenDenseRetriever,
    qwen_dense_retriever,
)

__all__ = ["QWEN_MODEL_ID", "QwenDenseRetriever", "qwen_dense_retriever"]
