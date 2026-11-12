"""Dense retrieval: embedding + filtered vector search.

The dense retriever is the shared, reusable piece that every dense-backed
strategy builds on.  It embeds the query with an injected model and runs a
filtered search against a collection, checking the backend's capabilities
before it searches.

:mod:`retrieval.dense.qwen` binds it to the production encoder.
"""

from retrieval.dense.retriever import (
    DenseRetriever,
    filter_fields,
    required_filter_ops,
)

__all__ = ["DenseRetriever", "filter_fields", "required_filter_ops"]
