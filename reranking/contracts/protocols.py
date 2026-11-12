"""The reranking protocol.

A reranker turns a :class:`~reranking.contracts.models.RerankRequest`
into a :class:`~reranking.contracts.models.RerankResponse`. The contract is
deliberately small and independent of Qwen, Sentence Transformers, or any
other library: it describes the *interface* only.

Implementations must:

- validate the request and fail loudly on an empty query or invalid candidates;
- never modify the original candidate text or metadata;
- preserve the original candidate ID and retrieval result reference;
- report what they scored via :class:`~reranking.contracts.models.RerankDiagnostics`;
- return scores in the model's native scale (raw logits by default);
- support configurable task instructions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from reranking.contracts.models import RerankRequest, RerankResponse


@runtime_checkable
class Reranker(Protocol):
    """Interface every Prome reranker implements.

    The core operation is equivalent to:

        rerank(request) -> response

    Implementations must:

    - validate the request and fail loudly on an empty query or invalid candidates;
    - pair the query with each candidate and score the query--candidate pair
      using the underlying model (e.g. CrossEncoder);
    - return a :class:`~reranking.contracts.models.RerankResponse` with the
      candidates ordered by descending raw relevance score;
    - preserve the original candidate ID, text, metadata, and
      ``retrieval_result`` reference so the original result remains recoverable;
    - support configurable task instructions that override the model's default
      query prompt;
    - support lazy model initialization and configurable device placement.
    """

    model_id: str

    @property
    def model_revision(self) -> str | None:
        """Return the model revision / tag, when available."""
        ...

    @property
    def capabilities(self) -> dict[str, object]:
        """Return the capabilities of this reranker."""
        ...

    def rerank(self, request: RerankRequest) -> RerankResponse:
        """Re-rank candidates based on relevance to the query.

        Args:
            request: the reranking request containing the query and candidate list.

        Returns:
            the reranked response with candidates ordered by descending relevance.

        Raises:
            RerankConfigError: if the request is invalid.
            RerankDependencyError: if a required optional dependency is missing.
            RerankInferenceError: if the model fails during inference.
            UnsupportedDeviceError: if the requested device is not available.
        """
        ...