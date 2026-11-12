"""Shared data models for the reranking ecosystem.

These models are dependency-free: they contain no imports of PyTorch, Transformers,
or Sentence Transformers. They describe the candidates, requests, and responses
that flow through the reranking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RerankScoreInterpretation(str, Enum):
    """How to interpret the raw relevance scores produced by the model."""

    RAW = "raw"
    """Raw cross-encoder logits; higher is more relevant. Not necessarily in [0, 1]."""
    SIGMOID = "sigmoid"
    """Transformed probability via sigmoid, in (0, 1)."""


@dataclass(frozen=True)
class RerankCandidate:
    """One candidate passed to the reranker for scoring.

    Attributes:
        candidate_id: a stable identifier for this candidate. Must be unique
            within a single rerank request.
        text: the chunk/text content to score against the query.
        original_rank: the 1-based rank of this candidate before reranking.
            Used as a deterministic tie-breaker when scores are equal.
        metadata: optional metadata dict that is preserved through reranking.
        source_provenance: optional provenance information (e.g. page/block) from
            the original retrieval result.
        retrieval_result: optional opaque reference to the original retrieval
            result that produced this candidate. The original result must remain
            recoverable after reranking.
    """

    candidate_id: str
    text: str
    original_rank: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    source_provenance: Any | None = None
    retrieval_result: Any | None = None


@dataclass(frozen=True)
class RerankRequest:
    """A reranking request from a query to a list of candidates.

    Attributes:
        query: the user query text.
        candidates: ordered list of candidates to re-rank.
        top_k: optional limit on the number of results to return. When None,
            all candidates are returned.
        task_instruction: optional task-specific instruction/prompt that
            overrides the model's default query instruction.
    """

    query: str
    candidates: tuple[RerankCandidate, ...]
    top_k: int | None = None
    task_instruction: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("the query text must be a non-empty string")
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple of RerankCandidate")
        if len(self.candidates) == 0:
            raise ValueError("must have at least one candidate")
        if self.top_k is not None:
            if not isinstance(self.top_k, int) or self.top_k < 1:
                raise ValueError(f"top_k must be a positive integer or None, got {self.top_k!r}")
            if self.top_k > len(self.candidates):
                raise ValueError(
                    f"top_k ({self.top_k}) cannot exceed the number of candidates "
                    f"({len(self.candidates)})"
                )


@dataclass(frozen=True)
class RerankedItem:
    """One candidate after reranking, preserving its original identity.

    Attributes:
        candidate: the original rerank candidate that was scored.
        raw_score: the raw relevance score from the model (higher = more relevant).
        original_rank: the candidate's rank before reranking (1-based).
        new_rank: the candidate's 1-based rank in the reranked result (1 = most relevant).
    """

    candidate: RerankCandidate
    raw_score: float
    original_rank: int
    new_rank: int


@dataclass(frozen=True)
class RerankResponse:
    """The response of a reranker: ordered reranked results plus diagnostics.

    Attributes:
        reranked_items: the candidates ordered by relevance (best first).
        model_id: the identity of the reranker model.
        model_revision: the model revision / tag, when available.
        candidates_scored: how many candidates were scored.
        elapsed_ms: optional wall-clock time for the scoring operation.
        score_interpretation: how to interpret the raw scores.
        diagnostics: additional diagnostics about the reranking operation.
    """

    reranked_items: tuple[RerankedItem, ...]
    model_id: str
    model_revision: str | None = None
    candidates_scored: int = 0
    elapsed_ms: float | None = None
    score_interpretation: RerankScoreInterpretation = RerankScoreInterpretation.RAW
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RerankerProtocol:
    """Convenience subclass that provides sensible defaults for the Reranker protocol.

    This is not the protocol itself (that is :class:`~reranking.contracts.models.Reranker`,
    which is a :class:`typing.Protocol`), but a concrete mixin that can help
    implementations satisfy the protocol contract.
    """

    model_id: str = "Qwen/Qwen3-Reranker-0.6B"
    """Default model identity."""

    model_revision: str | None = None
    """Model revision / tag when available."""

    default_task_instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    """Default task instruction used when no ``task_instruction`` is supplied."""

    def _resolve_task_instruction(self, task_instruction: str | None) -> str:
        """Return the effective task instruction, falling back to the default."""
        return task_instruction if task_instruction is not None else self.default_task_instruction


@dataclass(frozen=True)
class RerankDiagnostics:
    """Non-score diagnostics a reranker reports alongside its results."""

    candidates_scored: int = 0
    """How many candidates were actually scored."""

    batch_count: int = 0
    """Number of inference batches executed."""

    model_loading_errors: tuple[str, ...] = ()
    """Errors encountered while loading the model, if any."""

    shape_mismatch_errors: tuple[str, ...] = ()
    """Errors related to output shape / dimension inconsistencies."""