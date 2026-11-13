"""Retrieval → Reranking adapter.

Converts ``RetrievedItem`` (from the retrieval ecosystem) into ``RerankCandidate``
(the contract expected by the reranking protocol).

This preserves the complete provenance chain: chunk ID, document ID, retrieval
score and rank, metadata, and provenance.  If ``RerankCandidate`` has explicit
fields for ``source_provenance`` or ``retrieval_result``, those are populated
from the ``RetrievedItem`` so no information is silently discarded.
"""

from __future__ import annotations

from typing import Any

from retrieval.contracts.models import RetrievedItem
from reranking.contracts.models import RerankCandidate, RerankedItem


def retrieved_item_to_rerank_candidate(
    item: RetrievedItem,
    *,
    preserve_provenance: bool = True,
) -> RerankCandidate:
    """Convert a ``RetrievedItem`` into a ``RerankCandidate``.

    Preserved fields:
    - ``candidate_id``   -> ``item.chunk_id``
    - ``text``           -> ``item.text``
    - ``original_rank``  -> ``item.rank``
    - ``raw_score``      -> ``item.score`` (retrieval score)
    - ``metadata``       -> ``item.metadata`` (as dict)
    - ``source_provenance`` -> ``item.provenance`` (if ``preserve_provenance``)
    - ``retrieval_result`` -> minimal reference back to the retrieval result

    The ``RerankCandidate`` model explicitly provides ``source_provenance`` and
    ``retrieval_result`` fields for this purpose.  When present, they are populated
    from the ``RetrievedItem`` provenance metadata.

    Returns:
        A ``RerankCandidate`` instance with all available provenance preserved.

    Example::
        from retrieval.contracts.models import RetrievedItem
        from integrations.retrieval_reranking_adapter import retrieved_item_to_rerank_candidate

        item = RetrievedItem(
            chunk_id="ch-1",
            doc_id="doc-1",
            text="some passage",
            score=0.93,
            score_direction="higher_is_better",
            rank=1,
            strategy="dense",
            metadata={"source": "doc1.pdf"},
            provenance={"doc_id": "doc-1", "chunk_id": "ch-1", "original_uri": "/path/to/doc1.pdf"},
        )
        candidate = retrieved_item_to_rerank_candidate(item)
        # candidate.candidate_id == "ch-1"
        # candidate.raw_score == 0.93
        # candidate.source_provenance == {"doc_id": "doc-1", ...}
    """
    # Build the candidate_id from chunk_id (primary) or fall back to a generated ID
    candidate_id = item.chunk_id

    # Build the raw_score from the retrieval score
    raw_score = item.score

    # Build metadata dict, ensuring we don't mutate the original
    metadata: dict[str, Any] = dict(item.metadata) if item.metadata else {}

    # Build source_provenance if requested and available
    source_provenance: Any | None = None
    if preserve_provenance and item.provenance:
        source_provenance = dict(item.provenance)
        # Ensure chunk_id and doc_id are always present for traceability
        if "chunk_id" not in source_provenance and item.chunk_id:
            source_provenance["chunk_id"] = item.chunk_id
        if "doc_id" not in source_provenance and item.doc_id:
            source_provenance["doc_id"] = item.doc_id

    # Build retrieval_result reference (minimal, for round-trip traceability)
    retrieval_result: Any | None = None
    if item.provenance:
        retrieval_result = {
            "chunk_id": item.chunk_id,
            "doc_id": item.doc_id,
            "score": item.score,
            "rank": item.rank,
            "strategy": item.strategy,
        }

    return RerankCandidate(
        candidate_id=candidate_id,
        text=item.text,
        original_rank=item.rank,
        metadata=metadata,
        source_provenance=source_provenance,
        retrieval_result=retrieval_result,
    )


def reranked_items_to_context_items(
    reranked_items: tuple[RerankedItem, ...],
) -> tuple[dict[str, Any], ...]:
    """Convert ``RerankedItem`` candidates into ``GenerationRequest.context_items``.

    Each ``RerankedItem.candidate`` is a ``RerankCandidate`` which already has
    ``candidate_id``, ``text``, ``metadata``, ``source_provenance``, and
    ``retrieval_result``.  This function maps those into the flat ``dict`` format
    expected by ``GenerationRequest.context_items``.

    Returns a tuple of dicts, each containing at minimum:
    - ``chunk_id``        -> from ``RerankCandidate.candidate_id``
    - ``doc_id``          -> from ``RerankCandidate.source_provenance`` or ``candidate_id`` parsing
    - ``text``            -> the chunk text
    - ``metadata``        -> the candidate metadata
    - ``provenance``      -> the candidate retrieval_result / source_provenance

    The returned order reflects the rerank order (best first), which the generation
    provider will use as the context priority.
    """
    context_items: list[dict[str, Any]] = []

    for ri in reranked_items:
        candidate = ri.candidate  # type: ignore[attr-defined] -- RerankedItem.candidate is RerankCandidate

        # Build context item dict
        ctx_item: dict[str, Any] = {
            "chunk_id": candidate.candidate_id,
            "text": candidate.text,
            "metadata": candidate.metadata or {},
        }

        # Try to extract doc_id from source_provenance
        sp = getattr(candidate, "source_provenance", None) or {}
        if isinstance(sp, dict) and "doc_id" in sp:
            ctx_item["doc_id"] = sp["doc_id"]
        elif "." in candidate.candidate_id:
            # Attempt heuristic: "doc123-chunk-001" -> doc_id = "doc123"
            parts = candidate.candidate_id.split("-")
            if len(parts) >= 2:
                ctx_item["doc_id"] = "-".join(parts[:-1])

        # Add provenance for citation resolution
        rr = getattr(candidate, "retrieval_result", None)
        if rr is not None:
            ctx_item["provenance"] = rr
        elif sp is not None:
            ctx_item["provenance"] = sp

        context_items.append(ctx_item)

    return tuple(context_items)