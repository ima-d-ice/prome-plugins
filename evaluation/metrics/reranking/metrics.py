from typing import List, Dict, Any, Optional, Set
from evaluation.contracts.models import RerankCandidate, RerankResponse, RetrievalResult
from evaluation.metrics.retrieval.recall import (
    recall_at_k, precision_at_k, mean_reciprocal_rank,
    ndcg_at_k, candidate_reduction_ratio
)


def reranking_metrics(
    before_rerank: List[RerankCandidate],
    after_rerank: List[RerankCandidate],
    relevant_ids: Set[str],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute reranking improvement metrics."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {}

    # Convert to RetrievalResult format for reuse
    before_results = [
        RetrievalResult(
            chunk_id=c.candidate_id,
            doc_id=c.candidate_id,
            text=c.text,
            rank=c.original_rank,
            score=c.raw_score,
            strategy=RetrievalStrategy.HYBRID_RRF,
            metadata={"original_rank": c.original_rank},
        )
        for c in before_rerank
    ]

    after_results = [
        RetrievalResult(
            chunk_id=c.candidate_id,
            doc_id=c.candidate_id,
            text=c.text,
            rank=c.new_rank,
            score=c.raw_score,
            strategy=RetrievalStrategy.HYBRID_RRF,
            metadata={"new_rank": c.new_rank},
        )
        for c in after_rerank
    ]

    # Metrics before reranking
    before = retrieval_metrics(before_results, relevant_ids, k_values)
    results.update({f"before_{k}": v for k, v in before.items()})

    # Metrics after reranking
    after = retrieval_metrics(after_results, relevant_ids, k_values)
    results.update({f"after_{k}": v for k, v in after.items()})

    # Candidate reduction ratio
    results["candidate_reduction_ratio"] = candidate_reduction_ratio(
        before_results, after_results
    )

    # Improvement deltas
    for k in k_values:
        before_recall = before.get(f"recall_at_{k}", 0.0)
        after_recall = after.get(f"recall_at_{k}", 0.0)
        results[f"recall_at_{k}_improvement"] = after_recall - before_recall

        before_precision = before.get(f"precision_at_{k}", 0.0)
        after_precision = after.get(f"precision_at_{k}", 0.0)
        results[f"precision_at_{k}_improvement"] = after_precision - before_precision

    return results


def before_after_comparison(
    before: List[RerankCandidate],
    after: List[RerankCandidate],
    relevant_ids: Set[str],
) -> Dict[str, Any]:
    """Full before-after comparison for reranking."""
    metrics = reranking_metrics(before, after, relevant_ids)
    return {
        "before_metrics": {k: v for k, v in metrics.items() if not k.endswith("_improvement")},
        "improvements": {k: v for k, v in metrics.items() if k.endswith("_improvement")},
        "candidate_recall_before": len(before) / max(len(relevant_ids), 1),
        "candidate_recall_after": len(after) / max(len(relevant_ids), 1),
    }