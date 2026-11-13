from typing import List, Dict, Any, Set, Optional
from evaluation.contracts.models import RetrievalResult, EvalSample


def retrieval_metrics(
    retrieved: List[RetrievalResult],
    relevant_ids: Set[str],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute multiple retrieval metrics."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {}

    for k in k_values:
        results[f"recall_at_{k}"] = recall_at_k(retrieved, relevant_ids, k)
        results[f"precision_at_{k}"] = precision_at_k(retrieved, relevant_ids, k)
        results[f"ndcg_at_{k}"] = ndcg_at_k(retrieved, relevant_ids, k)

    results["map"] = mean_average_precision(retrieved, relevant_ids)
    results["mrr"] = mean_reciprocal_rank(retrieved, relevant_ids)

    return results


def hierarchical_recall_at_k(
    parent_results: List[RetrievalResult],
    child_results: List[RetrievalResult],
    relevant_parent_ids: Set[str],
    relevant_child_ids: Set[str],
    k_parent: int = None,
    k_child: int = None,
) -> Dict[str, float]:
    """Compute hierarchical retrieval metrics."""
    if k_parent is None:
        k_parent = 3
    if k_child is None:
        k_child = 5

    results = {}

    # Parent Recall@K
    parent_top_k = parent_results[:k_parent] if parent_results else []
    parent_relevant = {id for id in relevant_parent_ids}
    results["parent_recall_at_k"] = recall_at_k(parent_top_k, parent_relevant, k_parent)

    # Child Recall@K
    child_top_k = child_results[:k_child] if child_results else []
    child_relevant = {id for id in relevant_child_ids}
    results["child_recall_at_k"] = recall_at_k(child_top_k, child_relevant, k_child)

    # Candidate reduction ratio
    results["candidate_reduction_ratio"] = candidate_reduction_ratio(
        parent_results, child_results
    )

    return results