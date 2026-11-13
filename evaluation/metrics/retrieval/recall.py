from typing import List, Dict, Any, Set
from evaluation.contracts.models import RetrievalResult, EvalSample


def recall_at_k(retrieved: List[RetrievalResult], relevant_ids: Set[str], k: int) -> float:
    """Recall@K: fraction of relevant items in top-K retrieved."""
    if not retrieved or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    retrieved_ids = {item.doc_id for item in top_k}
    relevant_retrieved = retrieved_ids & relevant_ids

    if not relevant_ids:
        return 0.0

    return len(relevant_retrieved) / len(relevant_ids)


def recall_at_k_chunk(retrieved: List[RetrievalResult], relevant_chunk_ids: Set[str], k: int) -> float:
    """Recall@K at chunk level."""
    return recall_at_k(retrieved, relevant_chunk_ids, k)


def recall_at_1(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Recall@1."""
    return recall_at_k(retrieved, relevant_ids, 1)


def recall_at_3(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Recall@3."""
    return recall_at_k(retrieved, relevant_ids, 3)


def recall_at_5(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Recall@5."""
    return recall_at_k(retrieved, relevant_ids, 5)


def recall_at_10(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Recall@10."""
    return recall_at_k(retrieved, relevant_ids, 10)


def precision_at_k(retrieved: List[RetrievalResult], relevant_ids: Set[str], k: int) -> float:
    """Precision@K: fraction of retrieved items in top-K that are relevant."""
    if not retrieved or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    retrieved_ids = {item.doc_id for item in top_k}
    relevant_retrieved = retrieved_ids & relevant_ids

    return len(relevant_retrieved) / k


def precision_at_1(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Precision@1."""
    return precision_at_k(retrieved, relevant_ids, 1)


def precision_at_3(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Precision@3."""
    return precision_at_k(retrieved, relevant_ids, 3)


def precision_at_5(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Precision@5."""
    return precision_at_k(retrieved, relevant_ids, 5)


def precision_at_10(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Precision@10."""
    return precision_at_k(retrieved, relevant_ids, 10)


def mean_average_precision(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Mean Average Precision (MAP)."""
    if not retrieved or not relevant_ids:
        return 0.0

    cumulative_relevant = 0.0
    precision_sum = 0.0

    for i, item in enumerate(retrieved):
        if i >= len(retrieved):
            break
        if item.doc_id in relevant_ids:
            cumulative_relevant += 1
            precision_sum += cumulative_relevant / (i + 1)

    if not relevant_ids:
        return 0.0

    return precision_sum / len(relevant_ids)


def mean_reciprocal_rank(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """Mean Reciprocal Rank (MRR)."""
    if not retrieved:
        return 0.0

    for i, item in enumerate(retrieved):
        if item.doc_id in relevant_ids:
            return 1.0 / (i + 1)

    return 0.0


def ndcg_at_k(retrieved: List[RetrievalResult], relevant_ids: Set[str], k: int) -> float:
    """nDCG@K: normalized discounted cumulative gain."""
    if not retrieved or k <= 0 or not relevant_ids:
        return 0.0

    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item.doc_id in relevant_ids:
            # relevance grade: 1 for relevant, 0 for irrelevant
            relevance = 1.0
            dcg += relevance / __import__('math').log2(i + 2)

    # Compute IDCG (ideal DCG)
    ideal_retrieved = sorted(retrieved, key=lambda x: x.rank)[:k]
    idcg = 0.0
    for i, item in enumerate(ideal_retrieved[:k]):
        if item.doc_id in relevant_ids:
            relevance = 1.0
            idcg += relevance / __import__('math').log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def ndcg_at_1(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """nDCG@1."""
    return ndcg_at_k(retrieved, relevant_ids, 1)


def ndcg_at_3(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """nDCG@3."""
    return ndcg_at_k(retrieved, relevant_ids, 3)


def ndcg_at_5(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """nDCG@5."""
    return ndcg_at_k(retrieved, relevant_ids, 5)


def ndcg_at_10(retrieved: List[RetrievalResult], relevant_ids: Set[str]) -> float:
    """nDCG@10."""
    return ndcg_at_k(retrieved, relevant_ids, 10)


def candidate_reduction_ratio(candidates: List[RetrievalResult],
                              after_rerank: List[RetrievalResult]) -> float:
    """Compute candidate reduction ratio after reranking."""
    if not candidates:
        return 0.0
    return len(after_rerank) / len(candidates)