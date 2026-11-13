from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class EvaluationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class RetrievalStrategy(Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RRF_BEFORE_RERANK = "hybrid_rrf_before_rerank"
    HYBRID_RRF_AFTER_RERANK = "hybrid_rrf_after_rerank"
    HIERARCHICAL_FLAT = "hierarchical_flat"
    HIERARCHICAL_PARENT_CHILD = "hierarchical_parent_child"
    HIERARCHICAL_BEFORE_RERANK = "hierarchical_before_rerank"
    HIERARCHICAL_AFTER_RERANK = "hierarchical_after_rerank"


class RerankingQuality(Enum):
    BEFORE = "before"
    AFTER = "after"


@dataclass
class SampleID:
    """Stable identifier for an evaluation sample."""
    value: str


@dataclass
class EvalSample:
    """Base evaluation sample with stable identity."""
    sample_id: str
    user_input: str
    reference: Optional[str] = None
    reference_contexts: Optional[List[str]] = None
    retrieved_contexts: Optional[List[str]] = None
    retrieved_context_ids: Optional[List[str]] = None
    reranked_contexts: Optional[List[str]] = None
    generated_response: Optional[str] = None
    cited_source_ids: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_version: Optional[str] = None
    strategy: Optional[RetrievalStrategy] = None


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    chunk_id: str
    doc_id: str
    text: str
    rank: int
    score: float
    strategy: RetrievalStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentConfig:
    """Configuration for an evaluation experiment."""
    experiment_id: str
    dataset_id: str
    retrieval_strategy: RetrievalStrategy
    reranker_on: bool = False
    generation_model: Optional[str] = None
    judge_provider: Optional[str] = None
    judge_model: Optional[str] = None
    judge_temperature: float = 0.01
    judge_max_tokens: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_revision: Optional[str] = None
    ragas_version: Optional[str] = None
    metric_names: List[str] = field(default_factory=list)
    prompt_metric_versions: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[str] = None


@dataclass
class ExperimentResult:
    """Result of an evaluation experiment."""
    experiment_id: str
    dataset_id: str
    retrieval_strategy: RetrievalStrategy
    reranker_on: bool
    generation_model: Optional[str]
    judge_provider: Optional[str]
    judge_model: Optional[str]
    ragas_version: str
    metric_names: List[str]
    sample_count: int
    failed_count: int
    skipped_count: int
    ragas_scores: Dict[str, float] = field(default_factory=dict)
    deterministic_scores: Dict[str, float] = field(default_factory=dict)
    per_sample_results: str = ""  # JSONL string
    summary_md: str = ""
    timestamp: Optional[str] = None


@dataclass
class MetricProfile:
    """Named evaluation metric profile."""
    name: str
    metrics: List[str]
    description: str


# Predefined metric profiles
RETRIEVAL_QUALITY = MetricProfile(
    name="retrieval_quality",
    metrics=["context_precision", "context_recall", "context_entity_recall"],
    description="Measures retrieved context quality",
)

RERANKING_QUALITY = MetricProfile(
    name="reranking_quality",
    metrics=["context_precision_before", "context_precision_after",
             "context_recall_before", "context_recall_after"],
    description="Measures reranking improvement",
)

GENERATION_QUALITY = MetricProfile(
    name="generation_quality",
    metrics=["faithfulness", "answer_relevancy", "factual_correctness",
             "noise_sensitivity"],
    description="Measures answer quality",
)

FULL_RAG_QUALITY = MetricProfile(
    name="full_rag_quality",
    metrics=["context_precision", "context_recall", "faithfulness",
             "answer_relevancy"],
    description="Full RAG pipeline evaluation",
)


@dataclass
class RerankCandidate:
    """A candidate document for reranking."""
    candidate_id: str
    text: str
    original_rank: int
    new_rank: Optional[int] = None
    raw_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResponse:
    """Response from a reranker."""
    reranked_items: List[RerankCandidate] = field(default_factory=list)
    model_id: str = ""
    model_revision: Optional[str] = None
    candidates_scored: int = 0
    elapsed_ms: float = 0.0
    score_interpretation: str = "RAW"