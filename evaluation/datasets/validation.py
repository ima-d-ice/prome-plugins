import json
from typing import List, Optional, Dict, Any, Set
from evaluation.datasets.schema import EvalSample, validate_dataset, hash_dataset
from evaluation.datasets.conversion import sample_json
from evaluation.contracts.models import (
    ExperimentConfig, ExperimentResult, RetrievalStrategy, MetricProfile,
)


def default_retrieval_profile() -> MetricProfile:
    """Return the default retrieval quality profile."""
    from evaluation.contracts.models import RETRIEVAL_QUALITY
    return RETRIEVAL_QUALITY


def default_reranking_profile() -> MetricProfile:
    """Return the default reranking quality profile."""
    from evaluation.contracts.models import RERANKING_QUALITY
    return RERANKING_QUALITY


def default_generation_profile() -> MetricProfile:
    """Return the default generation quality profile."""
    from evaluation.contracts.models import GENERATION_QUALITY
    return GENERATION_QUALITY


def default_full_rag_profile() -> MetricProfile:
    """Return the default full RAG quality profile."""
    from evaluation.contracts.models import FULL_RAG_QUALITY
    return FULL_RAG_QUALITY


def create_experiment_config(
    experiment_id: str,
    dataset_id: str,
    retrieval_strategy: RetrievalStrategy,
    **kwargs,
) -> ExperimentConfig:
    """Create an ExperimentConfig with default values."""
    return ExperimentConfig(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        retrieval_strategy=retrieval_strategy,
        **kwargs,
    )


def sample_has_reference(sample: EvalSample) -> bool:
    """Check if a sample has a reference answer."""
    return sample.reference is not None and sample.reference.strip() != ""


def sample_has_contexts(sample: EvalSample) -> bool:
    """Check if a sample has reference contexts."""
    return sample.reference_contexts is not None and len(sample.reference_contexts) > 0


def sample_has_retrieved_contexts(sample: EvalSample) -> bool:
    """Check if a sample has retrieved contexts."""
    return sample.retrieved_contexts is not None and len(sample.retrieved_contexts) > 0


def compute_dataset_hash(samples: List[EvalSample]) -> str:
    """Compute hash for dataset integrity."""
    return hash_dataset(samples)


def merge_samples(
    base_samples: List[EvalSample],
    additional_samples: List[EvalSample],
) -> List[EvalSample]:
    """Merge two sample lists, using sample_id to deduplicate."""
    seen: Set[str] = {s.sample_id for s in base_samples}
    result = list(base_samples)
    for sample in additional_samples:
        if sample.sample_id not in seen:
            result.append(sample)
            seen.add(sample.sample_id)
    return result


def filter_samples_by_strategy(
    samples: List[EvalSample],
    strategy: RetrievalStrategy,
) -> List[EvalSample]:
    """Filter samples by retrieval strategy."""
    return [s for s in samples if s.strategy == strategy or s.strategy is None]


def filter_samples_with_generated_response(
    samples: List[EvalSample],
) -> List[EvalSample]:
    """Filter samples that have generated responses."""
    return [s for s in samples if s.generated_response is not None and s.generated_response.strip() != ""]


def filter_samples_with_reference(
    samples: List[EvalSample],
) -> List[EvalSample]:
    """Filter samples that have reference answers."""
    return [s for s in samples if sample_has_reference(s)]