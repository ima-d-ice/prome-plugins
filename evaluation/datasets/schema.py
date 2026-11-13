from ragas.dataset_schema import EvaluationDataset, SingleTurnSample, MultiTurnSample
from evaluation.contracts.models import EvalSample, ExperimentConfig, ExperimentResult, MetricProfile, RetrievalStrategy
from typing import List, Optional, Dict, Any
import json
import hashlib
from datetime import datetime


def sample_id_from_text(text: str) -> str:
    """Generate a stable sample ID from text content."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_jsonl_dataset(filepath: str) -> List[EvalSample]:
    """Load EvalSample instances from a JSONL dataset file."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            sample = EvalSample(
                sample_id=data.get("sample_id", sample_id_from_text(json.dumps(data))),
                user_input=data.get("user_input", ""),
                reference=data.get("reference"),
                reference_contexts=data.get("reference_contexts"),
                retrieved_contexts=data.get("retrieved_contexts"),
                retrieved_context_ids=data.get("retrieved_context_ids"),
                reranked_contexts=data.get("reranked_contexts"),
                generated_response=data.get("generated_response"),
                cited_source_ids=data.get("cited_source_ids"),
                metadata=data.get("metadata", {}),
                dataset_version=data.get("dataset_version"),
                strategy=data.get("strategy"),
            )
            samples.append(sample)
    return samples


def validate_sample(sample: EvalSample) -> bool:
    """Validate an EvalSample instance."""
    if not sample.sample_id:
        raise DatasetValidationError("Sample must have a sample_id")
    if not sample.user_input or not sample.user_input.strip():
        raise DatasetValidationError("Sample must have a non-empty user_input")
    if sample.reference is not None and not isinstance(sample.reference, str):
        raise DatasetValidationError("reference must be a string or None")
    if sample.reference_contexts is not None:
        if not isinstance(sample.reference_contexts, list):
            raise DatasetValidationError("reference_contexts must be a list or None")
    if sample.retrieved_contexts is not None:
        if not isinstance(sample.retrieved_contexts, list):
            raise DatasetValidationError("retrieved_contexts must be a list or None")
    if sample.generated_response is not None and not isinstance(sample.generated_response, str):
        raise DatasetValidationError("generated_response must be a string or None")
    return True


def validate_dataset(samples: List[EvalSample]) -> bool:
    """Validate all samples in a dataset."""
    for i, sample in enumerate(samples):
        try:
            validate_sample(sample)
        except DatasetValidationError as e:
            raise DatasetValidationError(f"Sample {i} failed validation: {e}")
    return True


def sample_to_ragas_single_turn(sample: EvalSample) -> SingleTurnSample:
    """Convert an EvalSample to a Ragas SingleTurnSample."""
    return SingleTurnSample(
        user_input=sample.user_input,
        retrieved_contexts=sample.retrieved_contexts or [],
        reference_contexts=sample.reference_contexts or [],
        response=sample.generated_response,
        reference=sample.reference,
        retrieved_context_ids=sample.retrieved_context_ids,
        reference_context_ids=sample.reference_context_ids,
        metadata=sample.metadata,
    )


def samples_to_ragas_dataset(samples: List[EvalSample]) -> EvaluationDataset:
    """Convert a list of EvalSamples to a Ragas EvaluationDataset."""
    from ragas.dataset_schema import SingleTurnSample

    ragas_samples = [sample_to_ragas_single_turn(s) for s in samples]
    return EvaluationDataset.from_samples(ragas_samples)


def hash_dataset(samples: List[EvalSample]) -> str:
    """Compute a hash of the dataset for integrity checking."""
    content = "".join(
        s.sample_id + s.user_input + (s.reference or "")
        for s in sorted(samples, key=lambda x: x.sample_id)
    )
    return hashlib.sha256(content.encode()).hexdigest()


def detect_sample_type(samples: List[EvalSample]) -> str:
    """Detect if samples are single-turn or multi-turn."""
    if not samples:
        return "single_turn"
    # Check if any sample has multi-turn features
    for s in samples:
        if s.generated_response and s.metadata.get("is_multi_turn"):
            return "multi_turn"
    return "single_turn"