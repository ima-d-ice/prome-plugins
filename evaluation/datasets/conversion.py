import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from evaluation.datasets.schema import EvalSample, validate_dataset, hash_dataset
from evaluation.datasets.schema import detect_sample_type
from evaluation.contracts.models import (
    ExperimentConfig, ExperimentResult, RetrievalStrategy, MetricProfile,
)
from evaluation.contracts.exceptions import DatasetValidationError


def default_experiment_dir(experiment_id: str) -> str:
    """Get the default directory for an experiment."""
    return f"evaluation/runs/{experiment_id}"


def save_experiment_config(
    config: ExperimentConfig,
    experiment_id: str,
) -> str:
    """Save experiment configuration to disk."""
    run_dir = default_experiment_dir(experiment_id)
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    return config_path


def save_ragas_results(
    results: Dict[str, float],
    experiment_id: str,
) -> str:
    """Save Ragas metric results."""
    run_dir = default_experiment_dir(experiment_id)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "ragas_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results_path


def save_deterministic_results(
    results: Dict[str, float],
    experiment_id: str,
) -> str:
    """Save deterministic metric results."""
    run_dir = default_experiment_dir(experiment_id)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "deterministic_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results_path


def save_per_sample_results(
    per_sample_data: str,
    experiment_id: str,
) -> str:
    """Save per-sample results as JSONL."""
    run_dir = default_experiment_dir(experiment_id)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "per_sample.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(per_sample_data)
    return results_path


def save_summary_markdown(
    summary_md: str,
    experiment_id: str,
) -> str:
    """Save experiment summary as Markdown."""
    run_dir = default_experiment_dir(experiment_id)
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    return summary_path


def load_experiment_config(experiment_id: str) -> ExperimentConfig:
    """Load experiment configuration from disk."""
    config_path = os.path.join("evaluation/runs", experiment_id, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Remove underscore-prefixed keys
    clean_data = {k: v for k, v in data.items() if not k.startswith("_")}
    return ExperimentConfig(**clean_data)


def load_ragas_results(experiment_id: str) -> Dict[str, float]:
    """Load Ragas metric results."""
    results_path = os.path.join("evaluation/runs", experiment_id, "ragas_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Ragas results not found: {results_path}")
    import json
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_deterministic_results(experiment_id: str) -> Dict[str, float]:
    """Load deterministic metric results."""
    results_path = os.path.join("evaluation/runs", experiment_id, "deterministic_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Deterministic results not found: {results_path}")
    import json
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_per_sample_results(experiment_id: str) -> List[Dict[str, Any]]:
    """Load per-sample results."""
    path = os.path.join("evaluation/runs", experiment_id, "per_sample.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Per-sample results not found: {path}")
    import json
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def sample_id_from_text(text: str) -> str:
    """Generate a stable sample ID from text content."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_jsonl_dataset(filepath: str) -> List[dict]:
    """Load a JSONL dataset from file."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(data)
    return samples


def validate_sample(sample: dict) -> bool:
    """Validate a sample dict has required fields."""
    if "sample_id" not in sample:
        raise DatasetValidationError("Sample must have a sample_id")
    if "user_input" not in sample or not sample["user_input"].strip():
        raise DatasetValidationError("Sample must have a non-empty user_input")
    return True


def validate_dataset(samples: List[dict]) -> bool:
    """Validate all samples in a dataset."""
    for i, sample in enumerate(samples):
        try:
            validate_sample(sample)
        except DatasetValidationError as e:
            raise DatasetValidationError(f"Sample {i} failed validation: {e}")
    return True


def sample_to_ragas_single_turn(sample: EvalSample) -> dict:
    """Convert an EvalSample to a Ragas SingleTurnSample dict."""
    return {
        "user_input": sample.user_input,
        "retrieved_contexts": sample.retrieved_contexts or [],
        "reference_contexts": sample.reference_contexts or [],
        "response": sample.generated_response,
        "reference": sample.reference,
        "retrieved_context_ids": sample.retrieved_context_ids,
        "reference_context_ids": sample.reference_context_ids,
        "metadata": sample.metadata,
    }


def samples_to_ragas_dataset(samples: List[EvalSample]) -> "EvaluationDataset":
    """Convert a list of EvalSamples to a Ragas EvaluationDataset."""
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

    ragas_samples = [sample_to_ragas_single_turn(s) for s in samples]
    return EvaluationDataset.from_samples(ragas_samples)


def hash_dataset(samples: List[EvalSample]) -> str:
    """Compute a hash of the dataset for integrity checking."""
    import hashlib
    content = "".join(
        s.sample_id + s.user_input + (s.reference or "")
        for s in sorted(samples, key=lambda x: x.sample_id)
    )
    return hashlib.sha256(content.encode()).hexdigest()


def detect_sample_type(samples: List[EvalSample]) -> str:
    """Detect if samples are single-turn or multi-turn."""
    if not samples:
        return "single_turn"
    for s in samples:
        if s.metadata.get("is_multi_turn"):
            return "multi_turn"
    return "single_turn"


def sample_json(sample: EvalSample) -> str:
    """Serialize an EvalSample to JSON."""
    return json.dumps(
        {k: v for k, v in sample.__dict__.items() if not k.startswith("_")},
        ensure_ascii=False,
        indent=2,
    )


def generate_experiment_summary(
    config: ExperimentConfig,
    ragas_results: Dict[str, float],
    deterministic_results: Dict[str, float],
    per_sample_count: int,
    failed_count: int,
    skipped_count: int,
) -> str:
    """Generate a Markdown summary for an experiment."""
    lines = [
        f"# Experiment Summary: {config.experiment_id}",
        "",
        f"**Timestamp**: {config.timestamp or datetime.utcnow().isoformat()}",
        "",
        f"**Dataset**: {config.dataset_id}",
        f"**Retrieval Strategy**: {config.retrieval_strategy.value}",
        f"**Reranker**: {'On' if config.reranker_on else 'Off'}",
    ]
    if config.generation_model:
        lines.append(f"**Generation Model**: {config.generation_model}")
    if config.judge_provider:
        lines.append(f"**Judge Provider**: {config.judge_provider}")
    if config.judge_model:
        lines.append(f"**Judge Model**: {config.judge_model}")
    lines.extend([
        "",
        f"**Ragas Version**: {config.ragas_version or 'unknown'}",
        f"**Metric Names**: {', '.join(config.metric_names)}",
        "",
        f"**Sample Count**: {config.sample_count}",
        f"**Successful**: {config.sample_count - failed_count - skipped_count}",
        f"**Failed**: {failed_count}",
        f"**Skipped**: {skipped_count}",
        "",
        "## Ragas Metrics",
    ])
    for metric_name, score in ragas_results.items():
        lines.append(f"- **{metric_name}**: {score:.4f}")
    lines.extend([
        "",
        "## Deterministic Metrics",
    ])
    for metric_name, score in deterministic_results.items():
        lines.append(f"- **{metric_name}**: {score:.4f}")
    lines.extend([
        "",
        "## Per-Sample Results",
        f"- Processed: {per_sample_count}",
        f"- Failed: {failed_count}",
        f"- Skipped: {skipped_count}",
        "",
        "---",
    ])
    return "\n".join(lines)