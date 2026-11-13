from typing import Dict, Any, List, Optional
from datetime import datetime


def format_ragas_scores(ragas_scores: Dict[str, float]) -> Dict[str, str]:
    """Format Ragas scores as human-readable strings."""
    formatted = {}
    for metric, score in ragas_scores.items():
        formatted[metric] = f"{score:.4f}"
    return formatted


def format_deterministic_scores(deterministic_scores: Dict[str, float]) -> Dict[str, str]:
    """Format deterministic scores as human-readable strings."""
    formatted = {}
    for metric, score in deterministic_scores.items():
        formatted[metric] = f"{score:.4f}"
    return formatted


def experiment_to_json(
    experiment_result: "ExperimentResult",
) -> Dict[str, Any]:
    """Convert an experiment result to a JSON-serializable dict."""
    return {
        "experiment_id": experiment_result.experiment_id,
        "timestamp": experiment_result.timestamp,
        "dataset_id": experiment_result.dataset_id,
        "retrieval_strategy": experiment_result.retrieval_strategy.value
        if hasattr(experiment_result.retrieval_strategy, 'value')
        else experiment_result.retrieval_strategy,
        "reranker_on": experiment_result.reranker_on,
        "generation_model": experiment_result.generation_model,
        "judge_provider": experiment_result.judge_provider,
        "judge_model": experiment_result.judge_model,
        "ragas_version": experiment_result.ragas_version,
        "metric_names": experiment_result.metric_names,
        "sample_count": experiment_result.sample_count,
        "failed_count": experiment_result.failed_count,
        "skipped_count": experiment_result.skipped_count,
        "ragas_scores": format_ragas_scores(experiment_result.ragas_scores)
        if experiment_result.ragas_scores
        else {},
        "deterministic_scores": format_deterministic_scores(
            experiment_result.deterministic_scores
        )
        if experiment_result.deterministic_scores
        else {},
        "per_sample_results": experiment_result.per_sample_results,
        "summary_md": experiment_result.summary_md,
    }


def ragas_dataset_to_json(
    dataset: "EvaluationDataset",
) -> Dict[str, Any]:
    """Convert a Ragas EvaluationDataset to JSON."""
    import json

    samples = dataset.to_list()
    return {
        "dataset_name": getattr(dataset, 'name', 'unknown'),
        "sample_count": len(samples),
        "samples": [sample_to_json(s) for s in samples],
    }


def sample_to_json(sample: "SingleTurnSample") -> Dict[str, Any]:
    """Convert a Ragas SingleTurnSample to JSON."""
    return {
        "user_input": sample.user_input,
        "retrieved_contexts": sample.retrieved_contexts or [],
        "reference_contexts": sample.reference_contexts or [],
        "response": sample.response,
        "reference": sample.reference,
        "metadata": sample.metadata or {},
    }


def scores_summary(
    ragas_scores: Dict[str, float],
    deterministic_scores: Dict[str, float],
) -> Dict[str, Any]:
    """Create a summary of all evaluation scores."""
    return {
        "ragas": format_ragas_scores(ragas_scores),
        "deterministic": format_deterministic_scores(deterministic_scores),
        "overall_max": max(
            list(ragas_scores.values()) + list(deterministic_scores.values())
        )
        if ragas_scores or deterministic_scores
        else 0.0,
        "overall_min": min(
            list(ragas_scores.values()) + list(deterministic_scores.values())
        )
        if ragas_scores or deterministic_scores
        else 0.0,
    }


def experiment_summary_md(
    result: Dict[str, Any],
) -> str:
    """Generate markdown summary from experiment data."""
    lines = [
        f"# Experiment: {result.get('experiment_id', 'unknown')}",
        "",
        f"**Timestamp**: {result.get('timestamp', 'unknown')}",
        "",
        f"**Retrieval Strategy**: {result.get('retrieval_strategy', 'unknown')}",
        f"**Reranker**: {'On' if result.get('reranker_on', False) else 'Off'}",
    ]
    if result.get('generation_model'):
        lines.append(f"**Generation Model**: {result['generation_model']}")
    if result.get('judge_provider'):
        lines.append(f"**Judge**: {result['judge_provider']} / {result['judge_model']}")
    lines.extend([
        "",
        f"**Ragas Version**: {result.get('ragas_version', 'unknown')}",
        f"**Metric Names**: {result.get('metric_names', [])}",
        "",
        f"**Sample Count**: {result.get('sample_count', 0)}",
        f"**Successful**: {result.get('sample_count', 0) - result.get('failed_count', 0) - result.get('skipped_count', 0)}",
        f"**Failed**: {result.get('failed_count', 0)}",
        f"**Skipped**: {result.get('skipped_count', 0)}",
        "",
        "## Ragas Metrics",
    ])

    ragas = result.get('ragas_scores', {})
    for metric_name, score in ragas.items():
        lines.append(f"- **{metric_name}**: {score:.4f}")

    lines.extend([
        "",
        "## Deterministic Metrics",
    ])

    deterministic = result.get('deterministic_scores', {})
    for metric_name, score in deterministic.items():
        lines.append(f"- **{metric_name}**: {score:.4f}")

    lines.extend([
        "",
        "---",
    ])

    return "\n".join(lines)


def per_sample_export(
    per_sample_data: List[Dict[str, Any]],
) -> str:
    """Export per-sample data as JSONL string."""
    import json
    lines = []
    for item in per_sample_data:
        lines.append(json.dumps(item, ensure_ascii=False))
    return "\n".join(lines)