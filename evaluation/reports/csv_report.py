from typing import Dict, List, Any, Optional
import csv
import io


def ragas_scores_to_csv(
    ragas_scores: Dict[str, float],
    sample_ids: Optional[List[str]] = None,
) -> str:
    """Convert Ragas scores to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    if sample_ids:
        header = ["sample_id"] + list(ragas_scores.keys())
    else:
        header = list(ragas_scores.keys())
    writer.writerow(header)

    # Write data rows
    if sample_ids:
        for i, sample_id in enumerate(sample_ids):
            row = [sample_id]
            # Placeholder scores - in real use, these would be the actual scores
            for _ in ragas_scores.keys():
                row.append("")
            writer.writerow(row)
    else:
        # Single row with average scores
        writer.writerow(list(ragas_scores.keys()))
        writer.writerow(list(ragas_scores.values()))

    return output.getvalue()


def deterministic_scores_to_csv(
    deterministic_scores: Dict[str, float],
) -> str:
    """Convert deterministic scores to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(list(deterministic_scores.keys()))
    writer.writerow(list(deterministic_scores.values()))

    return output.getvalue()


def experiment_to_csv(
    experiment_result: "ExperimentResult",
) -> str:
    """Convert experiment result to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["field", "value"])
    writer.writerow(["experiment_id", experiment_result.experiment_id])
    writer.writerow(["timestamp", experiment_result.timestamp])
    writer.writerow(["dataset_id", experiment_result.dataset_id])
    writer.writerow([
        "retrieval_strategy",
        experiment_result.retrieval_strategy.value
        if hasattr(experiment_result.retrieval_strategy, 'value')
        else experiment_result.retrieval_strategy,
    ])
    writer.writerow(["reranker_on", experiment_result.reranker_on])
    writer.writerow(["generation_model", experiment_result.generation_model or ""])
    writer.writerow(["judge_provider", experiment_result.judge_provider or ""])
    writer.writerow(["judge_model", experiment_result.judge_model or ""])
    writer.writerow(["ragas_version", experiment_result.ragas_version])
    writer.writerow(["metric_names", ",".join(experiment_result.metric_names)])
    writer.writerow(["sample_count", experiment_result.sample_count])
    writer.writerow(["failed_count", experiment_result.failed_count])
    writer.writerow(["skipped_count", experiment_result.skipped_count])

    # Ragas scores
    writer.writerow([])
    writer.writerow(["ragas_score_field", "ragas_score_value"])
    for metric, score in experiment_result.ragas_scores.items():
        writer.writerow([metric, f"{score:.4f}"])

    # Deterministic scores
    writer.writerow([])
    writer.writerow(["deterministic_score_field", "deterministic_score_value"])
    for metric, score in experiment_result.deterministic_scores.items():
        writer.writerow([metric, f"{score:.4f}"])

    return output.getvalue()