from typing import Dict, Any, List, Optional
from datetime import datetime


def format_score(value: float, metric_name: str) -> str:
    """Format a single metric score for display."""
    return f"{value:.4f} ({metric_name})"


def ragas_results_markdown(
    ragas_scores: Dict[str, float],
    title: str = "Ragas Metrics Results",
) -> str:
    """Generate markdown table for Ragas scores."""
    lines = [
        f"## {title}",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]

    for metric_name, score in ragas_scores.items():
        lines.append(f"| {metric_name} | {format_score(score, metric_name)} |")

    lines.append("")
    return "\n".join(lines)


def deterministic_results_markdown(
    deterministic_scores: Dict[str, float],
    title: str = "Deterministic Metrics Results",
) -> str:
    """Generate markdown table for deterministic scores."""
    lines = [
        f"## {title}",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]

    for metric_name, score in deterministic_scores.items():
        lines.append(f"| {metric_name} | {format_score(score, metric_name)} |")

    lines.append("")
    return "\n".join(lines)


def experiment_result_markdown(
    result: Dict[str, Any],
    title: str = "Experiment Result",
) -> str:
    """Generate full markdown report for an experiment result."""
    lines = [
        f"# {title}",
        "",
        f"**Timestamp**: {result.get('timestamp', 'unknown')}",
        "",
        f"- **Experiment ID**: {result.get('experiment_id', 'unknown')}",
        f"- **Dataset ID**: {result.get('dataset_id', 'unknown')}",
        f"- **Retrieval Strategy**: {result.get('retrieval_strategy', 'unknown')}",
        f"- **Reranker**: {'On' if result.get('reranker_on', False) else 'Off'}",
    ]
    if result.get('generation_model'):
        lines.append(f"- **Generation Model**: {result['generation_model']}")
    if result.get('judge_provider'):
        lines.append(f"- **Judge Provider**: {result['judge_provider']}")
        lines.append(f"- **Judge Model**: {result['judge_model']}")
    lines.extend([
        "",
        f"- **Ragas Version**: {result.get('ragas_version', 'unknown')}",
        f"- **Metric Names**: {', '.join(result.get('metric_names', []))}",
        "",
        f"- **Sample Count**: {result.get('sample_count', 0)}",
        f"- **Successful**: {result.get('sample_count', 0) - result.get('failed_count', 0) - result.get('skipped_count', 0)}",
        f"- **Failed**: {result.get('failed_count', 0)}",
        f"- **Skipped**: {result.get('skipped_count', 0)}",
        "",
        "## Ragas Metrics",
    ])

    ragas = result.get('ragas_scores', {})
    if ragas:
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for metric_name, score in ragas.items():
            lines.append(f"| {metric_name} | {format_score(score, metric_name)} |")
        lines.append("")
    else:
        lines.append("*No Ragas metrics were evaluated.*")
        lines.append("")

    deterministic = result.get('deterministic_scores', {})
    if deterministic:
        lines.append("## Deterministic Metrics")
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for metric_name, score in deterministic.items():
            lines.append(f"| {metric_name} | {format_score(score, metric_name)} |")
        lines.append("")
    else:
        lines.append("*No deterministic metrics were evaluated.*")
        lines.append("")

    lines.extend([
        "---",
        "",
    ])

    return "\n".join(lines)


def comparison_markdown(
    results: Dict[str, Dict[str, Any]],
    title: str = "Experiment Comparisons",
) -> str:
    """Generate markdown for comparison of multiple experiments."""
    lines = [
        f"# {title}",
        "",
    ]

    # Compare key metrics across experiments
    all_metrics = set()
    for result_name, result in results.items():
        ragas = result.get('ragas_scores', {})
        det = result.get('deterministic_scores', {})
        all_metrics.update(ragas.keys())
        all_metrics.update(det.keys())

    lines.append("| Experiment | " + " | ".join(sorted(all_metrics)) + " |")
    lines.append("|------------" * (len(all_metrics) + 1))

    for result_name, result in results.items():
        ragas = result.get('ragas_scores', {})
        det = result.get('deterministic_scores', {})
        row = [result_name]
        for metric in sorted(all_metrics):
            val = ragas.get(metric, det.get(metric, "N/A"))
            row.append(f"{val:.4f}" if isinstance(val, float) else val)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def report_header(
    experiment_id: str,
    dataset_id: str,
    retrieval_strategy: str,
    timestamp: str,
) -> str:
    """Generate a standard report header."""
    lines = [
        f"# Evaluation Report",
        "",
        f"**Experiment ID**: {experiment_id}",
        f"**Dataset**: {dataset_id}",
        f"**Retrieval Strategy**: {retrieval_strategy}",
        f"**Generated**: {timestamp}",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)