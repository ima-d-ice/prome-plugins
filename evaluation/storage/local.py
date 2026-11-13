from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
from evaluation.datasets.conversion import (
    sample_json,
    default_experiment_dir,
    save_experiment_config,
    save_ragas_results,
    save_deterministic_results,
    save_per_sample_results,
    save_summary_markdown,
    load_experiment_config,
    load_ragas_results,
    load_deterministic_results,
    load_per_sample_results,
    generate_experiment_summary,
)
from evaluation.contracts.models import (
    ExperimentConfig, ExperimentResult, EvalSample, RetrievalStrategy,
)
from evaluation.reports.json_report import experiment_to_json, ragas_dataset_to_json
from evaluation.reports.csv_report import experiment_to_csv
from evaluation.reports.markdown_report import (
    experiment_result_markdown,
    comparison_markdown,
    report_header,
)


class LocalStorage:
    """Local file-based storage for experiment results."""

    def __init__(self, base_dir: str = "evaluation/runs"):
        self.base_dir = base_dir

    def experiment_dir(self, experiment_id: str) -> str:
        """Get the directory for an experiment."""
        return os.path.join(self.base_dir, experiment_id)

    def save_experiment(
        self,
        experiment_result: "ExperimentResult",
        experiment_config: "ExperimentConfig",
    ) -> str:
        """Save a complete experiment to disk."""
        run_dir = self.experiment_dir(experiment_result.experiment_id)
        os.makedirs(run_dir, exist_ok=True)

        # Save config
        config_path = save_experiment_config(experiment_config, experiment_result.experiment_id)

        # Save Ragas results
        if experiment_result.ragas_scores:
            save_ragas_results(experiment_result.ragas_scores, experiment_result.experiment_id)

        # Save deterministic results
        if experiment_result.deterministic_scores:
            save_deterministic_results(experiment_result.deterministic_scores, experiment_result.experiment_id)

        # Save per-sample results
        if experiment_result.per_sample_results:
            save_per_sample_results(experiment_result.per_sample_results, experiment_result.experiment_id)

        # Save summary
        save_summary_markdown(experiment_result.summary_md, experiment_result.experiment_id)

        return run_dir

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load a saved experiment."""
        config = load_experiment_config(experiment_id)
        ragas_results = load_ragas_results(experiment_id)
        deterministic_results = load_deterministic_results(experiment_id)
        per_sample = load_per_sample_results(experiment_id)

        return {
            "config": config,
            "ragas_results": ragas_results,
            "deterministic_results": deterministic_results,
            "per_sample": per_sample,
        }

    def experiment_exists(self, experiment_id: str) -> bool:
        """Check if an experiment exists."""
        config_path = os.path.join(self.base_dir, experiment_id, "config.json")
        return os.path.exists(config_path)

    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        if not os.path.exists(self.base_dir):
            return []
        return [
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its results."""
        run_dir = self.experiment_dir(experiment_id)
        if os.path.exists(run_dir):
            import shutil
            shutil.rmtree(run_dir)
            return True
        return False

    def export_json(self, experiment_id: str) -> str:
        """Export experiment as JSON."""
        experiment = self.load_experiment(experiment_id)
        return json.dumps(experiment, indent=2, default=str)

    def export_csv(self, experiment_id: str) -> str:
        """Export experiment as CSV."""
        # Build CSV from experiment data
        experiment = self.load_experiment(experiment_id)
        return experiment_to_csv(experiment["config"]) if experiment["config"] else ""

    def export_markdown(self, experiment_id: str) -> str:
        """Export experiment as Markdown."""
        experiment = self.load_experiment(experiment_id)
        result_dict = {
            "experiment_id": experiment_id,
            "timestamp": experiment["config"].timestamp if experiment["config"] else "",
            "dataset_id": experiment["config"].dataset_id if experiment["config"] else "",
            "retrieval_strategy": experiment["config"].retrieval_strategy.value
            if experiment["config"] else "",
            "reranker_on": experiment["config"].reranker_on if experiment["config"] else False,
            "generation_model": experiment["config"].generation_model if experiment["config"] else None,
            "judge_provider": experiment["config"].judge_provider if experiment["config"] else None,
            "judge_model": experiment["config"].judge_model if experiment["config"] else None,
            "ragas_version": experiment["config"].ragas_version if experiment["config"] else "0.4.3",
            "metric_names": experiment["config"].metric_names if experiment["config"] else [],
            "sample_count": experiment["config"].sample_count if experiment["config"] else 0,
            "failed_count": experiment["config"].failed_count if experiment["config"] else 0,
            "skipped_count": experiment["config"].skipped_count if experiment["config"] else 0,
            "ragas_scores": experiment["ragas_results"],
            "deterministic_scores": experiment["deterministic_results"],
        }
        return experiment_result_markdown(result_dict)

    def save_json_report(
        self,
        experiment_id: str,
        filepath: Optional[str] = None,
    ) -> str:
        """Save JSON report to file."""
        if filepath is None:
            filepath = f"evaluation/runs/{experiment_id}/ragas_results.json"

        experiment = self.load_experiment(experiment_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(experiment, f, indent=2, default=str)

        return filepath

    def save_markdown_report(
        self,
        experiment_id: str,
        filepath: Optional[str] = None,
    ) -> str:
        """Save Markdown report to file."""
        if filepath is None:
            filepath = f"evaluation/runs/{experiment_id}/summary.md"

        experiment = self.load_experiment(experiment_id)
        result_dict = {
            "experiment_id": experiment_id,
            "timestamp": experiment["config"].timestamp if experiment["config"] else "",
            "dataset_id": experiment["config"].dataset_id if experiment["config"] else "",
            "retrieval_strategy": experiment["config"].retrieval_strategy.value
            if experiment["config"] else "",
            "reranker_on": experiment["config"].reranker_on if experiment["config"] else False,
            "generation_model": experiment["config"].generation_model if experiment["config"] else None,
            "judge_provider": experiment["config"].judge_provider if experiment["config"] else None,
            "judge_model": experiment["config"].judge_model if experiment["config"] else None,
            "ragas_version": experiment["config"].ragas_version if experiment["config"] else "0.4.3",
            "metric_names": experiment["config"].metric_names if experiment["config"] else [],
            "sample_count": experiment["config"].sample_count if experiment["config"] else 0,
            "failed_count": experiment["config"].failed_count if experiment["config"] else 0,
            "skipped_count": experiment["config"].skipped_count if experiment["config"] else 0,
            "ragas_scores": experiment["ragas_results"],
            "deterministic_scores": experiment["deterministic_results"],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(experiment_result_markdown(result_dict))

        return filepath