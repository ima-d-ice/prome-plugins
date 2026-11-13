from typing import List, Dict, Any, Optional
from evaluation.contracts.models import (
    ExperimentConfig, ExperimentResult, RetrievalStrategy, EvalSample
)
from evaluation.runners.experiment import ExperimentRunner


class ComparisonRunner:
    """Runs comparisons between different evaluation configurations."""

    def __init__(self, config):
        self.config = config

    def compare_retrieval_strategies(
        self,
        experiment_configs: List[ExperimentConfig],
        dataset_samples: List[EvalSample],
        judge_config,
    ) -> Dict[str, ExperimentResult]:
        """Compare multiple retrieval strategies on the same dataset."""
        runner = ExperimentRunner(self.config)
        results = {}
        for config in experiment_configs:
            result = runner.run(
                config, dataset_samples, judge_config
            )
            results[config.experiment_id] = result
        return results

    def compare_reranking(
        self,
        experiment_configs: List[ExperimentConfig],
        dataset_samples: List[EvalSample],
        judge_config,
    ) -> Dict[str, ExperimentResult]:
        """Compare with and without reranking."""
        runner = ExperimentRunner(self.config)
        results = {}
        for config in experiment_configs:
            result = runner.run(
                config, dataset_samples, judge_config,
                reranker_on=config.reranker_on,
                generation_model=config.generation_model,
            )
            results[config.experiment_id] = result
        return results

    def compare_generation_models(
        self,
        experiment_configs: List[ExperimentConfig],
        dataset_samples: List[EvalSample],
        judge_config,
    ) -> Dict[str, ExperimentResult]:
        """Compare different generation models."""
        runner = ExperimentRunner(self.config)
        results = {}
        for config in experiment_configs:
            result = runner.run(
                config, dataset_samples, judge_config,
                reranker_on=config.reranker_on,
                generation_model=config.generation_model,
            )
            results[config.experiment_id] = result
        return results