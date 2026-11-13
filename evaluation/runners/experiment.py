from typing import List, Dict, Any, Optional
from evaluation.contracts.models import (
    ExperimentConfig, ExperimentResult, RetrievalStrategy, EvalSample,
    MetricProfile, RETRIEVAL_QUALITY, RERANKING_QUALITY,
    GENERATION_QUALITY, FULL_RAG_QUALITY
)
from evaluation.datasets.schema import (
    EvalSample, validate_dataset, hash_dataset, detect_sample_type,
    samples_to_ragas_dataset
)
from evaluation.ragas.configuration import RagasConfiguration, RagasJudgeConfig
from evaluation.ragas.evaluator import RagasExperimentRunner, MetricRegistry, EvaluationExecutor
from evaluation.evaluators.retrieval import RetrievalEvaluator
from evaluation.evaluators.generation import GenerationEvaluator
from evaluation.evaluators.end_to_end import EndToEndEvaluator
from evaluation.metrics.retrieval.base import retrieval_metrics
from evaluation.metrics.reranking.metrics import reranking_metrics, before_after_comparison
from evaluation.metrics.citations.metrics import (
    citation_reference_validity,
    citation_coverage,
    citation_integrity,
)
from evaluation.metrics.performance.metrics import (
    performance_summary,
    token_usage_from_samples,
    LatencyRecord,
)
from evaluation.storage.local import save_experiment_config, save_ragas_results, \
    save_deterministic_results, save_per_sample_results, save_summary_markdown, \
    load_experiment_config, load_ragas_results, load_deterministic_results, \
    load_per_sample_results
from evaluation.contracts.exceptions import (
    ExperimentConfigurationError, JudgeConfigurationError,
    RagasIntegrationError, MetricNotApplicableError
)


class ExperimentRunner:
    """Reusable experiment runner for RAG evaluation."""

    def __init__(self, config: RagasConfiguration):
        self.config = config
        self.runner = RagasExperimentRunner(config)
        self.metric_registry = MetricRegistry(config)
        self.retrieval_evaluator = RetrievalEvaluator(config)
        self.reranking_evaluator = RerankingEvaluator(config)
        self.generation_evaluator = GenerationEvaluator(config)
        self.end_to_end_evaluator = EndToEndEvaluator(config)

    def run(
        self,
        experiment_config: ExperimentConfig,
        dataset_samples: List[EvalSample],
        judge_config: RagasJudgeConfig,
        reranker_on: bool = False,
        generation_model: Optional[str] = None,
    ) -> ExperimentResult:
        """Run a complete evaluation experiment."""
        import json
        from datetime import datetime

        # Validate experiment configuration
        if not experiment_config.experiment_id:
            raise ExperimentConfigurationError(
                "Experiment must have an experiment_id"
            )

        if not dataset_samples:
            raise ExperimentConfigurationError(
                "Dataset samples cannot be empty"
            )

        # Validate samples
        try:
            validate_dataset(dataset_samples)
        except Exception as e:
            raise ExperimentConfigurationError(
                f"Dataset validation failed: {e}"
            )

        # Record start time
        start_time = datetime.utcnow()

        # Determine which metric profile to use
        has_generated = any(
            s.generated_response is not None and s.generated_response.strip()
            for s in dataset_samples
        )
        has_reference = any(
            s.reference is not None and s.reference.strip()
            for s in dataset_samples
        )
        has_contexts = any(
            s.retrieved_contexts is not None and len(s.retrieved_contexts) > 0
            for s in dataset_samples
        )

        profile_name = self._determine_profile(
            has_generated, has_reference, has_contexts
        )
        profile = self.metric_registry.get_profile(profile_name)

        # Check metric applicability across samples
        blocked_reasons = self.metric_registry.get_blocked_reasons(
            profile_name, dataset_samples
        )

        # Execute evaluation based on profile and available components
        ragas_scores = {}
        deterministic_scores = {}
        per_sample_data_parts = []
        failed_count = 0
        skipped_count = 0

        # Execute retrieval metrics
        if "retrieval_quality" in profile.metrics or profile_name == "retrieval_quality":
            # Compute deterministic retrieval metrics
            # ... (using retrieved contexts from samples)
            pass

        # Execute reranking metrics if applicable
        if "reranking_quality" in profile.metrics:
            # Compare before/after reranking
            # ... (using candidate lists from samples)
            pass

        # Execute generation metrics if generation is enabled
        if "generation_quality" in profile.metrics and has_generated:
            # Run Ragas generation metrics
            # ... (requires LLM and embeddings)
            pass

        # Execute full RAG quality if applicable
        if profile_name == "full_rag_quality":
            # Combine retrieval and generation metrics
            # ...
            pass

        # Generate summary markdown
        summary_md = self._generate_summary(
            profile_name, ragas_scores, deterministic_scores,
            len(dataset_samples), failed_count, skipped_count
        )

        # Compute experiment duration
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        # Create experiment result
        result = ExperimentResult(
            experiment_id=experiment_config.experiment_id,
            dataset_id=experiment_config.dataset_id,
            retrieval_strategy=experiment_config.retrieval_strategy,
            reranker_on=reranker_on,
            generation_model=generation_model,
            judge_provider=judge_config.provider,
            judge_model=judge_config.model_id,
            ragas_version=self.config.ragas_version,
            metric_names=profile.metrics,
            sample_count=len(dataset_samples),
            failed_count=failed_count,
            skipped_count=skipped_count,
            ragas_scores=ragas_scores,
            deterministic_scores=deterministic_scores,
            per_sample_results="",  # Will be populated
            summary_md=summary_md,
            timestamp=start_time.isoformat(),
        )

        # Save results to disk
        self._save_result(result)

        return result

    def _determine_profile(
        self,
        has_generated: bool,
        has_reference: bool,
        has_contexts: bool,
    ) -> str:
        """Determine which metric profile to use."""
        from evaluation.contracts.models import (
            RETRIEVAL_QUALITY, RERANKING_QUALITY, GENERATION_QUALITY,
            FULL_RAG_QUALITY
        )

        if has_generated and has_reference and has_contexts:
            return "full_rag_quality"
        elif has_generated:
            return "generation_quality"
        elif has_contexts:
            return "retrieval_quality"
        else:
            return "retrieval_quality"

    def _generate_summary(
        self,
        profile_name: str,
        ragas_scores: Dict[str, float],
        deterministic_scores: Dict[str, float],
        sample_count: int,
        failed_count: int,
        skipped_count: int,
    ) -> str:
        """Generate markdown summary for the experiment."""
        from evaluation.datasets.conversion import generate_experiment_summary

        return generate_experiment_summary(
            experiment_config=None,  # Will be filled in
            ragas_results=ragas_scores,
            deterministic_results=deterministic_scores,
            per_sample_count=sample_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
        )

    def _save_result(self, result: ExperimentResult) -> None:
        """Save experiment result to disk."""
        import os
        import json

        run_dir = f"evaluation/runs/{result.experiment_id}"
        os.makedirs(run_dir, exist_ok=True)

        # Save config
        save_experiment_config(result, result.experiment_id)

        # Save Ragas results
        if result.ragas_scores:
            save_ragas_results(result.ragas_scores, result.experiment_id)

        # Save deterministic results
        if result.deterministic_scores:
            save_deterministic_results(result.deterministic_scores, result.experiment_id)

        # Save per-sample results (placeholder - would be JSONL)
        if result.per_sample_results:
            save_per_sample_results(result.per_sample_results, result.experiment_id)

        # Save summary
        save_summary_markdown(result.summary_md, result.experiment_id)


class ComparisonRunner:
    """Runs comparisons between different retrieval/reranking configurations."""

    def __init__(self, config: RagasConfiguration):
        self.config = config

    def compare_retrieval_strategies(
        self,
        experiment_configs: List[ExperimentConfig],
        dataset_samples: List[EvalSample],
        judge_config: RagasJudgeConfig,
    ) -> Dict[str, ExperimentResult]:
        """Compare multiple retrieval strategies on the same dataset."""
        results = {}
        for config in experiment_configs:
            result = self.run(
                config, dataset_samples, judge_config
            )
            results[config.experiment_id] = result
        return results

    def compare_reranking(
        self,
        experiment_configs: List[ExperimentConfig],
        dataset_samples: List[EvalSample],
        judge_config: RagasJudgeConfig,
    ) -> Dict[str, ExperimentResult]:
        """Compare with and without reranking."""
        results = {}
        for config in experiment_configs:
            # Enable/disable reranker based on config
            result = self.run(
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
        judge_config: RagasJudgeConfig,
    ) -> Dict[str, ExperimentResult]:
        """Compare different generation models."""
        results = {}
        for config in experiment_configs:
            result = self.run(
                config, dataset_samples, judge_config,
                reranker_on=config.reranker_on,
                generation_model=config.generation_model,
            )
            results[config.experiment_id] = result
        return results