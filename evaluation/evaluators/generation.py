from typing import List, Dict, Any, Optional
from evaluation.contracts.models import (
    EvalSample, ExperimentConfig, ExperimentResult, RetrievalStrategy,
    RetrievalResult, RerankCandidate
)
from evaluation.ragas.configuration import RagasConfiguration, RagasJudgeConfig
from evaluation.ragas.evaluator import RagasExperimentRunner
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
from evaluation.contracts.exceptions import (
    ExperimentConfigurationError, JudgeConfigurationError,
    RagasIntegrationError
)


class RetrievalEvaluator:
    """Evaluates retrieval strategy performance with deterministic metrics."""

    def __init__(self, config: RagasConfiguration):
        self.config = config

    def evaluate_strategy(
        self,
        samples: List[EvalSample],
        relevant_doc_ids: Set[str],
    ) -> Dict[str, float]:
        """Evaluate a retrieval strategy using deterministic metrics."""
        results = {}

        for sample in samples:
            # Extract retrieved doc IDs from sample metadata or contexts
            retrieved_ids = self._extract_retrieved_ids(sample)

            # Compute metrics
            strategy_results = retrieval_metrics(
                [],  # Will be populated from actual retrieval
                relevant_doc_ids,
            )
            results[sample.sample_id] = strategy_results

        return results

    def _extract_retrieved_ids(self, sample: EvalSample) -> Set[str]:
        """Extract retrieved document IDs from a sample."""
        # Try various fields
        ids = set()

        # Check retrieved_context_ids
        if hasattr(sample, 'retrieved_context_ids') and sample.retrieved_context_ids:
            ids.update(sample.retrieved_context_ids)

        # Check metadata
        metadata = getattr(sample, 'metadata', {}) or {}
        if 'retrieved_doc_ids' in metadata:
            ids.update(metadata['retrieved_doc_ids'])

        return ids

    def compute_summary(
        self,
        all_results: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute summary statistics across all samples."""
        if not all_results:
            return {}

        summary = {}
        metric_names = set()

        # Collect all metric names
        for results in all_results.values():
            metric_names.update(results.keys())

        # Compute averages
        for metric in metric_names:
            values = [
                r.get(metric, 0.0)
                for r in all_results.values()
                if metric in r
            ]
            if values:
                summary[metric] = sum(values) / len(values)

        return summary


class RerankingEvaluator:
    """Evaluates reranker improvements."""

    def __init__(self, config: RagasConfiguration):
        self.config = config

    def evaluate_before_after(
        self,
        before_candidates: List[RerankCandidate],
        after_candidates: List[RerankCandidate],
        relevant_ids: Set[str],
    ) -> Dict[str, float]:
        """Compare retrieval before and after reranking."""
        return reranking_metrics(before_candidates, after_candidates, relevant_ids)

    def improvement_summary(
        self,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute improvement deltas between before and after metrics."""
        improvements = {}
        for key in after_metrics:
            if key in before_metrics:
                improvements[f"{key}_improvement"] = after_metrics[key] - before_metrics[key]
        return improvements


class GenerationEvaluator:
    """Evaluates generated answer quality using Ragas metrics."""

    def __init__(self, config: RagasConfiguration):
        self.config = config

    def evaluate_faithfulness(
        self,
        samples: List[EvalSample],
        llm: Any,
        embeddings: Any = None,
    ) -> Dict[str, float]:
        """Evaluate faithfulness of generated answers using Ragas."""
        from ragas.metrics import Faithfulness

        metric = Faithfulness(llm=llm)
        # ... score samples
        return {}

    def evaluate_relevancy(
        self,
        samples: List[EvalSample],
        llm: Any,
        embeddings: Any = None,
    ) -> Dict[str, float]:
        """Evaluate answer relevancy using Ragas."""
        from ragas.metrics import AnswerRelevancy

        metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
        # ... score samples
        return {}

    def evaluate_factual_correctness(
        self,
        samples: List[EvalSample],
        llm: Any,
    ) -> Dict[str, float]:
        """Evaluate factual correctness using Ragas."""
        from ragas.metrics import FactualCorrectness

        metric = FactualCorrectness(llm=llm)
        # ... score samples
        return {}


class EndToEndEvaluator:
    """Evaluates complete RAG pipeline end-to-end."""

    def __init__(self, config: RagasConfiguration):
        self.config = config

    def run_experiment(
        self,
        experiment_config: ExperimentConfig,
        dataset_samples: List[EvalSample],
        retrieval_results: dict,
        judge_config: RagasJudgeConfig,
        reranker_on: bool = False,
        generation_model: Optional[str] = None,
    ) -> ExperimentResult:
        """Run a complete evaluation experiment."""
        # Implementation in the runner
        from evaluation.runners.experiment import ExperimentRunner
        return ExperimentRunner.run(
            experiment_config, dataset_samples, retrieval_results,
            judge_config, reranker_on, generation_model
        )