from typing import Optional, List, Dict, Any
from datetime import datetime
from evaluation.contracts.models import (
    ExperimentConfig, ExperimentResult, RetrievalStrategy, MetricProfile,
    EvalSample
)
from evaluation.ragas.configuration import RagasConfiguration, RagasJudgeConfig
from evaluation.ragas.evaluator import RagasExperimentRunner, MetricRegistry, EvaluationExecutor
from evaluation.datasets.schema import (
    EvalSample, sample_to_ragas_single_turn, samples_to_ragas_dataset,
    validate_dataset, detect_sample_type
)
from evaluation.contracts.exceptions import (
    ExperimentConfigurationError, JudgeConfigurationError,
    RagasIntegrationError, MetricNotApplicableError
)


class RagasExperimentRunner:
    """Runner for Ragas-based evaluation experiments."""

    def __init__(self, config: RagasConfiguration):
        self.config = config
        self.ragas_version = config.ragas_version

    def initialize_from_config(
        self,
        experiment_config: ExperimentConfig,
        dataset_samples: List[EvalSample],
    ) -> None:
        """Initialize runner from experiment config and dataset."""
        self.experiment_config = experiment_config
        self.dataset_samples = dataset_samples
        self.metric_registry = MetricRegistry(self.config)

    def run_retrieval_quality(
        self,
        llm: Any,
        embeddings: Any = None,
    ) -> Dict[str, float]:
        """Run retrieval quality evaluation."""
        from evaluation.datasets.schema import samples_to_ragas_dataset

        # Determine applicable metrics
        profile = self.metric_registry.get_profile("retrieval_quality")
        applicable_metrics = self.metric_registry.validate_profile_applicability(
            "retrieval_quality",
            self.dataset_samples[0] if self.dataset_samples else EvalSample(sample_id="", user_input="")
        )

        # Select metrics that are applicable
        metric_names = [m for m in profile.metrics if m in applicable_metrics]

        if not metric_names:
            return {}

        # Execute metrics
        executor = EvaluationExecutor(self, self.config)
        results = asyncio_loop_metrics(
            self,
            self.dataset_samples,
            metric_names,
            llm,
            embeddings
        )
        return results

    def run_reranking_quality(
        self,
        llm: Any,
        embeddings: Any = None,
        has_before: bool = True,
        has_after: bool = True,
    ) -> Dict[str, float]:
        """Run reranking quality evaluation."""
        profile_name = "reranking_quality"
        profile = self.metric_registry.get_profile(profile_name)

        # Filter metrics based on availability
        metric_names = []
        if has_before:
            metric_names.extend([m for m in profile.metrics if "before" in m.lower()])
        if has_after:
            metric_names.extend([m for m in profile.metrics if "after" in m.lower()])

        if not metric_names:
            return {}

        # Execute metrics
        # ... (similar pattern)
        return {}

    def run_generation_quality(
        self,
        llm: Any,
        embeddings: Any = None,
    ) -> Dict[str, float]:
        """Run generation quality evaluation."""
        profile = self.metric_registry.get_profile("generation_quality")

        # Check if samples have generated responses and references
        has_response = any(s.generated_response is not None for s in self.dataset_samples)
        has_ref = any(sample_has_reference(s) for s in self.dataset_samples)

        # Select applicable metrics
        metric_names = []
        for m in profile.metrics:
            if m == "faithfulness" and has_response:
                metric_names.append(m)
            elif m == "answer_relevancy" and has_response:
                metric_names.append(m)
            elif m == "factual_correctness" and has_response and has_ref:
                metric_names.append(m)
            elif m == "noise_sensitivity" and has_response:
                metric_names.append(m)

        if not metric_names:
            return {}

        # Execute metrics
        executor = EvaluationExecutor(self, self.config)
        # ... run metrics
        return {}

    def run_full_rag_quality(
        self,
        llm: Any,
        embeddings: Any = None,
    ) -> Dict[str, float]:
        """Run full RAG quality evaluation."""
        profile = self.metric_registry.get_profile("full_rag_quality")

        # Check conditions
        has_response = any(s.generated_response is not None for s in self.dataset_samples)
        has_ref = any(sample_has_reference(s) for s in self.dataset_samples)
        has_contexts = any(
            s.retrieved_contexts is not None and len(s.retrieved_contexts) > 0
            for s in self.dataset_samples
        )

        # Select applicable metrics
        metric_names = []
        for m in profile.metrics:
            if m == "context_precision" and has_contexts:
                metric_names.append(m)
            elif m == "context_recall" and (has_ref or has_contexts):
                metric_names.append(m)
            elif m == "faithfulness" and has_response:
                metric_names.append(m)
            elif m == "answer_relevancy" and has_response:
                metric_names.append(m)

        if not metric_names:
            return {}

        # Execute metrics
        # ...
        return {}

    async def _score_with_metric(
        self,
        metric_name: str,
        metric_instance: Any,
        sample: EvalSample,
    ) -> Optional[float]:
        """Score a single sample with a specific metric."""
        try:
            # Convert sample to Ragas format
            from evaluation.datasets.schema import sample_to_ragas_single_turn
            ragas_sample = sample_to_ragas_single_turn(sample)

            # Build keyword args based on metric
            kwargs = {
                "user_input": sample.user_input,
                "retrieved_contexts": sample.retrieved_contexts or [],
            }

            if metric_name == "context_precision":
                kwargs["reference_contexts"] = sample.reference_contexts or []

            elif metric_name == "context_recall":
                if sample.reference:
                    kwargs["reference"] = sample.reference
                if sample.reference_contexts:
                    kwargs["reference_contexts"] = sample.reference_contexts

            elif metric_name == "faithfulness":
                kwargs["response"] = sample.generated_response or ""
                kwargs["retrieved_contexts"] = sample.retrieved_contexts or []

            elif metric_name == "answer_relevancy":
                kwargs["response"] = sample.generated_response or ""

            elif metric_name == "noise_sensitivity":
                kwargs["response"] = sample.generated_response or ""
                kwargs["retrieved_contexts"] = sample.retrieved_contexts or []

            elif metric_name == "context_entity_recall":
                if sample.reference:
                    kwargs["reference"] = sample.reference
                if sample.reference_contexts:
                    kwargs["reference_contexts"] = sample.reference_contexts

            elif metric_name == "factual_correctness":
                if sample.generated_response:
                    kwargs["response"] = sample.generated_response
                if sample.reference:
                    kwargs["reference"] = sample.reference

            # Execute metric
            metric = self.config.metric_profiles.get(metric_name)
            if metric is None:
                return None

            # Create metric instance
            from ragas.metrics import (
                ContextPrecision, ContextRecall, Faithfulness,
                AnswerRelevancy, NoiseSensitivity, ContextEntityRecall,
                FactualCorrectness,
            )

            metric_map = {
                "context_precision": ContextPrecision(llm=None),
                "context_recall": ContextRecall(llm=None),
                "faithfulness": Faithfulness(llm=None),
                "answer_relevancy": AnswerRelevancy(llm=None, embeddings=None),
                "noise_sensitivity": NoiseSensitivity(llm=None),
                "context_entity_recall": ContextEntityRecall(llm=None),
                "factual_correctness": FactualCorrectness(llm=None),
            }

            instance = metric_map.get(metric_name)
            if instance is None:
                return None

            # Try to score
            if hasattr(instance, 'ascore'):
                result = instance.ascore(**kwargs)
                if hasattr(result, 'value'):
                    return result.value
                elif hasattr(result, 'score'):
                    return result.score

            return None

        except Exception as e:
            print(f"Failed to score with {metric_name}: {e}")
            return None