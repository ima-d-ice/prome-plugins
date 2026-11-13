from typing import Optional, List, Dict, Any
from evaluation.contracts.models import (
    EvalSample, ExperimentConfig, ExperimentResult, RetrievalStrategy,
    MetricProfile, RETRIEVAL_QUALITY, RERANKING_QUALITY,
    GENERATION_QUALITY, FULL_RAG_QUALITY
)
from evaluation.ragas.configuration import RagasConfiguration, RagasJudgeConfig, RagasExperimentRunner
from evaluation.datasets.schema import (
    EvalSample, sample_to_ragas_single_turn, samples_to_ragas_dataset,
    validate_dataset, hash_dataset, detect_sample_type
)
from evaluation.contracts.exceptions import (
    ExperimentConfigurationError, JudgeConfigurationError,
    RagasIntegrationError, MetricNotApplicableError
)


class MetricRegistry:
    """Registry for evaluation metrics and profiles."""

    def __init__(self, config: RagasConfiguration):
        self.config = config
        self._profiles: Dict[str, MetricProfile] = {
            name: profile for name, profile in config.metric_profiles.items()
        }

    def get_profile(self, name: str) -> MetricProfile:
        """Get a metric profile by name."""
        if name in self._profiles:
            return self._profiles[name]
        # Try to get from config
        return self.config.get_profile(name)

    def list_profiles(self) -> List[str]:
        """List available metric profile names."""
        return list(self._profiles.keys())

    def get_metric_names_for_profile(self, profile_name: str) -> List[str]:
        """Get metric names for a specific profile."""
        profile = self.get_profile(profile_name)
        return profile.metrics

    def validate_profile_applicability(
        self,
        profile_name: str,
        sample: EvalSample,
    ) -> List[str]:
        """Validate which metrics in a profile are applicable for a sample."""
        applicable = []
        not_applicable = []

        profile = self.get_profile(profile_name)

        for metric_name in profile.metrics:
            try:
                # Check if the metric can be applied given the sample data
                if metric_name == "context_precision":
                    if sample.retrieved_contexts is not None:
                        applicable.append(metric_name)
                    else:
                        not_applicable.append(metric_name)

                elif metric_name == "context_recall":
                    if sample.reference_contexts is not None or sample.reference is not None:
                        applicable.append(metric_name)
                    else:
                        not_applicable.append(metric_name)

                elif metric_name == "faithfulness":
                    if sample.generated_response is not None and sample.retrieved_contexts is not None:
                        applicable.append(metric_name)
                    else:
                        not_applicable.append(metric_name)

                elif metric_name == "answer_relevancy":
                    if sample.generated_response is not None:
                        applicable.append(metric_name)
                    else:
                        not_applicable.append(metric_name)

                elif metric_name == "noise_sensitivity":
                    if sample.generated_response is not None and sample.retrieved_contexts is not None:
                        applicable.append(metric_name)
                    else:
                        not_applicable.append(metric_name)

                elif metric_name == "context_entity_recall":
                    if sample.reference is not None and sample.reference_contexts is not None:
                        applicable.append(metric_name)
                    else:
                        not_applicable.append(metric_name)

                elif metric_name == "factual_correctness":
                    if sample.reference is not None and sample.generated_response is not None:
                        applicable.append(metric_name)
                    else:
                        not_applicable.append(metric_name)

                else:
                    # Unknown metric - mark as not applicable
                    not_applicable.append(metric_name)

            except Exception:
                not_applicable.append(metric_name)

        return applicable

    def get_blocked_reasons(self, profile_name: str, samples: List[EvalSample]) -> Dict[str, List[str]]:
        """Get blocked metric reasons for each sample."""
        reasons: Dict[str, List[str]] = {}
        for sample in samples:
            applicable = self.validate_profile_applicability(profile_name, sample)
            blocked = [m for m in self.get_profile(profile_name).metrics if m not in applicable]
            if blocked:
                reasons[sample.sample_id] = blocked
        return reasons


class EvaluationExecutor:
    """Executes evaluation metrics against samples."""

    def __init__(self, runner: RagasExperimentRunner, config: RagasConfiguration):
        self.runner = runner
        self.config = config

    async def execute_metrics(
        self,
        samples: List[EvalSample],
        metric_names: List[str],
        llm: Any,
        embeddings: Any = None,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Execute metrics against a list of samples."""
        from evaluation.datasets.schema import samples_to_ragas_dataset

        # Validate samples
        validate_dataset(samples)

        # Convert to Ragas dataset
        ragas_dataset = samples_to_ragas_dataset(samples)

        # Get metric instances
        metrics_map = self._build_metrics_map(llm, embeddings)

        # Filter to requested metrics
        selected_metrics = [metrics_map[name] for name in metric_names if name in metrics_map]

        if not selected_metrics:
            return {}

        # Run evaluation
        return await self.runner._run_evaluation_with_metrics(
            ragas_dataset, metrics_map, selected_metrics
        )

    def _build_metrics_map(
        self,
        llm: Any,
        embeddings: Any = None,
    ) -> Dict[str, Any]:
        """Build a map of metric name -> metric instance."""
        from ragas.metrics import (
            ContextPrecision, ContextRecall, Faithfulness,
            AnswerRelevancy, NoiseSensitivity, ContextEntityRecall,
            FactualCorrectness,
        )

        return {
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "noise_sensitivity": NoiseSensitivity(llm=llm),
            "context_entity_recall": ContextEntityRecall(llm=llm),
            "factual_correctness": FactualCorrectness(llm=llm),
        }

    def determine_metric_profile(
        self,
        profile_name: str,
        retrieval_strategy: RetrievalStrategy,
        has_generated_response: bool,
        has_reference: bool,
        has_contexts: bool,
    ) -> MetricProfile:
        """Determine which profile to use based on conditions."""
        from evaluation.contracts.models import (
            RETRIEVAL_QUALITY, RERANKING_QUALITY, GENERATION_QUALITY,
            FULL_RAG_QUALITY
        )

        # Check conditions
        has_retrieval = True  # Always has some form of retrieval

        if profile_name == "retrieval_quality":
            return RETRIEVAL_QUALITY

        elif profile_name == "reranking_quality":
            return RERANKING_QUALITY

        elif profile_name == "generation_quality":
            if has_generated_response:
                return GENERATION_QUALITY
            else:
                # No generation, fall back to retrieval
                return RETRIEVAL_QUALITY

        elif profile_name == "full_rag_quality":
            if has_generated_response and has_reference and has_contexts:
                return FULL_RAG_QUALITY
            elif has_generated_response:
                return GENERATION_QUALITY
            else:
                return RETRIEVAL_QUALITY

        return RETRIEVAL_QUALITY