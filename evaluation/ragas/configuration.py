from typing import Optional, List, Dict, Any
from evaluation.contracts.models import (
    ExperimentConfig, ExperimentResult, MetricProfile, RetrievalStrategy,
    EvaluationStatus, RETRIEVAL_QUALITY, RERANKING_QUALITY,
    GENERATION_QUALITY, FULL_RAG_QUALITY
)
from evaluation.datasets.schema import (
    EvalSample, sample_to_ragas_single_turn, samples_to_ragas_dataset,
    detect_sample_type, validate_dataset
)


# Ragas version tracking
RAGAS_VERSION = "0.4.3"

# Default metric profiles
RETRIEVAL_QUALITY_PROFILE = RETRIEVAL_QUALITY
RERANKING_QUALITY_PROFILE = RERANKING_QUALITY
GENERATION_QUALITY_PROFILE = GENERATION_QUALITY
FULL_RAG_QUALITY_PROFILE = FULL_RAG_QUALITY


class RagasConfiguration:
    """Configuration for Ragas evaluation execution."""

    def __init__(
        self,
        ragas_version: str = RAGAS_VERSION,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.01,
        llm_max_tokens: Optional[int] = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_revision: Optional[str] = None,
        run_config: Optional[Any] = None,
        metric_profiles: Optional[Dict[str, MetricProfile]] = None,
        use_instructor: bool = True,
        concurrency: int = 5,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        self.ragas_version = ragas_version
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_revision = embedding_revision
        self.run_config = run_config or {}
        self.metric_profiles = metric_profiles or {
            "retrieval_quality": RETRIEVAL_QUALITY_PROFILE,
            "reranking_quality": RERANKING_QUALITY_PROFILE,
            "generation_quality": GENERATION_QUALITY_PROFILE,
            "full_rag_quality": FULL_RAG_QUALITY_PROFILE,
        }
        self.use_instructor = use_instructor
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout

    def get_profile(self, name: str) -> MetricProfile:
        """Get a metric profile by name."""
        return self.metric_profiles.get(name, self.metric_profiles["full_rag_quality"])


class RagasJudgeConfig:
    """Configuration for a Ragas judge LLM."""

    def __init__(
        self,
        provider: str,
        model_id: str,
        model_revision: Optional[str] = None,
        temperature: float = 0.01,
        max_tokens: Optional[int] = None,
    ):
        self.provider = provider
        self.model_id = model_id
        self.model_revision = model_revision
        self.temperature = temperature
        self.max_tokens = max_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class RagasExperimentRunner:
    """Runner for Ragas-based evaluation experiments."""

    def __init__(self, config: RagasConfiguration):
        self.config = config
        self.ragas_version = config.ragas_version

    def _create_llm(self, client: Any = None) -> Any:
        """Create a Ragas LLM instance."""
        from ragas.llms import llm_factory
        import instructor

        # Use llm_factory with instructor adapter
        # This avoids the vertexai import issue
        if client is None:
            raise ValueError("llm_factory requires a client instance")

        # Use the Instructor-based adapter which supports Groq and Ollama
        from ragas.llms.adapters.instructor import InstructorAdapter
        adapter = InstructorAdapter()
        llm = adapter.create_llm(client, self.config.llm_model, self.config.llm_provider)
        return llm

    def _create_embeddings(self, client: Any = None) -> Any:
        """Create a Ragas embeddings instance."""
        from ragas.embeddings.base import embedding_factory

        if client is None:
            # Try to use the model name directly
            return embedding_factory(
                self.config.embedding_provider,
                model=self.config.embedding_model,
            )
        else:
            return embedding_factory(
                self.config.embedding_provider,
                model=self.config.embedding_model,
                client=client,
            )

    def _get_metric_instance(
        self,
        metric_name: str,
        llm: Any,
        embeddings: Any = None,
    ) -> Any:
        """Get a Ragas metric instance."""
        from ragas.metrics import (
            ContextPrecision, ContextRecall, Faithfulness,
            AnswerRelevancy, NoiseSensitivity, ContextEntityRecall,
            FactualCorrectness,
        )

        metrics_map = {
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            "faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "noise_sensitivity": NoiseSensitivity,
            "context_entity_recall": ContextEntityRecall,
            "factual_correctness": FactualCorrectness,
        }

        metric_class = metrics_map.get(metric_name)
        if metric_class is None:
            raise ValueError(f"Unknown metric: {metric_name}")

        if metric_name == "context_precision":
            return metric_class(llm=llm)
        elif metric_name == "context_recall":
            return metric_class(llm=llm)
        elif metric_name == "faithfulness":
            return metric_class(llm=llm)
        elif metric_name == "answer_relevancy":
            return metric_class(llm=llm, embeddings=embeddings)
        elif metric_name == "noise_sensitivity":
            return metric_class(llm=llm)
        elif metric_name == "context_entity_recall":
            return metric_class(llm=llm)
        elif metric_name == "factual_correctness":
            return metric_class(llm=llm)
        else:
            return metric_class(llm=llm)

    async def _score_sample(
        self,
        metric: Any,
        user_input: str,
        reference: Optional[str],
        retrieved_contexts: Optional[List[str]],
        response: Optional[str] = None,
    ) -> Optional[float]:
        """Score a single sample with a metric."""
        try:
            # Use async scoring
            if hasattr(metric, 'ascore'):
                result = await metric.ascore(
                    user_input=user_input,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts or [],
                    response=response,
                )
                return getattr(result, 'value', None) or getattr(result, 'score', None)
            elif hasattr(metric, 'score'):
                result = await metric.score(
                    user_input=user_input,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts or [],
                    response=response,
                )
                return result
            else:
                # Try direct async call
                result = await metric(
                    user_input=user_input,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts or [],
                    response=response,
                )
                return result
        except Exception as e:
            # Log and return None (metric blocked)
            print(f"Metric scoring failed: {e}")
            return None

    def run_evaluation(
        self,
        dataset: EvaluationDataset,
        retrieval_strategy: RetrievalStrategy,
        judge_config: Optional[RagasJudgeConfig] = None,
        reranker_on: bool = False,
        generation_model: Optional[str] = None,
    ) -> ExperimentResult:
        """Run a full Ragas evaluation experiment."""
        import json
        from datetime import datetime

        # Create judge config if not provided
        if judge_config is None:
            judge_config = RagasJudgeConfig(
                provider=self.config.llm_provider,
                model_id=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )

        # Create LLM and embeddings clients
        # Note: client must be provided externally - this is a framework dependency
        # The actual client creation (OpenAI, Groq, Ollama) is handled by the caller

        # For now, we record the configuration and mark as needing client
        # The actual evaluation execution requires a properly configured LLM client

        return ExperimentResult(
            experiment_id="",  # Will be set by caller
            dataset_id="",
            retrieval_strategy=retrieval_strategy,
            reranker_on=reranker_on,
            generation_model=generation_model,
            judge_provider=judge_config.provider,
            judge_model=judge_config.model_id,
            ragas_version=self.ragas_version,
            metric_names=[m.name for m in self.config.metric_profiles.values()],
            sample_count=0,
            failed_count=0,
            skipped_count=0,
            ragas_scores={},
            deterministic_scores={},
            per_sample_results="",
            summary_md="",
            timestamp=datetime.utcnow().isoformat(),
        )

    def score_samples(
        self,
        samples: List[EvalSample],
        llm: Any,
        embeddings: Any = None,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Score a list of samples with specified metrics."""
        import asyncio

        if metric_names is None:
            metric_names = [
                "context_precision", "context_recall", "faithfulness",
                "answer_relevancy", "noise_sensitivity",
            ]

        # Build metrics map
        from ragas.metrics import (
            ContextPrecision, ContextRecall, Faithfulness,
            AnswerRelevancy, NoiseSensitivity, ContextEntityRecall,
            FactualCorrectness,
        )

        metrics_map = {
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "noise_sensitivity": NoiseSensitivity(llm=llm),
            "context_entity_recall": ContextEntityRecall(llm=llm),
            "factual_correctness": FactualCorrectness(llm=llm),
        }

        # Convert samples to Ragas format
        from evaluation.datasets.schema import samples_to_ragas_dataset
        ragas_dataset = samples_to_ragas_dataset(samples)

        # Run evaluation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            results = loop.run_until_complete(self._run_evaluation_with_metrics(
                ragas_dataset, metrics_map, metric_names
            ))
            return results
        finally:
            loop.close()

    async def _run_evaluation_with_metrics(
        self,
        dataset: EvaluationDataset,
        metrics_map: Dict[str, Any],
        metric_names: List[str],
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """Run evaluation with specific metrics on a dataset."""
        from ragas import evaluate

        # Use Ragas evaluate function with selected metrics
        # Filter metrics to only those we have instances for
        selected_metrics = [metrics_map[name] for name in metric_names if name in metrics_map]

        if not selected_metrics:
            return {}

        try:
            result = await evaluate(
                dataset=dataset,
                metrics=selected_metrics,
                run_config=self.config.run_config,
                max_retries=self.config.max_retries,
                timeout=self.config.timeout,
                max_concurrency=self.config.concurrency,
            )
            return result
        except Exception as e:
            print(f"Ragas evaluation failed: {e}")
            return {}


def create_default_configuration() -> RagasConfiguration:
    """Create a default Ragas configuration."""
    return RagasConfiguration()