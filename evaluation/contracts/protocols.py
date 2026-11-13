from typing import Protocol, TypeVar, Generic, Optional, List
from dataclasses import dataclass

# Type variables for generic typing
T = TypeVar("T")
M = TypeVar("M")


class EvaluationMetric(Protocol):
    """Protocol for Ragas-based evaluation metrics."""

    name: str

    async def acode(
        self,
        user_input: str,
        reference: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None,
        response: Optional[str] = None,
    ) -> float:
        """Evaluate a single sample asynchronously."""
        ...

    def code(
        self,
        user_input: str,
        reference: Optional[str] = None,
        retrieved_contexts: Optional[List[str]] = None,
        response: Optional[str] = None,
    ) -> float:
        """Evaluate a single sample synchronously."""
        ...


class EvaluationConfig(Protocol):
    """Protocol for evaluation configuration."""

    llm: Any
    embeddings: Any
    run_config: Any
    metric_names: List[str]


class RetrievalStrategyProtocol(Protocol):
    """Protocol for retrieval strategies."""

    name: str

    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve chunks for a query."""
        ...

    def retrieve_async(self, query: str) -> List[RetrievalResult]:
        """Async retrieval."""
        ...


class RerankerProtocol(Protocol):
    """Protocol for rerankers."""

    model_id: str
    model_revision: Optional[str] = None

    def rerank(self, query: str, candidates: List[str], top_k: int = 10) -> List[str]:
        """Rerank candidates for a query."""
        ...


class JudgeLLMProtocol(Protocol):
    """Protocol for evaluation judge LLMs."""

    provider: str
    model_id: str
    model_revision: Optional[str] = None
    temperature: float
    max_tokens: Optional[int]

    def generate(self, prompt: str) -> str:
        """Generate text from the judge."""
        ...


class EmbeddingModelProtocol(Protocol):
    """Protocol for embedding models."""

    name: str
    dimensions: int

    def embed_query(self, query: str) -> List[float]:
        """Embed a query string."""
        ...

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        ...