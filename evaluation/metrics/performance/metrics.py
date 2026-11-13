from typing import List, Dict, Any, Optional, NamedTuple
from evaluation.contracts.models import EvalSample, RetrievalResult


class LatencyRecord(NamedTuple):
    """Latency measurement for a pipeline step."""
    step: str
    milliseconds: float
    timestamp: float


class TokenUsage(NamedTuple):
    """Token usage measurement."""
    input_tokens: int
    output_tokens: int
    total_tokens: int


class PerformanceMetrics(NamedTuple):
    """Performance benchmark metrics."""
    retrieval_latency_ms: float
    reranking_latency_ms: Optional[float]
    generation_latency_ms: Optional[float]
    evaluation_latency_ms: Optional[float]
    end_to_end_latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_load_time_ms: Optional[float]


def compute_latency_metrics(
    latencies: Dict[str, float],
) -> Dict[str, float]:
    """Compute latency metrics from a dictionary of step timings."""
    return {
        step: time
        for step, time in latencies.items()
        if isinstance(time, (int, float))
    }


def token_usage_from_samples(
    samples: List[EvalSample],
) -> TokenUsage:
    """Compute total token usage from evaluation samples."""
    total_input = 0
    total_output = 0

    for sample in samples:
        # Count from metadata if available
        metadata = getattr(sample, 'metadata', {}) or {}
        total_input += metadata.get('input_tokens', 0)
        total_output += metadata.get('output_tokens', 0)

    return TokenUsage(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
    )


def performance_summary(
    latencies: Dict[str, float],
    model_load_time: Optional[float] = None,
) -> PerformanceMetrics:
    """Create a performance summary from latency measurements."""
    total = sum(latencies.values()) if latencies else 0.0

    return PerformanceMetrics(
        retrieval_latency_ms=latencies.get("retrieval", 0.0),
        reranking_latency_ms=latencies.get("reranking", None),
        generation_latency_ms=latencies.get("generation", None),
        evaluation_latency_ms=latencies.get("evaluation", None),
        end_to_end_latency_ms=total,
        input_tokens=0,  # Will be filled in
        output_tokens=0,
        total_tokens=0,
        model_load_time_ms=model_load_time,
    )