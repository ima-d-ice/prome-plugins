"""Canonical Prome RAG metric catalog and convenience recorders.

All names are prefixed ``prome_rag.`` and use bounded labels only
(``component``, ``operation``, ``provider``, ``model``, ``outcome``,
``error_category``, ``retrieval_strategy``, ``stage``, ``status``).

Token counters are recorded only from provider usage reports supplied by
the caller; nothing here estimates or invents token counts.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .registry import default_registry


class MetricNames:
    # Ingestion
    INGESTION_DOCUMENTS_DISCOVERED = "prome_rag.ingestion.documents_discovered"
    INGESTION_DOCUMENTS_LOADED = "prome_rag.ingestion.documents_loaded"
    INGESTION_DOCUMENTS_REJECTED = "prome_rag.ingestion.documents_rejected"
    INGESTION_DOCUMENTS_PARSED = "prome_rag.ingestion.documents_parsed"
    INGESTION_PARSE_FAILURES = "prome_rag.ingestion.parse_failures"
    INGESTION_CHUNKS_CREATED = "prome_rag.ingestion.chunks_created"
    INGESTION_EMBEDDING_REQUESTS = "prome_rag.ingestion.embedding_requests"
    INGESTION_EMBEDDING_FAILURES = "prome_rag.ingestion.embedding_failures"
    INGESTION_DENSE_INDEX_UPSERTS = "prome_rag.ingestion.dense_index_upserts"
    INGESTION_LEXICAL_INDEX_UPSERTS = "prome_rag.ingestion.lexical_index_upserts"
    INGESTION_INDEX_CONSISTENCY_FAILURES = "prome_rag.ingestion.index_consistency_failures"
    INGESTION_DURATION = "prome_rag.ingestion.duration"

    # Retrieval
    RETRIEVAL_QUERIES = "prome_rag.retrieval.queries"
    RETRIEVAL_LATENCY = "prome_rag.retrieval.latency"
    RETRIEVAL_DENSE_CANDIDATES = "prome_rag.retrieval.dense_candidates"
    RETRIEVAL_BM25_CANDIDATES = "prome_rag.retrieval.bm25_candidates"
    RETRIEVAL_FUSED_CANDIDATES = "prome_rag.retrieval.fused_candidates"
    RETRIEVAL_HIERARCHICAL_PARENTS = "prome_rag.retrieval.hierarchical_parent_candidates"
    RETRIEVAL_HIERARCHICAL_CHILDREN = "prome_rag.retrieval.hierarchical_child_candidates"
    RETRIEVAL_EMPTY_RESULTS = "prome_rag.retrieval.empty_results"
    RETRIEVAL_FAILURES = "prome_rag.retrieval.failures"

    # Reranking
    RERANK_CANDIDATES = "prome_rag.rerank.candidates_reranked"
    RERANK_DURATION = "prome_rag.rerank.duration"
    RERANK_FAILURES = "prome_rag.rerank.failures"

    # Generation
    GENERATION_REQUESTS = "prome_rag.generation.requests"
    GENERATION_INPUT_TOKENS = "prome_rag.generation.input_tokens"
    GENERATION_OUTPUT_TOKENS = "prome_rag.generation.output_tokens"
    GENERATION_TOTAL_TOKENS = "prome_rag.generation.total_tokens"
    GENERATION_DURATION = "prome_rag.generation.duration"
    GENERATION_FIRST_TOKEN_LATENCY = "prome_rag.generation.first_token_latency"
    GENERATION_RETRIES = "prome_rag.generation.retries"
    GENERATION_FALLBACKS = "prome_rag.generation.fallbacks"
    GENERATION_RATE_LIMIT_WAIT = "prome_rag.generation.rate_limit_wait"
    GENERATION_ERRORS = "prome_rag.generation.errors"

    # Guardrails
    GUARDRAILS_PROMPT_GUARD_REQUESTS = "prome_rag.guardrails.prompt_guard_requests"
    GUARDRAILS_BENIGN_CLASSIFICATIONS = "prome_rag.guardrails.benign_classifications"
    GUARDRAILS_MALICIOUS_CLASSIFICATIONS = "prome_rag.guardrails.malicious_classifications"
    GUARDRAILS_SAFEGUARD_INVOCATIONS = "prome_rag.guardrails.safeguard_invocations"
    GUARDRAILS_DECISIONS_ALLOW = "prome_rag.guardrails.decisions_allow"
    GUARDRAILS_DECISIONS_WARN = "prome_rag.guardrails.decisions_warn"
    GUARDRAILS_BLOCKS = "prome_rag.guardrails.blocks"
    GUARDRAILS_REVIEWS = "prome_rag.guardrails.reviews"
    GUARDRAILS_PRIVACY_DETECTIONS = "prome_rag.guardrails.privacy_detections"
    GUARDRAILS_CITATION_FAILURES = "prome_rag.guardrails.citation_failures"

    # Evaluation
    EVALUATION_EXPERIMENT_RUNS = "prome_rag.evaluation.experiment_runs"
    EVALUATION_SAMPLES_EVALUATED = "prome_rag.evaluation.samples_evaluated"
    EVALUATION_FAILED_SAMPLES = "prome_rag.evaluation.failed_samples"
    EVALUATION_RAGAS_METRIC_DURATION = "prome_rag.evaluation.ragas_metric_duration"
    EVALUATION_JUDGE_TOKENS = "prome_rag.evaluation.judge_tokens"

    # Runtime
    RUNTIME_PROCESS_MEMORY_BYTES = "prome_rag.runtime.process_memory_bytes"
    RUNTIME_CPU_PERCENT = "prome_rag.runtime.cpu_percent"
    RUNTIME_MODEL_LOAD_DURATION = "prome_rag.runtime.model_load_duration"
    RUNTIME_ACTIVE_REQUESTS = "prome_rag.runtime.active_requests"
    RUNTIME_QUEUE_LENGTH = "prome_rag.runtime.queue_length"


# Candidate keys used to locate provider usage reports (OpenAI-style and
# Ollama-style). First match wins.
_INPUT_KEYS = ("input_tokens", "prompt_tokens", "prompt_eval_count")
_OUTPUT_KEYS = ("output_tokens", "completion_tokens", "eval_count")
_TOTAL_KEYS = ("total_tokens",)


def _lookup_usage(usage: Mapping[str, Any], keys) -> float | None:
    for key in keys:
        value = usage.get(key)
        if value is None:
            # Also tolerate nested dicts: {"usage": {...}}
            nested = usage.get("usage")
            if isinstance(nested, Mapping):
                value = nested.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def record_generation_usage(
    usage: Mapping[str, Any] | None,
    *,
    provider: str,
    model: str,
    outcome: str = "ok",
) -> Mapping[str, float]:
    """Record token usage exactly as reported by a provider.

    Accepts OpenAI-style (``prompt_tokens``/``completion_tokens``),
    Ollama-style (``prompt_eval_count``/``eval_count``), or already
    normalized (``input_tokens``/``output_tokens``/``total_tokens``)
    reports. Keys that are absent are simply not recorded; ``total_tokens``
    is only synthesized when both input and output are present.

    Returns the resolved usage dict (useful in tests).
    """
    resolved: dict[str, float] = {}
    if not usage:
        return resolved

    input_tokens = _lookup_usage(usage, _INPUT_KEYS)
    output_tokens = _lookup_usage(usage, _OUTPUT_KEYS)
    total_tokens = _lookup_usage(usage, _TOTAL_KEYS)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    registry = default_registry()
    labels = {"provider": provider, "model": model, "outcome": outcome}
    if input_tokens is not None:
        resolved["input_tokens"] = input_tokens
        registry.counter(MetricNames.GENERATION_INPUT_TOKENS, unit="{token}").add(input_tokens, labels)
    if output_tokens is not None:
        resolved["output_tokens"] = output_tokens
        registry.counter(MetricNames.GENERATION_OUTPUT_TOKENS, unit="{token}").add(output_tokens, labels)
    if total_tokens is not None:
        resolved["total_tokens"] = total_tokens
        registry.counter(MetricNames.GENERATION_TOTAL_TOKENS, unit="{token}").add(total_tokens, labels)
    return resolved


def record_runtime_snapshot(*, component: str = "runtime") -> dict[str, float]:
    """Best-effort process metrics via psutil; silently skipped if absent."""
    try:
        import psutil  # type: ignore
    except Exception:
        return {}

    registry = default_registry()
    snapshot: dict[str, float] = {}
    try:
        memory = float(psutil.Process().memory_info().rss)
        snapshot["memory_rss_bytes"] = memory
        registry.gauge(MetricNames.RUNTIME_PROCESS_MEMORY_BYTES, unit="By").set(memory, {"component": component})
    except Exception:
        pass
    try:
        cpu = float(psutil.Process().cpu_percent(interval=None))
        snapshot["cpu_percent"] = cpu
        registry.gauge(MetricNames.RUNTIME_CPU_PERCENT, unit="%").set(cpu, {"component": component})
    except Exception:
        pass
    return snapshot


def increment_active_requests(delta: int, *, component: str) -> None:
    default_registry().up_down_counter(
        MetricNames.RUNTIME_ACTIVE_REQUESTS, description="Currently active top-level requests", unit="{request}"
    ).add(float(delta), {"component": component})


def set_queue_length(length: int, *, component: str, operation: str | None = None) -> None:
    labels: dict[str, str] = {"component": component}
    if operation:
        labels["operation"] = operation
    default_registry().gauge(MetricNames.RUNTIME_QUEUE_LENGTH, unit="{item}").set(float(length), labels)
