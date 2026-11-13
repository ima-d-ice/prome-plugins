"""Thin, optional instrumentation wrappers for Prome RAG components.

None of these functions import the wrapped packages; each works on any
duck-typed object that exposes the expected method names. Wrapping never
mutates the target — a lightweight proxy is returned that forwards
attribute access and instruments bound methods.

All wrappers are transparent when observability is disabled (spans become
no-ops) and never alter call signatures or return values.

Integration seam summary (method names matched case-sensitively):

=================================  =====================================  ==========================
Helper                             Wrapped method(s)                      Span / metrics
=================================  =====================================  ==========================
instrument_parser                  parse                                  ingestion.parse + documents_parsed / parse_failures
instrument_chunker                 chunk                                  ingestion.chunk + chunks_created
instrument_embedding               embed_documents, embed_query,          ingestion.embed + embedding_requests / embedding_failures
                                   embed_queries
instrument_vector_store            upsert, search                         ingestion.index_dense / retrieval.dense + index counters
                                   (``index="lexical"`` for lexical)      + candidates / retrieval_failures
instrument_indexing                index, index_document, index_chunks,   ingestion.index_dense + index_consistency_failures on error
                                   run, upsert
instrument_dense_retriever         retrieve                               retrieval.dense + latency / candidates / empty_results / failures
instrument_bm25_retriever          retrieve                               retrieval.bm25 + latency / candidates / empty_results / failures
instrument_hybrid_retriever        retrieve                               retrieval.hybrid + latency / empty_results / failures
instrument_fusion                  fuse                                   retrieval.rrf_fusion + fused_candidates
instrument_hierarchical_retriever  retrieve                               retrieval.hierarchical + latency / empty_results / failures
instrument_parent_stage            search, search_text                    retrieval.parent_search + hierarchical_parent_candidates
instrument_child_stage             search, search_text                    retrieval.child_search + hierarchical_child_candidates
instrument_context_expander        expand, expand_context                 retrieval.context_expand
instrument_reranker                rerank, (_make_pairs -> prepare)       rerank.inference (+ rerank.prepare) + candidates_reranked / duration / failures
instrument_generation              generate, chat, complete,               generation.request + requests / duration / errors /
                                   generate_response, route, __call__       token usage (only if reported) / fallbacks (if flagged)
instrument_guardrails              evaluate, check, classify, scan,       guardrails.* (method-name heuristic) + decision metrics
                                   safeguard, prompt_guard, check_prompt,
instrument_evaluation              evaluate, run, run_experiment, judge    evaluation.ragas / evaluation.judge_call + runs / samples / judge tokens / failed samples
instrument_loader                  load, load_file, load_document,        ingestion.load + documents_loaded / documents_rejected
                                   ingest
instrument_orchestration           run, run_pipeline, answer, ask,        orchestration.pipeline + active_requests up/down
                                   query, invoke, __call__
=================================  =====================================  ==========================

For packages not present in the installed distribution (generation,
guardrails, evaluation, loaders — historically shipped on separate
feature branches), the generic helpers still work: they instrument
whatever object is passed to them. To wire a new component class, call
:func:`instrument_component` with the method map you need, or open a span
and metrics directly using ``observability.instrumentation.operation_span``.
"""
from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..contracts.models import SpanKind
from ..metrics.rag_metrics import (
    MetricNames,
    increment_active_requests,
    record_generation_usage,
)
from ..metrics.registry import default_registry
from ..tracing.spans import SpanNames
from .context_managers import operation_span

logger = logging.getLogger("prome_rag.instrumentation")


def _counter(name: str, description: str, unit: str | None = None):
    return default_registry().counter(name, description=description, unit=unit)


def _histogram(name: str, description: str, unit: str | None = None):
    return default_registry().histogram(name, description=description, unit=unit)


def _len(obj: Any) -> int | None:
    try:
        if obj is None:
            return None
        if isinstance(obj, int):
            return obj
        return len(obj)
    except TypeError:
        return None


def _result_items(result: Any) -> int | None:
    """Best-effort count of items produced by a retriever/reranker call."""
    if result is None:
        return None
    for attr in ("results", "items", "reranked_items", "documents", "samples"):
        value = getattr(result, attr, None)
        count = _len(value)
        if count is not None:
            return count
    if isinstance(result, dict):
        for attr in ("results", "items", "documents", "samples"):
            count = _len(result.get(attr))
            if count is not None:
                return count
    return _len(result)


_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "prompt_eval_count",
    "eval_count",
)


def _result_usage(result: Any) -> dict | None:
    """Extract a provider usage report from a response object/dict.

    Accepts nested ``usage`` dicts (OpenAI/Groq style), a response dict or
    object carrying top-level token attributes (GenerationResponse style),
    or Ollama's flat response fields. Returns only data the provider
    actually reported; nothing is estimated.
    """
    if result is None:
        return None
    usage = getattr(result, "usage", None)
    if isinstance(usage, Mapping):
        return dict(usage)
    if isinstance(result, Mapping):
        nested = result.get("usage")
        if isinstance(nested, Mapping):
            return dict(nested)
        top = {key: result[key] for key in _USAGE_KEYS if result.get(key) is not None}
        return top or None
    top = {
        key: getattr(result, key)
        for key in _USAGE_KEYS
        if getattr(result, key, None) is not None
    }
    return top or None


def _result_attr(result: Any, names: Iterable[str]):
    if result is None:
        return None
    if isinstance(result, dict):
        for name in names:
            if name in result:
                return result[name]
        return None
    for name in names:
        value = getattr(result, name, None)
        if value is not None:
            return value
    return None


def _usage_provider_model(result: Any, fallback_obj: Any) -> tuple[str, str]:
    provider = _result_attr(result, ("provider", "provider_name")) or _result_attr(
        fallback_obj, ("provider", "provider_name")
    )
    model = _result_attr(result, ("model", "model_id", "model_name")) or _result_attr(
        fallback_obj, ("model", "model_id", "model_name")
    )
    return str(provider or "unknown"), str(model or "unknown")


def instrument_component(
    target: Any,
    *,
    component: str,
    wrappers: dict[str, Callable[[Callable], Callable]],
):
    """Build a transparent proxy wrapping selected methods of ``target``.

    ``wrappers`` maps method names to decorator factories. Methods that do
    not exist on ``target`` are skipped silently (recorded in the wrapper's
    ``__wrapped_methods__`` attribute for introspection).
    """

    wrapped: dict[str, str] = {}

    class InstrumentedProxy:
        def __init__(self, target: Any) -> None:
            object.__setattr__(self, "_target", target)
            object.__setattr__(self, "_local_names", set())

        def __getattr__(self, name: str) -> Any:
            return getattr(object.__getattribute__(self, "_target"), name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name in object.__getattribute__(self, "_local_names"):
                object.__setattr__(self, name, value)
                return
            setattr(object.__getattribute__(self, "_target"), name, value)

        def __repr__(self) -> str:
            return f"InstrumentedProxy(component={component!r}, target={self._target!r})"

        def uninstrument(self) -> Any:
            """Return the original object (the proxy never mutates it)."""
            return object.__getattribute__(self, "_target")

    proxy = InstrumentedProxy(target)
    local_names = set()
    for method_name, factory in wrappers.items():
        original = getattr(target, method_name, None)
        if not callable(original):
            continue
        wrapper = factory(original)
        # The wrapper lives on the proxy only; the wrapped target keeps its
        # original methods untouched.
        object.__setattr__(proxy, method_name, wrapper)
        local_names.add(method_name)
        wrapped[method_name] = method_name
    object.__setattr__(proxy, "_local_names", local_names | {"_target", "_local_names", "__wrapped_methods__"})
    object.__setattr__(proxy, "__wrapped_methods__", tuple(sorted(wrapped)))
    logger.debug(
        "instrumented %s (%s): methods=%s",
        component,
        type(target).__name__,
        sorted(wrapped),
    )
    return proxy


def _make_span_wrapper(
    original: Callable,
    *,
    span_name: str,
    component: str,
    kind: SpanKind,
    on_success: Callable[[Any, float], None] | None = None,
    on_error: Callable[[BaseException, float], None] | None = None,
    extra_attributes: dict | None = None,
) -> Callable:
    operation = span_name.rsplit(".", 1)[-1]
    attributes = {"component": component, "operation": operation}
    if extra_attributes:
        attributes.update(extra_attributes)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        with operation_span(
            span_name,
            component=component,
            operation=operation,
            kind=kind,
            attributes=attributes,
        ) as span:
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                span.set_attribute("outcome", "error")
                if on_error is not None:
                    try:
                        on_error(exc, time.perf_counter() - start)
                    except Exception:
                        logger.debug("metrics on_error callback failed", exc_info=True)
                raise
            if on_success is not None:
                try:
                    on_success(result, time.perf_counter() - start)
                except Exception:
                    logger.debug("metrics on_success callback failed", exc_info=True)
            return result

    return wrapper


# --- ingestion ------------------------------------------------------------


def instrument_parser(parser: Any, *, component: str = "parser") -> Any:
    def on_success(_result: Any, _elapsed: float) -> None:
        _counter(
            MetricNames.INGESTION_DOCUMENTS_PARSED,
            "Documents parsed successfully",
            "{document}",
        ).inc({"component": component, "outcome": "ok"})

    def on_error(_exc: BaseException, _elapsed: float) -> None:
        _counter(
            MetricNames.INGESTION_PARSE_FAILURES,
            "Document parse failures",
            "{failure}",
        ).inc({"component": component, "outcome": "error"})

    return instrument_component(
        parser,
        component=component,
        wrappers={
            "parse": lambda fn: _make_span_wrapper(
                fn,
                span_name=SpanNames.INGESTION_PARSE,
                component=component,
                kind=SpanKind.INTERNAL,
                on_success=on_success,
                on_error=on_error,
            ),
        },
    )


def instrument_chunker(chunker: Any, *, component: str = "chunker") -> Any:
    def on_success(result: Any, _elapsed: float) -> None:
        created = _len(result)
        _counter(
            MetricNames.INGESTION_CHUNKS_CREATED,
            "Chunks produced",
            "{chunk}",
        ).add(float(created if created is not None else 1), {"component": component})

    return instrument_component(
        chunker,
        component=component,
        wrappers={
            "chunk": lambda fn: _make_span_wrapper(
                fn,
                span_name=SpanNames.INGESTION_CHUNK,
                component=component,
                kind=SpanKind.INTERNAL,
                on_success=on_success,
            ),
        },
    )


def instrument_embedding(embedding: Any, *, component: str = "embedding") -> Any:
    def on_success(_result: Any, _elapsed: float) -> None:
        _counter(
            MetricNames.INGESTION_EMBEDDING_REQUESTS,
            "Embedding requests issued",
            "{request}",
        ).inc({"component": component, "outcome": "ok"})

    def on_error(_exc: BaseException, _elapsed: float) -> None:
        _counter(
            MetricNames.INGESTION_EMBEDDING_REQUESTS,
            "Embedding requests issued",
            "{request}",
        ).inc({"component": component, "outcome": "error"})
        _counter(
            MetricNames.INGESTION_EMBEDDING_FAILURES,
            "Embedding requests that failed",
            "{failure}",
        ).inc({"component": component, "outcome": "error"})

    wrappers = {}
    for method in ("embed_documents", "embed_query", "embed_queries"):
        wrappers[method] = lambda fn: _make_span_wrapper(
            fn,
            span_name=SpanNames.INGESTION_EMBED,
            component=component,
            kind=SpanKind.CLIENT,
            on_success=on_success,
            on_error=on_error,
            extra_attributes={"model": getattr(embedding, "model_id", None) or "unknown"},
        )
    return instrument_component(embedding, component=component, wrappers=wrappers)


def instrument_vector_store(
    store: Any,
    *,
    index: str = "dense",
    component: str = "vector_store",
) -> Any:
    is_dense = index != "lexical"
    upsert_span = SpanNames.INGESTION_INDEX_DENSE if is_dense else SpanNames.INGESTION_INDEX_LEXICAL
    upsert_metric = (
        MetricNames.INGESTION_DENSE_INDEX_UPSERTS
        if is_dense
        else MetricNames.INGESTION_LEXICAL_INDEX_UPSERTS
    )

    def upsert_success(result: Any, _elapsed: float) -> None:
        # `records` is usually the first positional/keyword argument.
        _counter(upsert_metric, "Index upserts", "{upsert}").inc(
            {"component": component, "outcome": "ok"}
        )

    def upsert_error(_exc: BaseException, _elapsed: float) -> None:
        _counter(upsert_metric, "Index upserts", "{upsert}").inc(
            {"component": component, "outcome": "error"}
        )

    def search_success(result: Any, _elapsed: float) -> None:
        candidates = _result_items(result)
        if candidates is not None:
            metric = (
                MetricNames.RETRIEVAL_DENSE_CANDIDATES
                if is_dense
                else MetricNames.RETRIEVAL_BM25_CANDIDATES
            )
            _counter(metric, "Candidates returned by search", "{candidate}").add(
                float(candidates), {"component": component}
            )

    def search_error(_exc: BaseException, _elapsed: float) -> None:
        _counter(
            MetricNames.RETRIEVAL_FAILURES,
            "Retrieval operations that failed",
            "{failure}",
        ).inc({"component": component, "outcome": "error"})

    wrappers = {
        "upsert": lambda fn: _make_span_wrapper(
            fn,
            span_name=upsert_span,
            component=component,
            kind=SpanKind.CLIENT,
            on_success=upsert_success,
            on_error=upsert_error,
        ),
        "search": lambda fn: _make_span_wrapper(
            fn,
            span_name=SpanNames.RETRIEVAL_DENSE if is_dense else SpanNames.RETRIEVAL_BM25,
            component=component,
            kind=SpanKind.CLIENT,
            on_success=search_success,
            on_error=search_error,
        ),
    }
    return instrument_component(store, component=component, wrappers=wrappers)


def instrument_indexing(coordinator: Any, *, component: str = "indexing") -> Any:
    def on_error(_exc: BaseException, _elapsed: float) -> None:
        _counter(
            MetricNames.INGESTION_INDEX_CONSISTENCY_FAILURES,
            "Dense/lexical index consistency failures",
            "{failure}",
        ).inc({"component": component, "outcome": "error"})

    wrappers = {}
    for method in ("index", "index_document", "index_chunks", "upsert", "run"):
        wrappers[method] = lambda fn: _make_span_wrapper(
            fn,
            span_name=SpanNames.INGESTION_INDEX_DENSE,
            component=component,
            kind=SpanKind.INTERNAL,
            on_error=on_error,
        )
    return instrument_component(coordinator, component=component, wrappers=wrappers)


# --- retrieval ------------------------------------------------------------


def _retriever_wrapper(
    retriever: Any,
    *,
    span_name: str,
    component: str,
    strategy: str,
    candidates_metric: str,
    empty_metric: str = MetricNames.RETRIEVAL_EMPTY_RESULTS,
):
    def on_success(result: Any, elapsed: float) -> None:
        _histogram(
            MetricNames.RETRIEVAL_LATENCY,
            "Retrieval latency in seconds",
            "s",
        ).record(elapsed, {"component": component, "retrieval_strategy": strategy})
        _counter(
            MetricNames.RETRIEVAL_QUERIES,
            "Retrieval queries",
            "{query}",
        ).inc({"component": component, "retrieval_strategy": strategy})
        items = _result_items(result)
        if items is not None:
            _counter(candidates_metric, "Candidates returned", "{candidate}").add(
                float(items), {"component": component, "retrieval_strategy": strategy}
            )
            if items == 0:
                _counter(empty_metric, "Retrievals returning zero results", "{query}").inc(
                    {"component": component, "retrieval_strategy": strategy}
                )

    def on_error(_exc: BaseException, elapsed: float) -> None:
        _histogram(
            MetricNames.RETRIEVAL_LATENCY,
            "Retrieval latency in seconds",
            "s",
        ).record(elapsed, {"component": component, "retrieval_strategy": strategy, "outcome": "error"})
        _counter(
            MetricNames.RETRIEVAL_FAILURES,
            "Retrieval operations that failed",
            "{failure}",
        ).inc({"component": component, "retrieval_strategy": strategy, "outcome": "error"})

    return instrument_component(
        retriever,
        component=component,
        wrappers={
            "retrieve": lambda fn: _make_span_wrapper(
                fn,
                span_name=span_name,
                component=component,
                kind=SpanKind.INTERNAL,
                on_success=on_success,
                on_error=on_error,
                extra_attributes={"retrieval.strategy": strategy},
            ),
        },
    )


def instrument_dense_retriever(retriever: Any, *, component: str = "retrieval_dense") -> Any:
    return _retriever_wrapper(
        retriever,
        span_name=SpanNames.RETRIEVAL_DENSE,
        component=component,
        strategy="dense",
        candidates_metric=MetricNames.RETRIEVAL_DENSE_CANDIDATES,
    )


def instrument_bm25_retriever(retriever: Any, *, component: str = "retrieval_bm25") -> Any:
    return _retriever_wrapper(
        retriever,
        span_name=SpanNames.RETRIEVAL_BM25,
        component=component,
        strategy="bm25",
        candidates_metric=MetricNames.RETRIEVAL_BM25_CANDIDATES,
    )


def instrument_hybrid_retriever(retriever: Any, *, component: str = "retrieval_hybrid") -> Any:
    return _retriever_wrapper(
        retriever,
        span_name="retrieval.hybrid",
        component=component,
        strategy="hybrid",
        candidates_metric=MetricNames.RETRIEVAL_FUSED_CANDIDATES,
    )


def instrument_fusion(fusion: Any, *, component: str = "retrieval_fusion") -> Any:
    def on_success(result: Any, _elapsed: float) -> None:
        fused = _result_items(result)
        if fused is not None:
            _counter(
                MetricNames.RETRIEVAL_FUSED_CANDIDATES,
                "Candidates after fusion",
                "{candidate}",
            ).add(float(fused), {"component": component})

    return instrument_component(
        fusion,
        component=component,
        wrappers={
            "fuse": lambda fn: _make_span_wrapper(
                fn,
                span_name=SpanNames.RETRIEVAL_RRF_FUSION,
                component=component,
                kind=SpanKind.INTERNAL,
                on_success=on_success,
            ),
        },
    )


def instrument_hierarchical_retriever(retriever: Any, *, component: str = "retrieval_hierarchical") -> Any:
    return _retriever_wrapper(
        retriever,
        span_name="retrieval.hierarchical",
        component=component,
        strategy="hierarchical",
        candidates_metric=MetricNames.RETRIEVAL_HIERARCHICAL_CHILDREN,
    )


def _stage_wrapper(
    stage: Any,
    *,
    span_name: str,
    component: str,
    candidates_metric: str,
):
    def on_success(result: Any, _elapsed: float) -> None:
        items = _result_items(result)
        if items is not None:
            _counter(candidates_metric, "Candidates selected by stage", "{candidate}").add(
                float(items), {"component": component}
            )

    wrappers = {}
    for method in ("search", "search_text"):
        wrappers[method] = lambda fn: _make_span_wrapper(
            fn,
            span_name=span_name,
            component=component,
            kind=SpanKind.INTERNAL,
            on_success=on_success,
        )
    return instrument_component(stage, component=component, wrappers=wrappers)


def instrument_parent_stage(stage: Any, *, component: str = "retrieval_parent_stage") -> Any:
    return _stage_wrapper(
        stage,
        span_name=SpanNames.RETRIEVAL_PARENT_SEARCH,
        component=component,
        candidates_metric=MetricNames.RETRIEVAL_HIERARCHICAL_PARENTS,
    )


def instrument_child_stage(stage: Any, *, component: str = "retrieval_child_stage") -> Any:
    return _stage_wrapper(
        stage,
        span_name=SpanNames.RETRIEVAL_CHILD_SEARCH,
        component=component,
        candidates_metric=MetricNames.RETRIEVAL_HIERARCHICAL_CHILDREN,
    )


def instrument_context_expander(expander: Any, *, component: str = "retrieval_context_expander") -> Any:
    wrappers = {}
    for method in ("expand", "expand_context"):
        wrappers[method] = lambda fn: _make_span_wrapper(
            fn,
            span_name=SpanNames.RETRIEVAL_CONTEXT_EXPAND,
            component=component,
            kind=SpanKind.INTERNAL,
        )
    return instrument_component(expander, component=component, wrappers=wrappers)


# --- reranking -----------------------------------------------------------


def instrument_reranker(reranker: Any, *, component: str = "reranker") -> Any:
    def on_success(result: Any, elapsed: float) -> None:
        _histogram(
            MetricNames.RERANK_DURATION,
            "Reranking duration in seconds",
            "s",
        ).record(elapsed, {"component": component})
        scored = _result_attr(result, ("candidates_scored",))
        amount = None
        try:
            amount = float(scored) if scored is not None else _result_items(result)
        except (TypeError, ValueError):
            amount = None
        _counter(
            MetricNames.RERANK_CANDIDATES,
            "Candidates reranked",
            "{candidate}",
        ).add(amount if amount is not None else 1.0, {"component": component})

    def on_error(_exc: BaseException, elapsed: float) -> None:
        _histogram(
            MetricNames.RERANK_DURATION,
            "Reranking duration in seconds",
            "s",
        ).record(elapsed, {"component": component, "outcome": "error"})
        _counter(
            MetricNames.RERANK_FAILURES,
            "Reranking requests that failed",
            "{failure}",
        ).inc({"component": component, "outcome": "error"})

    wrappers = {
        "rerank": lambda fn: _make_span_wrapper(
            fn,
            span_name=SpanNames.RERANK_INFERENCE,
            component=component,
            kind=SpanKind.CLIENT,
            on_success=on_success,
            on_error=on_error,
            extra_attributes={
                "model": _result_attr(reranker, ("model_id", "model")) or "unknown",
            },
        ),
    }
    # Optional sub-stage wrappers when present.
    wrappers["_make_pairs"] = lambda fn: _make_span_wrapper(
        fn,
        span_name=SpanNames.RERANK_PREPARE,
        component=component,
        kind=SpanKind.INTERNAL,
    )
    return instrument_component(reranker, component=component, wrappers=wrappers)


# --- generation ----------------------------------------------------------


_GENERATION_METHODS = (
    "generate",
    "chat",
    "complete",
    "generate_response",
    "generate_text",
    "route",
    "__call__",
)


def instrument_generation(client: Any, *, component: str = "generation") -> Any:
    fallback_flags = ("fallback_used", "used_fallback", "fell_back", "fallback")

    def make_wrapper(fn: Callable) -> Callable:
        span_name = SpanNames.GENERATION_REQUEST

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            with operation_span(
                span_name,
                component=component,
                operation="request",
                kind=SpanKind.CLIENT,
                attributes={"component": component, "operation": "request"},
            ) as span:
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    from ..tracing.spans import categorize_exception

                    provider, model = _usage_provider_model(None, client)
                    category = categorize_exception(exc)
                    span.set_attribute("provider", provider)
                    span.set_attribute("model", model)
                    span.set_attribute("outcome", "error")
                    span.set_attribute("error.category", category)
                    _counter(
                        MetricNames.GENERATION_REQUESTS,
                        "Generation requests",
                        "{request}",
                    ).inc({"provider": provider, "model": model, "outcome": "error"})
                    _counter(
                        MetricNames.GENERATION_ERRORS,
                        "Generation requests that failed",
                        "{error}",
                    ).inc({"provider": provider, "model": model, "error_category": category})
                    raise
                provider, model = _usage_provider_model(result, client)
                span.set_attribute("provider", provider)
                span.set_attribute("model", model)
                usage = _result_usage(result)
                if usage is not None:
                    resolved = record_generation_usage(
                        usage, provider=provider, model=model, outcome="ok"
                    )
                    for key, value in resolved.items():
                        span.set_attribute(f"usage.{key}", value)
                _counter(
                    MetricNames.GENERATION_REQUESTS,
                    "Generation requests",
                    "{request}",
                ).inc({"provider": provider, "model": model, "outcome": "ok"})
                _histogram(
                    MetricNames.GENERATION_DURATION,
                    "Generation duration in seconds",
                    "s",
                ).record(time.perf_counter() - start, {"provider": provider, "model": model, "outcome": "ok"})
                if any(bool(_result_attr(result, (flag,))) for flag in fallback_flags):
                    _counter(
                        MetricNames.GENERATION_FALLBACKS,
                        "Times a fallback provider was used",
                        "{fallback}",
                    ).inc({"provider": provider, "model": model})
                    span.set_attribute("generation.fallback", True)
                return result

        return wrapper

    wrappers = {method: make_wrapper for method in _GENERATION_METHODS}
    return instrument_component(client, component=component, wrappers=wrappers)


# --- guardrails ----------------------------------------------------------


def _guardrail_span_name(method: str) -> str:
    lowered = method.lower()
    if "prompt" in lowered or "guard" in lowered:
        return SpanNames.GUARDRAILS_PROMPT_GUARD
    if "citation" in lowered:
        return SpanNames.GUARDRAILS_CITATION_VALIDATE
    if "privacy" in lowered or "pii" in lowered:
        return SpanNames.GUARDRAILS_PRIVACY_DETECT
    if "access" in lowered:
        return SpanNames.GUARDRAILS_ACCESS_CONTROL
    if "trust" in lowered:
        return SpanNames.GUARDRAILS_SOURCE_TRUST
    if "enforce" in lowered:
        return SpanNames.GUARDRAILS_ENFORCEMENT
    return SpanNames.GUARDRAILS_SAFEGUARD


_DECISION_MAP = {
    "allow": "allow",
    "allowed": "allow",
    "pass": "allow",
    "passed": "allow",
    "warn": "warn",
    "warning": "warn",
    "block": "block",
    "blocked": "block",
    "deny": "block",
    "denied": "block",
    "reject": "block",
    "review": "review",
    "requires_review": "review",
    "manual_review": "review",
    "benign": "benign",
    "malicious": "malicious",
    "attack": "malicious",
    "injection": "malicious",
}


def _guardrail_decision(result: Any) -> str | None:
    raw = _result_attr(
        result,
        ("decision", "action", "verdict", "status", "classification", "label", "result"),
    )
    if raw is None:
        return None
    lowered = str(raw).strip().lower()
    for needle, decision in _DECISION_MAP.items():
        if needle in lowered:
            return decision
    return None


def _guardrail_metric(decision: str | None, component: str) -> None:
    if decision is None:
        return
    mapping = {
        "allow": (MetricNames.GUARDRAILS_DECISIONS_ALLOW, "Allow decisions"),
        "warn": (MetricNames.GUARDRAILS_DECISIONS_WARN, "Warn decisions"),
        "block": (MetricNames.GUARDRAILS_BLOCKS, "Blocked generations"),
        "review": (MetricNames.GUARDRAILS_REVIEWS, "Review decisions"),
        "benign": (MetricNames.GUARDRAILS_BENIGN_CLASSIFICATIONS, "Benign classifications"),
        "malicious": (MetricNames.GUARDRAILS_MALICIOUS_CLASSIFICATIONS, "Malicious classifications"),
    }
    metric = mapping.get(decision)
    if metric is not None:
        _counter(metric[0], metric[1], "{decision}").inc({"component": component, "outcome": decision})


def instrument_guardrails(guard: Any, *, component: str = "guardrails") -> Any:
    metrics_for = {
        "prompt_guard": MetricNames.GUARDRAILS_PROMPT_GUARD_REQUESTS,
        "check_prompt": MetricNames.GUARDRAILS_PROMPT_GUARD_REQUESTS,
        "safeguard": MetricNames.GUARDRAILS_SAFEGUARD_INVOCATIONS,
        "evaluate": MetricNames.GUARDRAILS_SAFEGUARD_INVOCATIONS,
        "check": MetricNames.GUARDRAILS_SAFEGUARD_INVOCATIONS,
        "classify": MetricNames.GUARDRAILS_PROMPT_GUARD_REQUESTS,
        "scan": MetricNames.GUARDRAILS_PRIVACY_DETECTIONS,
    }

    def make_wrapper(method: str) -> Callable[[Callable], Callable]:
        def factory(original: Callable) -> Callable:
            span_name = _guardrail_span_name(method)

            @functools.wraps(original)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with operation_span(
                    span_name,
                    component=component,
                    operation=method,
                    kind=SpanKind.INTERNAL,
                    attributes={"component": component, "operation": method},
                ) as span:
                    metric = metrics_for.get(method)
                    if metric:
                        _counter(metric, "Guardrail invocations", "{invocation}").inc(
                            {"component": component, "operation": method}
                        )
                    try:
                        result = original(*args, **kwargs)
                    except Exception as exc:
                        _guardrail_metric("block" if _looks_like_block(exc) else None, component)
                        raise
                    decision = _guardrail_decision(result)
                    if decision:
                        span.set_attribute("guardrails.decision", decision)
                    _guardrail_metric(decision, component)
                    if _result_attr(result, ("pii_detected", "has_pii", "contains_pii")):
                        _counter(
                            MetricNames.GUARDRAILS_PRIVACY_DETECTIONS,
                            "Privacy detections",
                            "{detection}",
                        ).inc({"component": component})
                    citation_failure = _result_attr(
                        result, ("citation_valid", "citations_valid")
                    )
                    if citation_failure is False:
                        _counter(
                            MetricNames.GUARDRAILS_CITATION_FAILURES,
                            "Citation validation failures",
                            "{failure}",
                        ).inc({"component": component})
                    citation_failures = _result_attr(result, ("citation_failures",))
                    try:
                        if citation_failures and float(citation_failures) > 0:
                            _counter(
                                MetricNames.GUARDRAILS_CITATION_FAILURES,
                                "Citation validation failures",
                                "{failure}",
                            ).add(float(citation_failures), {"component": component})
                    except (TypeError, ValueError):
                        pass
                    return result

            return wrapper

        return factory

    wrappers = {
        method: make_wrapper(method)
        for method in (
            "evaluate",
            "check",
            "classify",
            "scan",
            "safeguard",
            "prompt_guard",
            "check_prompt",
            "privacy_scan",
            "check_pii",
            "validate_citations",
            "check_citation",
            "access_control",
            "check_access",
        )
    }
    return instrument_component(guard, component=component, wrappers=wrappers)


def _looks_like_block(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        token in text
        for token in ("block", "guardrail", "safeguard", "policy", "injection", "jailbreak")
    )


# --- evaluation ----------------------------------------------------------


def instrument_evaluation(evaluator: Any, *, component: str = "evaluation") -> Any:
    def on_success(result: Any, _elapsed: float, method: str) -> None:
        if method == "judge":
            usage = _result_usage(result)
            provider, model = _usage_provider_model(result, evaluator)
            if usage is not None:
                total = None
                try:
                    resolved_total = usage.get("total_tokens")
                    if resolved_total is None:
                        inp = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                        out = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                        resolved_total = (inp or 0) + (out or 0)
                    total = float(resolved_total)
                except (TypeError, ValueError):
                    total = None
                if total:
                    _counter(
                        MetricNames.EVALUATION_JUDGE_TOKENS,
                        "Tokens consumed by LLM judge calls",
                        "{token}",
                    ).add(total, {"provider": provider, "model": model})
            return
        _counter(
            MetricNames.EVALUATION_EXPERIMENT_RUNS,
            "Evaluation experiment runs",
            "{run}",
        ).inc({"component": component})
        samples = _result_items(result)
        _counter(
            MetricNames.EVALUATION_SAMPLES_EVALUATED,
            "Samples evaluated",
            "{sample}",
        ).add(float(samples if samples is not None else 1), {"component": component})

    def on_error(_exc: BaseException, _elapsed: float) -> None:
        _counter(
            MetricNames.EVALUATION_FAILED_SAMPLES,
            "Samples that failed evaluation",
            "{failure}",
        ).inc({"component": component, "outcome": "error"})

    wrappers = {}
    for method in ("evaluate", "run", "run_experiment"):
        wrappers[method] = lambda fn: _make_span_wrapper(
            fn,
            span_name=SpanNames.EVALUATION_RAGAS,
            component=component,
            kind=SpanKind.INTERNAL,
            on_success=lambda result, elapsed, meth=method: on_success(result, elapsed, meth),
            on_error=on_error,
        )
    wrappers["judge"] = lambda fn: _make_span_wrapper(
        fn,
        span_name=SpanNames.EVALUATION_JUDGE_CALL,
        component=component,
        kind=SpanKind.CLIENT,
        on_success=lambda result, elapsed: on_success(result, elapsed, "judge"),
        on_error=on_error,
    )
    return instrument_component(evaluator, component=component, wrappers=wrappers)


# --- loaders ---------------------------------------------------------------


def instrument_loader(loader: Any, *, component: str = "loader") -> Any:
    def on_success(_result: Any, _elapsed: float) -> None:
        _counter(
            MetricNames.INGESTION_DOCUMENTS_LOADED,
            "Documents loaded",
            "{document}",
        ).inc({"component": component, "outcome": "ok"})

    def on_error(_exc: BaseException, _elapsed: float) -> None:
        _counter(
            MetricNames.INGESTION_DOCUMENTS_REJECTED,
            "Documents rejected at load time",
            "{document}",
        ).inc({"component": component, "outcome": "error"})

    wrappers = {}
    for method in ("load", "load_file", "load_document", "ingest"):
        wrappers[method] = lambda fn: _make_span_wrapper(
            fn,
            span_name=SpanNames.INGESTION_LOAD,
            component=component,
            kind=SpanKind.CLIENT,
            on_success=on_success,
            on_error=on_error,
        )
    return instrument_component(loader, component=component, wrappers=wrappers)


# --- orchestration ---------------------------------------------------------


def instrument_orchestration(pipeline: Any, *, component: str = "orchestration") -> Any:
    def on_success(_result: Any, _elapsed: float) -> None:
        increment_active_requests(-1, component=component)

    def on_error(_exc: BaseException, _elapsed: float) -> None:
        increment_active_requests(-1, component=component)

    def wrap(fn: Callable) -> Callable:
        inner = _make_span_wrapper(
            fn,
            span_name=SpanNames.ORCHESTRATION_PIPELINE,
            component=component,
            kind=SpanKind.INTERNAL,
            on_success=on_success,
            on_error=on_error,
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            increment_active_requests(1, component=component)
            return inner(*args, **kwargs)

        return wrapper

    wrappers = {}
    for method in ("run", "run_pipeline", "answer", "ask", "query", "invoke", "__call__"):
        wrappers[method] = wrap
    return instrument_component(pipeline, component=component, wrappers=wrappers)
