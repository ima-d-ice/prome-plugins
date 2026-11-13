# Prome RAG Observability

Backend-independent observability for the Prome RAG plugin ecosystem:
distributed tracing, metrics, structured logging, error tracking, and
provider/model usage accounting — built on the official OpenTelemetry
Python APIs, with a strict no-op default and zero required external
services.

**No monitoring server, collector, dashboard, or vendor is required.**
The three operating modes are *no-op* (default), *console* (local
debugging), and *OTLP* (export to a collector you run yourself).

## Design principles

- **Zero side effects on import.** Importing `observability` creates no
  providers, registers nothing globally, and imports nothing heavy.
- **Explicit lifecycle.** Telemetry is activated only through
  `init_observability()` / torn down with `shutdown_observability()`.
- **No-op by default.** Without configuration, every instrument is free
  (span/metrics helpers become cheap no-ops, still maintaining local
  trace/request correlation so logging works).
- **Privacy first.** Prompts and document content are never recorded by
  default; known secret patterns (API keys, authorization headers,
  passwords, tokens, emails, private keys) are redacted by default in
  logs, span events, and errors.
- **Bounded cardinality.** Metric labels are limited to a fixed allowlist
  (`component`, `operation`, `provider`, `model`, `outcome`,
  `error_category`, `retrieval_strategy`, `stage`, `status`). User ids,
  request ids, trace ids, chunk/document ids, file names, query text, and
  arbitrary metadata are dropped from metric labels (traces/logs are the
  right home for those). A strict mode raises instead of dropping.
- **Independent components.** Tracing, metrics, logging, exporters, and
  instrumentation are usable in isolation; `contracts` has no dependencies
  and `observability` never imports plugin packages at module import time.

## Install

The base install needs nothing. OpenTelemetry is an optional extra:

```bash
pip install -e '.[observability]'   # API + SDK + OTLP gRPC/HTTP exporters
pip install -e '.[dev]'             # same, plus the test suite dependencies
```

## Quick start

```python
from observability import init_observability, shutdown_observability

rt = init_observability()     # reads PROMERAG_* env vars; no-op by default
...
shutdown_observability()      # flush + detach; idempotent
```

With `PROMERAG_OBSERVABILITY_ENABLED=false` (the default), `init_observability()`
still configures structured logging, but spans and metrics are no-ops.

## Configuration (environment)

| Variable | Default | Description |
| --- | --- | --- |
| `PROMERAG_OBSERVABILITY_ENABLED` | `false` | Master switch for tracing/metrics |
| `PROMERAG_EXPORTER` | `none` | `none` \| `console` \| `otlp` |
| `PROMERAG_OTEL_SERVICE_NAME` | `prome-rag` | `service.name` resource attribute |
| `PROMERAG_OTLP_ENDPOINT` | *(empty)* | Required when `PROMERAG_EXPORTER=otlp` |
| `PROMERAG_OTLP_PROTOCOL` | `grpc` | `grpc` \| `http` (HTTP paths are auto-appended) |
| `PROMERAG_OTLP_HEADERS` | *(empty)* | `k=v,k2=v2`; used for auth. Never logged |
| `PROMERAG_OTLP_TIMEOUT` | `10.0` | Per-export timeout (s) |
| `PROMERAG_OTLP_INSECURE` | `false` | gRPC only: allow plaintext |
| `PROMERAG_OTLP_BATCH_SIZE` | `512` | Max spans per OTLP batch |
| `PROMERAG_OTLP_SCHEDULE_DELAY_MS` | `5000` | Delay between span flushes |
| `PROMERAG_METRICS_EXPORT_INTERVAL_MS` | `10000` | Metric pull interval |
| `PROMERAG_METRICS_ENABLED` | `true` | Set `false` for traces only |
| `PROMERAG_LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` |
| `PROMERAG_LOG_FORMAT` | `json` | `json` \| `text` |
| `PROMERAG_LOG_FILE` | *(empty)* | Optional log file (stderr is always on) |
| `PROMERAG_TRACE_SAMPLING` | `parentbased_always_on` | `always_on`, `always_off`, `traceidratio`, `parentbased_always_on`, `parentbased_always_off`, `parentbased_traceidratio` |
| `PROMERAG_TRACE_SAMPLING_RATIO` | `1.0` | For ratio-based samplers |
| `PROMERAG_RECORD_PROMPTS` | `false` | Opt-in: record prompt text as span attributes (secrets still redacted) |
| `PROMERAG_RECORD_DOCUMENT_CONTENT` | `false` | Opt-in: record document content attributes |
| `PROMERAG_DIAGNOSTIC_MODE` | `false` | Opt-in diagnostic verbosity; never disables redaction |
| `PROMERAG_STRICT_CARDINALITY` | `false` | Raise instead of drop forbidden metric labels |
| `PROMERAG_MAX_ATTRIBUTE_LENGTH` | `512` | Truncate long span/log attribute strings |

Invalid values are collected and reported together at `init_observability()`
time with a `ConfigurationError`; nothing fails silently at import.

## Span model

Canonical span names (`<stage>.<operation>`), shared request correlation,
nested spans, and `request_id` attached to every span:

| Stage | Spans |
| --- | --- |
| Ingestion | `ingestion.discovery` `ingestion.upload` `ingestion.validate` `ingestion.load` `ingestion.parse` `ingestion.chunk` `ingestion.embed` `ingestion.index_dense` `ingestion.index_lexical` |
| Retrieval | `retrieval.validate` `retrieval.enhance` `retrieval.dense` `retrieval.bm25` `retrieval.rrf_fusion` `retrieval.parent_search` `retrieval.child_search` `retrieval.context_expand` `retrieval.hybrid`* `retrieval.hierarchical`* |
| Reranking | `rerank.prepare` `rerank.inference` `rerank.reorder` |
| Generation | `generation.provider_select` `generation.model_select` `generation.prompt` `generation.rate_limit_wait` `generation.request` `generation.fallback` `generation.output` |
| Guardrails | `guardrails.prompt_guard` `guardrails.source_trust` `guardrails.access_control` `guardrails.privacy_detect` `guardrails.safeguard` `guardrails.citation_validate` `guardrails.enforcement` |
| Evaluation | `evaluation.dataset_load` `evaluation.ragas` `evaluation.judge_call` `evaluation.deterministic` `evaluation.report` |
| Orchestration | `orchestration.pipeline` |

`*` documented extensions (no canonical name in the core catalog).

Attributes on spans are bounded and non-sensitive (`component`, `operation`,
`provider`, `model`, `outcome`, `error.category`, `error.type`,
`retrieval.strategy`, `request_id`, `usage.*`). Attribute keys that are
forbidden for metrics are also stripped from spans, except `prompt` /
`document_content`, which are additionally gated by
`PROMERAG_RECORD_PROMPTS` / `PROMERAG_RECORD_DOCUMENT_CONTENT`.

## Metrics catalog

All metric names are prefixed `prome_rag.`. Units are UCUM-ish (`s`, `By`,
`{token}`, `{request}` …). Distribution histograms are in seconds.

- **Ingestion**: `ingestion.documents_discovered` `ingestion.documents_loaded`
  `ingestion.documents_rejected` `ingestion.documents_parsed`
  `ingestion.parse_failures` `ingestion.chunks_created`
  `ingestion.embedding_requests` `ingestion.embedding_failures`
  `ingestion.dense_index_upserts` `ingestion.lexical_index_upserts`
  `ingestion.index_consistency_failures` `ingestion.duration`
- **Retrieval**: `retrieval.queries` `retrieval.latency`
  `retrieval.dense_candidates` `retrieval.bm25_candidates`
  `retrieval.fused_candidates` `retrieval.hierarchical_parent_candidates`
  `retrieval.hierarchical_child_candidates` `retrieval.empty_results`
  `retrieval.failures`
- **Reranking**: `rerank.candidates_reranked` `rerank.duration` `rerank.failures`
- **Generation**: `generation.requests` `generation.input_tokens`
  `generation.output_tokens` `generation.total_tokens` `generation.duration`
  `generation.first_token_latency` `generation.retries` `generation.fallbacks`
  `generation.rate_limit_wait` `generation.errors`
- **Guardrails**: `guardrails.prompt_guard_requests`
  `guardrails.benign_classifications` `guardrails.malicious_classifications`
  `guardrails.safeguard_invocations` `guardrails.decisions_allow`
  `guardrails.decisions_warn` `guardrails.blocks` `guardrails.reviews`
  `guardrails.privacy_detections` `guardrails.citation_failures`
- **Evaluation**: `evaluation.experiment_runs` `evaluation.samples_evaluated`
  `evaluation.failed_samples` `evaluation.ragas_metric_duration`
  `evaluation.judge_tokens`
- **Runtime**: `runtime.process_memory_bytes` `runtime.cpu_percent`
  `runtime.model_load_duration` `runtime.active_requests`
  `runtime.queue_length`

Hardware-specific metrics (`process_memory_bytes`, `cpu_percent`) are
collected best-effort via `psutil` when available and are skipped silently
otherwise.

### Token accounting

`record_generation_usage()` accepts provider usage reports as-is and never
estimates:

- OpenAI/Groq style: `{"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}`
- Ollama style: flat `{"prompt_eval_count": …, "eval_count": …}` or a
  `Usage` attribute dict
- `GenerationResponse`-style objects: top-level `input_tokens` /
  `output_tokens` / `total_tokens` attributes are read automatically by
  `instrument_generation`

`total_tokens` is recorded only when the provider reports it, or
synthesized as `input + output` when both are present.

## Structured logging

Python `logging` is used throughout (no custom framework). The `prome_rag`
namespace logger is configured by `init_observability()`; the root logger is
never modified. `json` emits one JSON object per line; `text` emits
readable `key=value` pairs. Every record carries timestamp, level, logger,
message, plus any of `component`, `operation`, `request_id`, `trace_id`,
`span_id`, `provider`, `model`, `duration_ms`, `outcome`, `error_category`,
`retrieval_strategy`, `stage`, and arbitrary extras (redacted,
length-truncated).

Redaction (`observability.logging.redaction`) covers: PEM private keys,
JWTs, `Bearer`/`Basic` authorization headers, AWS access key ids/secrets,
GitHub tokens, Slack tokens, Google API keys, `sk-…` provider keys,
generic `key=value` secrets (`api_key`, `password`, `secret`, `token`,
`credential`, `private_key`, …), and email addresses. Sensitive *keys*
(`password`, `token`, `api_key`, `authorization`, …) in mappings have
their values replaced wholesale.

`PROMERAG_DIAGNOSTIC_MODE=true` logs a one-time warning at init; it does
not relax redaction. `PROMERAG_RECORD_PROMPTS` /
`PROMERAG_RECORD_DOCUMENT_CONTENT` are the only content opt-ins, and they
still pass through the redactor.

## Error tracking

`SpanScope.record_exception` (called automatically by `operation_span` and
`traced`) marks the span as errored with a bounded `error.category`
(`timeout`, `rate_limit`, `authentication`, `authorization`, `validation`,
`dependency`, `network`, `storage`, `parse`, `model`, `guardrail_block`,
`not_found`, `unknown`), an `exception` event with the redacted message,
and always re-raises. `error.type` keeps the exception class name.

## Instrumentation helpers

Two styles, both no-op-safe:

```python
from observability.instrumentation import operation_span, traced

with operation_span(component="retrieval", operation="dense") as span:
    ...

@traced(component="parser", operation="parse")
def parse(doc): ...
```

Thin, package-agnostic wrappers for the plugin seams (no imports of the
wrapped packages; duck-typed; proxy objects; originals never mutated):

```python
from observability.instrumentation import instrument_parser, instrument_generation

parser = instrument_parser(parser)          # ingestion.parse span + metrics
client = instrument_generation(client)      # generation.request span + usage metrics
```

| Helper | Wrapped methods |
| --- | --- |
| `instrument_parser` | `parse` → `ingestion.parse` |
| `instrument_chunker` | `chunk` → `ingestion.chunk` |
| `instrument_embedding` | `embed_documents/embed_query/embed_queries` → `ingestion.embed` |
| `instrument_vector_store` | `upsert` / `search` → index upsert + search spans (`index="lexical"` switches to lexical) |
| `instrument_indexing` | `index/index_document/index_chunks/upsert/run` → consistency-failure metric |
| `instrument_dense_retriever` | `retrieve` → `retrieval.dense` |
| `instrument_bm25_retriever` | `retrieve` → `retrieval.bm25` |
| `instrument_hybrid_retriever` | `retrieve` → `retrieval.hybrid` |
| `instrument_fusion` | `fuse` → `retrieval.rrf_fusion` |
| `instrument_hierarchical_retriever` | `retrieve` → `retrieval.hierarchical` |
| `instrument_parent_stage` | `search/search_text` → `retrieval.parent_search` |
| `instrument_child_stage` | `search/search_text` → `retrieval.child_search` |
| `instrument_context_expander` | `expand/expand_context` → `retrieval.context_expand` |
| `instrument_reranker` | `rerank` (+ `_make_pairs` → `rerank.prepare`) → `rerank.inference` |
| `instrument_generation` | `generate/chat/complete/generate_response/generate_text/route/__call__` → `generation.request` + usage |
| `instrument_guardrails` | `evaluate/check/classify/scan/safeguard/prompt_guard/…` → guardrails spans + decision metrics |
| `instrument_evaluation` | `evaluate/run/run_experiment/judge` → evaluation spans + judge-token metric |
| `instrument_loader` | `load/load_file/load_document/ingest` → `ingestion.load` |
| `instrument_orchestration` | `run/run_pipeline/answer/ask/query/invoke/__call__` → `orchestration.pipeline` + active-request level |

When a package lives on a separate feature branch (generation, guardrails,
evaluation, loaders), the generic helpers instrument it without importing
it — pass whatever object you have. Custom components:

```python
from observability.instrumentation.integrations import instrument_component
from observability.tracing import SpanNames

proxy = instrument_component(obj, component="widget", wrappers={
    "do": lambda fn: traced(component="widget", operation="do")(fn),
})
```

The integration seam's contract: wrappers never change signatures, return
values, or exception behavior; wrapped methods keep working with
observability fully disabled.

## Exporters

- **No-op** (`PROMERAG_EXPORTER=none`, default) — everything recorded to
  nowhere.
- **Console** (`PROMERAG_EXPORTER=console`) — spans print at end (simple
  processor); metrics print every `PROMERAG_METRICS_EXPORT_INTERVAL_MS`;
  logs honor `PROMERAG_LOG_FORMAT`. Nothing is persisted or served.
- **OTLP** (`PROMERAG_EXPORTER=otlp`) — spans batch
  (`PROMERAG_OTLP_BATCH_SIZE`, `PROMERAG_OTLP_SCHEDULE_DELAY_MS`) and metrics
  poll to the configured collector over gRPC (4317) or HTTP (4318; `/v1/traces`
  and `/v1/metrics` appended as needed). Header-based auth is passed through;
  header values are never logged, spanned, or measured.

Example collector (user runs it; this package never starts one):

```bash
docker run -p 4317:4317 -p 4318:4318 otel/opentelemetry-collector
PROMERAG_EXPORTER=otlp PROMERAG_OTLP_ENDPOINT=http://localhost:4317 \
PROMERAG_OBSERVABILITY_ENABLED=true python app.py
```

## Using with OpenTelemetry directly

The stack builds standalone `TracerProvider` / `MeterProvider` objects held
by the runtime; the global OTel proxy provider is intentionally *not*
overridden, so importing this package cannot leak telemetry into unrelated
libraries. Wire other providers into the same setup via:

```python
from observability import init_observability, ObservabilityConfig

rt = init_observability(
    ObservabilityConfig(enabled=True),
    tracer_provider=my_provider,   # any opentelemetry TracerProvider
    meter_provider=my_meter,       # any opentelemetry MeterProvider
)
```

## Architecture

```
observability/
  contracts/      models (TraceContext, SpanRecord, …), protocols, exceptions
  tracing/        Tracer facade, SpanScope, contextvars, W3C propagation
  metrics/        registry (OTel | fallback | no-op), counters/gauges/histograms, domain catalog
  logging/        JSON/text formatters, redaction, correlation filter
  exporters/      console and OTLP provider builders
  instrumentation/ decorators, context managers, duck-typed plugin wrappers
  configuration.py  env parsing/validation, init/shutdown lifecycle
```

Dependency direction: `configuration` ← everything; `contracts` depends on
nothing; nothing imports plugin packages; `logging.redaction` /
`metrics._labels` / `tracing.spans` import each other only lazily.

## Tests

```bash
python -m pytest tests/observability -q
```

In-memory `InMemorySpanExporter` / `InMemoryMetricReader` from the SDK are
used for assertions; no collector or network is ever required.

## Limitations and non-goals

- No dashboards, servers, collectors, or Grafana/Jaeger/Tempo wiring.
- No automatic background services; exporters are attached only through
  `init_observability()` and are flushed/detached by `shutdown_observability()`.
- Hardware metrics require `psutil`; otherwise silently absent.
- Fallback (non-OTel) recording is a thin safety net, not a backend: values
  are inspectable via `MetricRegistry.snapshot()` but are not exported.
- The OpenTelemetry Logs SDK is intentionally not required; structured
  logging is plain Python `logging`.
- Spans and metrics are thread-bound through contextvars; fan-out across
  processes requires propagation via `inject_trace_context` /
  `extract_trace_context` (HTTP, queues).
- Diagnostic content opt-ins change what is recorded, never how it is
  redacted.
