"""Prome RAG Plugins — Integrations package.

Six end-to-end RAG pipeline adapters connecting the core ecosystems:

- loader_parser_adapter  : LoadedResource  →  ParsedDocument
- retrieval_reranking_adapter: RetrievedItem  →  RerankCandidate
- reranking_generation_adapter: RerankedItem  →  GenerationRequest.context_items
- evaluation_adapter     : Pipeline outputs → EvalSample / RetrievalResult / ExperimentConfig
- sync_deletion_coordinator: Vector/lexical index sync and deletion
- observability_integration: Request/trace context propagation and nested spanning
"""

from __future__ import annotations

from integrations.loader_parser_adapter import (
    parse_loaded_resource,
    resource_to_parser_input,
)
from integrations.retrieval_reranking_adapter import (
    retrieved_item_to_rerank_candidate,
    reranked_items_to_context_items,
)
from integrations.reranking_generation_adapter import reranked_to_generation_context
from integrations.evaluation_adapter import (
    eval_sample_from_pipeline,
    experiment_config_from_pipeline,
    pipeline_outputs_to_eval_sample,
)
from integrations.sync_deletion_coordinator import (
    IndexSyncCoordinator,
    effective_doc_id_from_detection,
)
from integrations.observability_integration import (
    ensure_observability,
    shutdown_observability,
    set_request_context,
    get_request_context,
    span,
    PipelineTracer,
    trace_pipeline_stage,
    trace_request_id,
    trace_trace_id,
)

__all__ = [
    "loader_parser_adapter",
    "retrieval_reranking_adapter",
    "reranking_generation_adapter",
    "evaluation_adapter",
    "sync_deletion_coordinator",
    "observability_integration",
    "parse_loaded_resource",
    "resource_to_parser_input",
    "retrieved_item_to_rerank_candidate",
    "reranked_items_to_context_items",
    "reranked_to_generation_context",
    "eval_sample_from_pipeline",
    "experiment_config_from_pipeline",
    "pipeline_outputs_to_eval_sample",
    "IndexSyncCoordinator",
    "effective_doc_id_from_detection",
    "ensure_observability",
    "shutdown_observability",
    "set_request_context",
    "get_request_context",
    "span",
    "PipelineTracer",
    "trace_pipeline_stage",
    "trace_request_id",
    "trace_trace_id",
]