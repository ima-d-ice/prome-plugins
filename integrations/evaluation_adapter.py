"""Evaluation integration adapter.

Maps pipeline outputs (retrieval results, generation responses) into the
existing ``evaluation`` contract models (`EvalSample`, ``RetrievalResult``,
``ExperimentConfig``) so that the project's evaluation runners can consume them.

This adapter does NOT depend on the ``ragas`` package being installed.  It
produces contract-compatible data structures that *can* be passed to ragas
evaluators when ragas is available, or used for custom metric computation
when it is not.

Key mappings:
- ``retrieved_contexts``  <- chunk IDs from retrieval results
- ``retrieved_context_ids`` <- same, as list of string IDs
- ``generated_response``  <- generation text
- ``cited_source_ids``    <- source/chunk IDs from generation citations
- ``reference_contexts``  <- ground-truth contexts from evaluation dataset
- ``reference``           <- reference answer string
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, List, Optional, Tuple

# Load evaluation contract models via exec to avoid the ragas dependency
# that lives in evaluation/contracts/__init__.py.  The model definitions
# in models.py are dependency-free.
_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "contracts", "models.py")
with open(_MODELS_PATH) as _f:
    _models_code = _f.read()
# Execute in an isolated namespace so package-level ragas imports don't leak
_models_namespace: Dict[str, Any] = {}
exec(compile(_models_code, _MODELS_PATH, "exec"), _models_namespace)

EvalSample = _models_namespace["EvalSample"]
RetrievalResult = _models_namespace["RetrievalResult"]
ExperimentConfig = _models_namespace["ExperimentConfig"]
MetricProfile = _models_namespace["MetricProfile"]
RerankCandidate = _models_namespace["RerankCandidate"]


def _ensure_list(value: Any) -> List[Any]:
    """Ensure *value* is a list, wrapping scalars or None as needed."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def eval_sample_from_pipeline(
    *,
    user_input: str,
    retrieved_chunks: Optional[List[Any]] = None,
    generation_response: Optional["GenerationResponse"] = None,
    cited_source_ids: Optional[List[str]] = None,
    reference: Optional[str] = None,
    reference_contexts: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    experiment_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> EvalSample:
    """Build an ``EvalSample`` from pipeline outputs.

    This is the primary entry point for connecting a completed RAG pipeline
    to the evaluation framework.  It maps:

    - ``retrieved_chunks`` (list of backend hits or ``RetrievalResult``) ->
      ``retrieved_contexts`` (list of text strings) and
      ``retrieved_context_ids`` (list of chunk/doc IDs).
    - ``generation_response`` -> ``generated_response`` (generated text) and
      ``cited_source_ids`` (source/chunk IDs from ``citation_source_refs``).
    - ``reference`` / ``reference_contexts`` -> ground-truth from the evaluation
      dataset, if available.

    The ``strategy`` field records which retrieval strategy was used (e.g.
    ``"dense"``, ``"bm25"``, ``"hybrid_rrf"``).

    Returns:
        An ``EvalSample`` ready for use with evaluation runners or ragas metrics.
    """
    # Build retrieved_contexts and retrieved_context_ids from chunks
    retrieved_contexts: List[str] = []
    retrieved_context_ids: List[str] = []

    if retrieved_chunks:
        for chunk in retrieved_chunks:
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else str(chunk.get("chunk_id", "")) if isinstance(chunk, dict) else ""
            doc_id = chunk.doc_id if hasattr(chunk, "doc_id") else str(chunk.get("doc_id", "")) if isinstance(chunk, dict) else ""

            retrieved_contexts.append(chunk_text)
            # Format: "doc_id:chunk_id" when both available, otherwise whichever is present
            if doc_id and chunk_id:
                retrieved_context_ids.append(f"{doc_id}:{chunk_id}")
            elif doc_id:
                retrieved_context_ids.append(doc_id)
            elif chunk_id:
                retrieved_context_ids.append(chunk_id)
            else:
                retrieved_context_ids.append("")

    # Build cited_source_ids from generation response citations
    generated_text: Optional[str] = None
    cited_ids: List[str] = _ensure_list(cited_source_ids)

    if generation_response is not None:
        generated_text = generation_response.text

        # Extract cited_source_ids from citation_source_refs if not provided
        if not cited_ids and generation_response.citation_source_refs:
            for ref in generation_response.citation_source_refs:
                if isinstance(ref, dict):
                    # Try chunk_id or doc_id
                    cid = ref.get("chunk_id") or ref.get("doc_id") or ""
                    if cid:
                        cited_ids.append(str(cid))
                elif isinstance(ref, str):
                    cited_ids.append(ref)

    # Deduplicate cited_source_ids while preserving order
    seen = set()
    unique_cited_ids: List[str] = []
    for cid in cited_ids:
        if cid and cid not in seen:
            seen.add(cid)
            unique_cited_ids.append(cid)

    return EvalSample(
        sample_id=experiment_id or "",
        user_input=user_input,
        reference=reference,
        reference_contexts=_ensure_list(reference_contexts),
        retrieved_contexts=retrieved_contexts,
        retrieved_context_ids=retrieved_context_ids,
        reranked_contexts=None,  # set by reranking-aware experiments
        generated_response=generated_text,
        cited_source_ids=unique_cited_ids,
        metadata={},
        dataset_version=dataset_id,
        strategy=strategy,
    )


def experiment_config_from_pipeline(
    *,
    experiment_id: str,
    dataset_id: str,
    retrieval_strategy: str = "dense",
    reranker_on: bool = False,
    generation_model: Optional[str] = None,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_temperature: float = 0.01,
    judge_max_tokens: Optional[int] = None,
    embedding_model: Optional[str] = None,
    embedding_revision: Optional[str] = None,
    metric_names: Optional[List[str]] = None,
    ragas_version: Optional[str] = None,
) -> ExperimentConfig:
    """Build an ``ExperimentConfig`` from pipeline configuration.

    This records the experiment metadata needed for report generation and
    metric selection.  All fields have sensible defaults so callers only need
    to supply the values that differ from the norm.
    """
    from evaluation.contracts.models import RetrievalStrategy  # local import; safe because contracts protocol is dependency-free

    strategy_enum = RetrievalStrategy(retrieval_strategy) if retrieval_strategy else RetrievalStrategy.DENSE

    return ExperimentConfig(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        retrieval_strategy=strategy_enum,
        reranker_on=reranker_on,
        generation_model=generation_model,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_temperature=judge_temperature,
        judge_max_tokens=judge_max_tokens,
        embedding_model=embedding_model,
        embedding_revision=embedding_revision,
        ragas_version=ragas_version,
        metric_names=_ensure_list(metric_names),
        prompt_metric_versions={},
        timestamp=None,  # will be set by the runner
    )


def pipeline_outputs_to_eval_sample(
    user_input: str,
    retrieval_results: Optional[List[Any]] = None,
    generation_response: Optional["GenerationResponse"] = None,
    reference: Optional[str] = None,
    reference_contexts: Optional[List[str]] = None,
    strategy: str = "dense",
    experiment_id: str = "default-experiment",
    dataset_id: str = "default-dataset",
) -> EvalSample:
    """High-level convenience: convert full pipeline outputs to ``EvalSample``.

    Combines retrieval results and generation response into a single evaluation
    sample.  If only one of retrieval or generation is provided, the other
    fields are set to ``None`` / empty where appropriate.
    """
    # Build retrieved_chunks from results
    retrieved_chunks: List[Any] = _ensure_list(retrieval_results)

    return eval_sample_from_pipeline(
        user_input=user_input,
        retrieved_chunks=retrieved_chunks,
        generation_response=generation_response,
        cited_source_ids=None,  # will be extracted from generation_response
        reference=reference,
        reference_contexts=reference_contexts,
        strategy=strategy,
        experiment_id=experiment_id,
        dataset_id=dataset_id,
    )