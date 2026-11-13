"""Shared data models for the Guardrails ecosystem.

These models are dependency-free: they contain no imports of Ollama, Groq,
Prompt Guard, Safeguard, Transformers, or PyTorch. They describe the guardrail
stages, decisions, requests, and capabilities that flow through the security
pipeline.

Follow the existing RAG contract patterns: dataclasses with frozen=True where
immutability is desired, __post_init__ validation, and minimal fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GuardrailStage:
    """Stage in the RAG pipeline where guardrails apply.

    Stages cover the full RAG lifecycle from user input through generated
    output and citation handling.

    Values:
        INPUT: User input entry point.
        QUERY: Processed query after enhancement/preparation.
        INGESTION: Document ingestion pipeline.
        RETRIEVED_CONTEXT: Retrieved evidence before context assembly.
        CONTEXT_ASSEMBLY: Context construction from retrieved chunks.
        GENERATION_INPUT: Input to the generation model.
        GENERATION_OUTPUT: Generated text output.
        CITATION_VALIDATION: Citation/source reference validation.
    """

    value: str

    # Pre-defined stage constants
    INPUT = "input"
    QUERY = "query"
    INGESTION = "ingestion"
    RETRIEVED_CONTEXT = "retrieved_context"
    CONTEXT_ASSEMBLY = "context_assembly"
    GENERATION_INPUT = "generation_input"
    GENERATION_OUTPUT = "generation_output"
    CITATION_VALIDATION = "citation_validation"


@dataclass(frozen=True)
class GuardrailDecision:
    """Top-level outcome of a guardrail check.

    Every decision contains enough information to be auditable and observable.

    Attributes:
        decision: One of ALLOW, WARN, BLOCK, REVIEW.
        stage: the pipeline stage this decision applies to.
        reason_codes: machine-readable codes explaining the decision.
        policy_ids: identifiers of policies that influenced this decision.
        policy_versions: versions of the influencing policies.
        source_references: references to sources involved (chunk IDs, doc IDs).
        detector_model: the model or classifier used, if applicable.
        timestamp: when the decision was made.
        diagnostics: additional structured diagnostic information.
    """

    decision: str
    stage: GuardrailStage
    reason_codes: list[str] | None = None
    policy_ids: list[str] | None = None
    policy_versions: list[str] | None = None
    source_references: list[str] | None = None
    detector_model: str | None = None
    timestamp: float | None = None
    diagnostics: dict[str, Any] | None = None

    # Decision outcome constants
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REVIEW = "review"


ALLOW_VALUES = (GuardrailDecision.ALLOW, GuardrailDecision.WARN, GuardrailDecision.BLOCK, GuardrailDecision.REVIEW)


@dataclass(frozen=True)
class GuardrailRequest:
    """Request structure supplied to the guardrail protocol.

    The caller supplies a request and receives a structured decision.

    Attributes:
        request_id: unique identifier for this request.
        pipeline_stage: the current guardrail stage being evaluated.
        user_input: the raw user input text.
        context_items: retrieved context items (with metadata).
        generated_output: optional generated text output.
        source_metadata: metadata about sources/documents.
        policy_config: optional policy configuration override.
        language_code: optional ISO-639-1 language code.
    """

    request_id: str
    pipeline_stage: GuardrailStage
    user_input: str
    context_items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    generated_output: str | None = None
    source_metadata: dict[str, Any] | None = None
    policy_config: dict[str, Any] | None = None
    language_code: str | None = None


@dataclass(frozen=True)
class GuardrailCapabilities:
    """Exposed security capabilities of this guardrails implementation.

    Covers the concrete protections supported. Callers can query which
    capabilities are available without assuming all are active by default.

    Attributes:
        input_validation: deterministic input validation is supported.
        prompt_injection_detection: Prompt Guard integration is available.
        context_scanning: retrieved context can be scanned for injections.
        source_trust: source trust management is supported.
        access_control: access-control checks are supported.
        privacy_detection: sensitive information detection is supported.
        redaction: content redaction is supported.
        output_validation: output validation is supported.
        citation_validation: citation identity validation is supported.
        policy_based_classification: policy-based classification is supported.
    """

    input_validation: bool = True
    prompt_injection_detection: bool = True
    context_scanning: bool = True
    source_trust: bool = True
    access_control: bool = True
    privacy_detection: bool = True
    redaction: bool = True
    output_validation: bool = True
    citation_validation: bool = True
    policy_based_classification: bool = True