"""Guardrails and Security ecosystem for Prome RAG Plugins.

This package provides deterministic security validation, enforcement, and audit
across the entire RAG pipeline: user input, query processing, document ingestion,
retrieved context, context assembly, generation, generated output, citation
handling, and sensitive information exposure.

It reuses Generation subsystem's Prompt Guard and Safeguard model adapters
through public interfaces. It does not implement duplicate model loaders.

Key responsibility areas:
- User input validation and prompt-injection detection
- Retrieved document trust and source security
- Privacy-sensitive information detection and redaction
- Output validation and citation verification
- Deterministic policy enforcement with documented precedence
- Audit event recording

The base package is dependency-light: no mandatory Prompt Guard, Safeguard,
or Groq runtime dependency. Optional model integration is available when
the Generation subsystem is present.
"""

from guardrails.audit.events import AuditEvent
from guardrails.contracts.models import (
    GuardrailCapabilities,
    GuardrailDecision,
    GuardrailRequest,
    GuardrailStage,
)
from guardrails.enforcement.engine import EnforcementEngine
from guardrails.policies.engine import PolicyEngine

__all__ = [
    "GuardrailStage",
    "GuardrailDecision",
    "GuardrailRequest",
    "GuardrailCapabilities",
    "PolicyEngine",
    "EnforcementEngine",
    "AuditEvent",
]
