"""Shared contracts for the Guardrails ecosystem.

Defines the core data models and protocol that all guardrails components
depend on. These contracts are dependency-free: no imports of Ollama, Groq,
Prompt Guard, Safeguard, Transformers, or PyTorch.

The following entities are defined:
- GuardrailStage: pipeline stages covered by guardrails
- GuardrailDecision: top-level decision outcomes with full metadata
- GuardrailRequest: request-carrying structure for guardrail evaluation
- GuardrailCapabilities: exposed security capabilities
- Guardrail protocol: small independent protocol for request/decision exchange
"""

from guardrails.contracts.models import (
    GuardrailStage,
    GuardrailDecision,
    GuardrailRequest,
    GuardrailCapabilities,
)

__all__ = [
    "GuardrailStage",
    "GuardrailDecision",
    "GuardrailRequest",
    "GuardrailCapabilities",
]