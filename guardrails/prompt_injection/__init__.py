"""Prompt injection detection for the Guardrails ecosystem.

Reuses the existing Generation subsystem's Prompt Guard integration
(meta-llama/llama-prompt-guard-2-22m and meta-llama/llama-prompt-guard-2-86m).

Do not build duplicate inference adapters. The scanner lazily imports
the Generation Prompt Guard classifier when available, and provides
full infrastructure for token-aware segmentation, context scanning,
and conservative aggregation.

Key design:
- Adaptive language routing: English → 22M, multilingual → 86M
- 512-token context window respected via segmentation
- Conservative aggregation: ANY malicious segment → overall MALICIOUS
- All eligible segments are scanned; none are silently skipped
- Report incomplete scanning if any segment could not be classified
- Enforcement: ALLOW, WARN, BLOCK, REVIEW, MARK_SOURCE_UNTRUSTED
"""

from guardrails.prompt_injection.context_scanner import ContextScanner
from guardrails.prompt_injection.decisions import PromptInjectionDecisionMapper
from guardrails.prompt_injection.scanner import PromptInjectionScanner
from guardrails.prompt_injection.segmentation import (
    detect_truncation,
    segment_text_for_prompt_guard,
)

__all__ = [
    "PromptInjectionScanner",
    "ContextScanner",
    "segment_text_for_prompt_guard",
    "detect_truncation",
    "PromptInjectionDecisionMapper",
]
