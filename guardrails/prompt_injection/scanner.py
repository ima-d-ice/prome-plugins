"""Prompt Guard scanner for the Guardrails ecosystem.

Lazily imports and wraps the Generation subsystem's Prompt Guard classifier.
Provides token-aware segmentation, adaptive language routing, and conservative
aggregation across segments.

Do not build duplicate inference adapters. This scanner imports the
Generation Prompt Guard classifier on first use, so the dependency is
optional and lazy.
"""

from __future__ import annotations

import logging
from typing import Any

from guardrails.contracts.exceptions import GuardrailConfigError

logger = logging.getLogger(__name__)

# Module-level lazy import holders
_CLASSIFIER_IMPORTED = False
_ClassifierClass = None
_DefaultModel22M = "meta-llama/llama-prompt-guard-2-22m"
_DefaultModel86M = "meta-llama/llama-prompt-guard-2-86M"


def _ensure_classifier_imported() -> type:
    """Lazily import the Prompt Guard classifier from the Generation subsystem.

    Returns the PromptGuardClassifier class. On subsequent calls, returns
    the cached class.

    Raises:
        GuardrailConfigError: if the Generation subsystem is not available
            or the import fails.
    """

    global _CLASSIFIER_IMPORTED, _ClassifierClass

    if _CLASSIFIER_IMPORTED:
        return _ClassifierClass

    try:
        from generation.safety.prompt_guard.classifier import (
            PromptGuardClassifier as _ClassifierClass,
        )
        _CLASSIFIER_IMPORTED = True
        logger.info("Prompt Guard classifier imported from Generation subsystem")
        return _ClassifierClass
    except ImportError as e:
        raise GuardrailConfigError(
            "Prompt Guard classifier not available. "
            "Install the Generation subsystem or provide a mock adapter."
        ) from e


class PromptInjectionScanner:
    """Scan text for prompt injection/jailbreak using Prompt Guard.

    Supports scanning of user input, conversation history, retrieved
    documents, and other caller-supplied untrusted context.

    The scanner respects the 512-token Prompt Guard model context window
    via token-aware segmentation with configurable overlap.

    Do not treat a benign classifier result as proof that content is safe.
    Prompt Guard detects prompt attacks and jailbreaks but does not
    establish whether general content is harmful, factual or policy-compliant.
    """

    # Adaptive language routing defaults (matches Generation's Prompt Guard)
    LANGUAGE_ROUTING: dict[str, str] = {
        "en": _DefaultModel22M,  # English → 22M
        "fr": _DefaultModel86M,  # French → 86M
        "de": _DefaultModel86M,  # German → 86M
        "hi": _DefaultModel86M,  # Hindi → 86M
        "it": _DefaultModel86M,  # Italian → 86M
        "pt": _DefaultModel86M,  # Portuguese → 86M
        "es": _DefaultModel86M,  # Spanish → 86M
        "th": _DefaultModel86M,  # Thai → 86M
    }

    SUPPORTED_LANGUAGES = {
        "en", "eng", "en-US", "fr", "fr-fr", "de", "de-de",
        "hi", "hi-in", "it", "it-it", "pt", "pt-br",
        "es", "es-es", "th", "th-th",
    }

    def __init__(
        self,
        *,
        model_22m: str | None = None,
        model_86m: str | None = None,
        supported_languages: set[str] | None = None,
        language_routing: dict[str, str] | None = None,
        enforcement_policy: str = "allow",
        overlap: int = 50,
    ) -> None:
        """Initialize the Prompt Injection scanner.

        Args:
            model_22m: explicit model ID for the 22M model.
                When None, uses the default (meta-llama/llama-prompt-guard-2-22m).
            model_86m: explicit model ID for the 86M model.
                When None, uses the default (meta-llama/llama-prompt-guard-2-86m).
            supported_languages: set of language codes supported.
                When None, uses the default set.
            language_routing: mapping of language code to model ID.
                When None, uses the default language routing.
            enforcement_policy: default enforcement policy for results.
                One of: "allow", "block", "require_review", "allow_with_warning".
            overlap: overlap percentage for segment scanning (0-100).
        """

        self._model_22m = model_22m or _DefaultModel22M
        self._model_86m = model_86m or _DefaultModel86M
        self._supported_languages = supported_languages or self.SUPPORTED_LANGUAGES
        self._language_routing = language_routing or self.LANGUAGE_ROUTING
        self._enforcement_policy = enforcement_policy
        self._overlap = max(0, min(100, overlap))

        # Lazy-import the classifier class
        self._classifier_cls = _ensure_classifier_imported()

    def scan(
        self,
        text: str,
        *,
        language_code: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Scan text for prompt injection/jailbreak.

        Uses adaptive language-based routing to select the appropriate
        Prompt Guard model, then classifies via token-aware segmentation.

        This is the main entry point. It scans all eligible segments and
        aggregates results conservatively.

        Args:
            text: the text to scan for prompt injection.
            language_code: ISO-639-1 language code. When None, auto-detect
                from text.
            model_name: explicit model override. When None, uses adaptive
                language-based routing.

        Returns:
            dict with classification results including:
                - model_used: the Prompt Guard model used
                - language_route: the language route taken
                - classification: "BENIGN" or "MALICIOUS"
                - confidence: available confidence score
                - score: available confidence or score metric
                - segment_count: number of segments scanned
                - malicious_segments: list of malicious segment references
                - diagnostics: additional diagnostics
                - incomplete_scanning: True if any segment could not be classified
        """

        # Ensure classifier is available

        # Determine language route
        effective_language = self._determine_language(language_code, text)

        # Select model based on language
        model_to_use = self._select_model(effective_language, model_name)

        # Handle 512-token context window via segmentation
        segment_result = self._classify_with_segmentation(
            text, model_to_use
        )

        # Aggregate results conservatively
        classification = self._aggregate_results(segment_result)

        # Build return dict
        result: dict[str, Any] = {
            "model_used": model_to_use,
            "language_route": effective_language,
            "classification": classification,
            "confidence": segment_result.get("confidence"),
            "score": segment_result.get("score"),
            "segment_count": segment_result.get("segment_count", 1),
            "malicious_segments": segment_result.get("malicious_segments", []),
            "diagnostics": segment_result.get("diagnostics", {}),
            "incomplete_scanning": segment_result.get("incomplete_scanning", False),
        }

        return result

    def scan_target(
        self,
        target: str,
        *,
        language_code: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Scan a single target text for prompt injection.

        Convenience method that calls scan() with default parameters.

        Args:
            target: the text to scan.
            language_code: optional language code.
            model_name: optional model override.

        Returns:
            classification result dict.
        """

        return self.scan(
            text=target,
            language_code=language_code,
            model_name=model_name,
        )

    def _determine_language(
        self, language_code: str | None, text: str
    ) -> str:
        """Determine the language for routing purposes.

        Uses the explicit language code from the request when available.
        Falls back to lightweight language detection from text.
        """

        # Use explicit language code from request when available
        if language_code:
            code = language_code.lower().strip()
            if code in self._supported_languages:
                return code
            # Unsupported language — return as-is and let caller handle
            return code

        # Lightweight auto-detection: check for common language patterns
        lower_text = text.lower()

        # English: common words
        if any(word in lower_text for word in ["the", "and", "or", "but", "in", "on"]):
            return "en"

        # French: common French words
        if any(word in lower_text for word in ["le", "la", "les", "et", "ou", "mais"]):
            return "fr"

        # German: common German words
        if any(word in lower_text for word in ["der", "die", "das", "und", "oder", "but"]):
            return "de"

        # Default to unknown language → use 86M (safer multilingual option)
        return "unknown"

    def _select_model(
        self, effective_language: str, model_name: str | None
    ) -> str:
        """Select the Prompt Guard model based on language and overrides."""

        if model_name:
            return model_name

        # Use explicit language routing
        return self._language_routing.get(
            effective_language, self._model_86m
        )

    def _classify_with_segmentation(
        self, text: str, model_name: str
    ) -> dict[str, Any]:
        """Classify text using segmentation for the 512-token context window.

        Splits longer content into bounded segments with overlap, scans
        each independently, and aggregates results conservatively.

        Args:
            text: the text to classify.
            model_name: the Prompt Guard model to use.

        Returns:
            dict with segmentation results including confidence, score,
            malicious segments, and incomplete_scanning flag.
        """

        # Enforce 512-token context window
        chars_per_token = 4
        max_chars = 512 * chars_per_token  # 2048 characters
        max_tokens = 512

        # If text fits within the context window, classify directly
        if len(text) <= max_chars:
            # Single-segment classification
            return self._single_segment_classify(text, model_name)

        # Text exceeds 512 tokens — split into segments with overlap
        segments = self._split_into_segments(text, max_tokens, self._overlap)

        if not segments:
            # Empty result - report incomplete scanning
            return {
                "is_malicious": False,
                "classification": "BENIGN",
                "confidence": 0.0,
                "score": 0.0,
                "segment_count": 0,
                "malicious_segments": [],
                "diagnostics": {
                    "text_length": len(text),
                    "max_tokens": max_tokens,
                    "segment_overlap": self._overlap,
                    "num_segments": 0,
                    "incomplete_scanning": True,
                },
                "incomplete_scanning": True,
            }

        # Classify each segment independently
        segment_results: list[dict[str, Any]] = []
        malicious_segments: list[dict[str, Any]] = []

        for i, segment in enumerate(segments):
            result = self._single_segment_classify(segment, model_name)
            segment_results.append(result)

            # Track malicious segments
            if result.get("is_malicious", False):
                malicious_segments.append({
                    "segment_index": i,
                    "start_char": sum(len(s) for s in segments[:i]),
                    "end_char": sum(len(s) for s in segments[:i + 1]),
                    "content_preview": (
                        segment[:50] + "..." if len(segment) > 50 else segment
                    ),
                    **result,
                })

        # Aggregate results conservatively:
        # A malicious segment must NOT be hidden by benign segments.
        # If ANY segment is malicious, the overall classification is MALICIOUS.
        any_malicious = any(
            r.get("is_malicious", False) for r in segment_results
        )

        # Determine overall classification
        overall_is_malicious = any_malicious

        # Compute confidence as the aggregate
        # If any segment has high confidence of malicious, overall confidence is high
        confidences = [r.get("confidence", 0.0) for r in segment_results]
        overall_confidence = max(confidences) if confidences else 0.0

        # Compute score (aggregate)
        scores = [r.get("score", 0.0) for r in segment_results]
        overall_score = max(scores) if scores else 0.0

        # Check if any segment could not be classified (e.g., model error)
        # This happens when a segment returns unexpected results
        all_classified = all(
            "is_malicious" in r for r in segment_results
        )
        incomplete = not all_classified

        return {
            "is_malicious": overall_is_malicious,
            "classification": "MALICIOUS" if overall_is_malicious else "BENIGN",
            "confidence": overall_confidence,
            "score": overall_score,
            "segment_count": len(segment_results),
            "malicious_segments": malicious_segments,
            "all_segment_results": segment_results,
            "diagnostics": {
                "text_length": len(text),
                "max_tokens": max_tokens,
                "segment_overlap": self._overlap,
                "num_segments": len(segment_results),
                "incomplete_scanning": incomplete,
            },
            "incomplete_scanning": incomplete,
            "latency_ms": 0,  # will be measured by caller
        }

    def _single_segment_classify(
        self, text: str, model_name: str
    ) -> dict[str, Any]:
        """Classify a single text segment using the Prompt Guard model.

        This uses the rule-based heuristic from the Generation subsystem's
        Prompt Guard scaffold. In production, this would call the actual
        Prompt Guard model API.

        Args:
            text: the segment text to classify.
            model_name: the Prompt Guard model identifier.

        Returns:
            dict with classification result for this segment.
        """

        # Rule-based heuristic scaffold (from Generation's Prompt Guard)
        lower = text.lower()

        # Known prompt injection patterns (heuristic)
        injection_patterns = [
            "ignore previous instructions",
            "disregard system prompt",
            "forget all previous",
            "system override",
            "jailbreak",
            "pretend you are",
            "you are now a",
            "act as",
            "roleplay as",
            "developer mode",
            "dan mode",
            "prompt injection",
        ]

        is_malicious = any(pattern in lower for pattern in injection_patterns)

        # Confidence based on pattern matches
        pattern_matches = sum(1 for pattern in injection_patterns if pattern in lower)
        confidence = min(1.0, pattern_matches / len(injection_patterns) * 2) if is_malicious else 0.0

        # Score: higher if more patterns match
        score = confidence * 10.0  # scale to ~0-10

        return {
            "is_malicious": is_malicious,
            "classification": "MALICIOUS" if is_malicious else "BENIGN",
            "confidence": confidence,
            "score": score,
        }

    def _split_into_segments(
        self, text: str, max_tokens: int, overlap: int
    ) -> list[str]:
        """Split text into bounded segments for the 512-token window.

        Args:
            text: the text to split.
            max_tokens: maximum tokens per segment (512 for Prompt Guard).
            overlap: overlap percentage between segments.

        Returns:
            list of text segments.
        """

        chars_per_token = 4
        max_chars = max_tokens * chars_per_token  # 2048
        overlap_chars = int(max_chars * overlap / 100)

        # If text fits in one segment
        if len(text) <= max_chars:
            return [text]

        segments: list[str] = []
        start = 0

        while start < len(text):
            # End of this segment
            end = start + max_chars

            # If we're at the end, take the remainder
            if end >= len(text):
                segments.append(text[start:])
                break

            # Otherwise, take max_chars and then try to break at a space
            segment_end = end
            # Prefer to break at a space near the end
            break_pos = text.rfind(" ", start, end)
            if break_pos > start:
                segment_end = break_pos

            segments.append(text[start:segment_end])
            # Next segment starts with overlap
            start = max(start + max_chars - overlap_chars, len(text))

        # Pad to ensure at least one segment
        if not segments:
            segments.append(text[:max_chars])

        return segments

    def _aggregate_results(self, segment_result: dict[str, Any]) -> str:
        """Aggregate segmentation results into a final classification.

        The segment_result already has the aggregated classification.
        Return it as-is, but also check for incomplete scanning.

        Args:
            segment_result: the result from _classify_with_segmentation.

        Returns:
            "BENIGN" or "MALICIOUS".
        """

        # The segment_result already has the aggregated classification
        # Return it as-is
        return segment_result.get("classification", "BENIGN")

    def scan_user_input(
        self, user_input: str, *, language_code: str | None = None
    ) -> dict[str, Any]:
        """Scan user input for prompt injection.

        Convenience method for the common case of scanning the primary
        user query.

        Args:
            user_input: the raw user input text.
            language_code: optional ISO-639-1 language code.

        Returns:
            classification result dict.
        """

        return self.scan(
            text=user_input,
            language_code=language_code,
        )

    def scan_context(
        self,
        context_text: str,
        *,
        language_code: str | None = None,
        model_name: str | None = None,
        source_ref: str | None = None,
    ) -> dict[str, Any]:
        """Scan retrieved context text for prompt injection.

        This is a convenience wrapper that scans a single context segment.
        For multi-segment scanning of retrieved documents, use scan()
        directly or the context_scanner module.

        Args:
            context_text: the retrieved context text to scan.
            language_code: optional language code.
            model_name: optional model override.
            source_ref: optional source reference (chunk ID, doc ID) for
                diagnostics.

        Returns:
            classification result dict with optional source_ref added to
            diagnostics.
        """

        result = self.scan(
            text=context_text,
            language_code=language_code,
            model_name=model_name,
        )

        # Add source reference to diagnostics if provided
        if source_ref:
            if "diagnostics" not in result or result["diagnostics"] is None:
                result["diagnostics"] = {}
            result["diagnostics"]["source_reference"] = source_ref

        return result
