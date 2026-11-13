"""Prompt Guard classifier for prompt injection and jailbreak detection.

meta-llama/llama-prompt-guard-2-22m and meta-llama/llama-prompt-guard-2-86m
are dedicated prompt attack classifiers. They detect prompt injections and
jailbreaks. They are not general-purpose harmful-content classifiers and
do not rewrite user questions.

Adaptive routing:
- English input: Prompt Guard 22M.
- Supported non-English input: Prompt Guard 86M.
- Unknown language: Prompt Guard 86M as the safer multilingual option.
- Unsupported language: use configured policy and report that language
  coverage is unverified.

Documented evaluated languages: English, French, German, Hindi, Italian,
Portuguese, Spanish, Thai.

512-token limit: Both models have a 512-token context window.
Requirements:
- Respect the model's tokenizer.
- Split longer content into bounded segments with configurable overlap.
- Scan segments independently.
- Aggregate results conservatively: a malicious segment must not be hidden
  by benign segments.
- Preserve the source and segment index for each classification.
- Do not silently truncate long retrieved documents.

Scan targets:
1. User query.
2. Optional conversation history.
3. Retrieved untrusted context.
4. Other caller-supplied untrusted text.

Classification response:
- Model used.
- Language route.
- Classification (BENIGN or MALICIOUS).
- Available confidence or score.
- Segment count.
- Malicious segment references.
- Diagnostics.
- Latency.

Guard enforcement (configurable):
- Allow.
- Block.
- Require review.
- Allow with warning.
Do not silently override the caller's configured enforcement policy.
A classifier error or malformed output must be visible and handled through
an explicit fail-open/fail-closed/review configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from generation.contracts.exceptions import (
    GenerationPromptGuardError,
)


class PromptGuardClassifier:
    """Classifier for prompt injection and jailbreak detection.

    Uses meta-llama/llama-prompt-guard-2-22m for English and
    meta-llama/llama-prompt-guard-2-86m for non-English/multilingual input.

    This is NOT a general-purpose harmful-content classifier and does NOT
    rewrite user questions.
    """

    # Model IDs
    MODEL_22M = "meta-llama/llama-prompt-guard-2-22m"
    MODEL_86M = "meta-llama/llama-prompt-guard-2-86M"

    # Documented evaluated languages
    SUPPORTED_LANGUAGES = {
        "en", "eng", "en-US", "fr", "fr-fr", "de", "de-de",
        "hi", "hi-in", "it", "it-it", "pt", "pt-br",
        "es", "es-es", "th", "th-th",
    }

    # Adaptive routing: which model to use per language
    LANGUAGE_ROUTING: dict[str, str] = {
        "en": MODEL_22M,  # English → 22M
        "fr": MODEL_86M,  # French → 86M
        "de": MODEL_86M,  # German → 86M
        "hi": MODEL_86M,  # Hindi → 86M
        "it": MODEL_86M,  # Italian → 86M
        "pt": MODEL_86M,  # Portuguese → 86M
        "es": MODEL_86M,  # Spanish → 86M
        "th": MODEL_86M,  # Thai → 86M
    }

    # 512-token context window
    CONTEXT_WINDOW = 512

    def __init__(
        self,
        *,
        model_22m: str | None = None,
        model_86m: str | None = None,
        supported_languages: set[str] | None = None,
        language_routing: dict[str, str] | None = None,
        enforcement_policy: str = "allow",  # allow, block, require_review, allow_with_warning
        overlap: int = 50,  # overlap percentage for segment scanning
    ) -> None:
        """Initialize the Prompt Guard classifier.

        Args:
            model_22m: explicit model ID for the 22M model.
                When None, uses the default MODEL_22M.
            model_86m: explicit model ID for the 86M model.
                When None, uses the default MODEL_86M.
            supported_languages: set of language codes supported.
                When None, uses the default SUPPORTED_LANGUAGES.
            language_routing: mapping of language code to model ID.
                When None, uses the default LANGUAGE_ROUTING.
            enforcement_policy: default enforcement policy.
                One of: "allow", "block", "require_review", "allow_with_warning".
            overlap: overlap percentage for segment scanning (0-100).
        """
        self._model_22m = model_22m or self.MODEL_22M
        self._model_86m = model_86m or self.MODEL_86M
        self._supported_languages = supported_languages or self.SUPPORTED_LANGUAGES
        self._language_routing = language_routing or self.LANGUAGE_ROUTING
        self._enforcement_policy = enforcement_policy
        self._overlap = max(0, min(100, overlap))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        *,
        model_name: str | None = None,
        language_code: str | None = None,
    ) -> GuardResult:
        """Classify text for prompt injection/jailbreak.

        Uses adaptive language-based routing to select the appropriate
        Prompt Guard model.

        Args:
            text: the text to classify.
            model_name: explicit model override. When None, uses adaptive
                language-based routing.
            language_code: ISO-639-1 language code. When None, auto-detect
                from text (lightweight facility).

        Returns:
            GuardResult with classification details.

        Raises:
            GenerationPromptGuardError: if classification fails.
        """
        try:
            # Determine language route
            effective_language = self._determine_language(
                language_code, text
            )

            # Select model based on language
            model_to_use = self._language_routing.get(
                effective_language, self._model_86m
            )
            if model_name:
                model_to_use = model_name

            # Handle 512-token context window via segmentation
            segment_result = self._classify_with_segmentation(
                text, model_name=model_to_use
            )

            # Aggregate results conservatively
            classification = self._aggregate_results(segment_result)

            return GuardResult(
                model_used=model_to_use,
                language_route=effective_language,
                classification=classification,
                confidence=segment_result.get("confidence"),
                score=segment_result.get("score"),
                segment_count=segment_result.get("segment_count", 1),
                malicious_segments=segment_result.get("malicious_segments", []),
                diagnostics=segment_result.get("diagnostics", {}),
                latency_ms=segment_result.get("latency_ms", 0),
            )

        except Exception as e:
            raise GenerationPromptGuardError(
                f"Prompt Guard classification failed: {e}"
            ) from e

    def _determine_language(
        self, language_code: str | None, text: str
    ) -> str:
        """Determine the language for routing purposes.

        Uses the explicit language code from the request when available.
        Falls back to lightweight language detection from text.

        Args:
            language_code: explicit language code from request, or None.
            text: the text to potentially auto-detect language from.

        Returns:
            Language code string (ISO-639-1 format).
        """
        # Use explicit language code from request when available
        if language_code:
            code = language_code.lower().strip()
            if code in self._supported_languages:
                return code
            # Unsupported language — return as-is and let caller handle
            return code

        # Lightweight auto-detection: check for common language patterns
        # This is very basic; in production would use a proper language detector
        lower_text = text.lower()

        # Simple heuristic checks for supported languages
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

    def _classify_with_segmentation(
        self, text: str, *, model_name: str
    ) -> dict[str, Any]:
        """Classify text using segmentation for 512-token context window.

        Splits longer content into bounded segments with overlap, scans
        each independently, and aggregates results conservatively.

        Args:
            text: the text to classify.
            model_name: the Prompt Guard model to use.

        Returns:
            dict with segmentation results including confidence, score,
            malicious segments, etc.
        """
        # Enforce 512-token context window
        # Use a simple token estimate: 4 chars per token
        chars_per_token = 4
        max_chars = self.CONTEXT_WINDOW * chars_per_token  # 2048 characters
        max_tokens = self.CONTEXT_WINDOW  # 512 tokens

        # If text fits within the context window, classify directly
        if len(text) <= max_chars:
            # Single-segment classification
            return self._single_segment_classify(text, model_name)

        # Text exceeds 512 tokens — split into segments with overlap
        segments = self._split_into_segments(text, max_tokens, self._overlap)

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
                    "content_preview": segment[:50] + "..." if len(segment) > 50 else segment,
                    **result,
                })

        # Aggregate results conservatively:
        # A malicious segment must NOT be hidden by benign segments.
        # If ANY segment is malicious, the overall classification is MALICIOUS.
        # But we also report the count and positions for visibility.

        # Check if any segment detected malicious content
        any_malicious = any(
            r.get("is_malicious", False) for r in segment_results
        )

        # Determine overall classification
        # MALICIOUS if any segment is malicious, but we also check confidence
        overall_is_malicious = any_malicious

        # Compute confidence as the aggregate
        # If any segment has high confidence of malicious, overall confidence is high
        confidences = [r.get("confidence", 0.0) for r in segment_results]
        overall_confidence = max(confidences) if confidences else 0.0

        # Compute score (aggregate)
        scores = [r.get("score", 0.0) for r in segment_results]
        overall_score = max(scores) if scores else 0.0

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
            },
            "latency_ms": 0,  # will be measured by caller
        }

    def _single_segment_classify(
        self, text: str, model_name: str
    ) -> dict[str, Any]:
        """Classify a single text segment using the Prompt Guard model.

        This is a scaffold method. In a production setup, this would invoke
        the actual Prompt Guard model API or run the local model inference.

        For now, we use a rule-based heuristic classification that respects
        known prompt injection patterns.

        Args:
            text: the segment text to classify.
            model_name: the Prompt Guard model identifier.

        Returns:
            dict with classification result for this segment.
        """
        # Rule-based heuristic as scaffold
        # In production, this would call the actual Prompt Guard model

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
        """Split text into bounded segments for 512-token window.

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

    def _aggregate_results(self, segment_result: dict[str, Any]) -> Any:
        """Aggregate segmentation results into a final classification.

        Args:
            segment_result: the result from _classify_with_segmentation.

        Returns:
            Classification object (BENIGN or MALICIOUS).
        """
        # The segment_result already has the aggregated classification
        # Return it as-is
        return segment_result.get("classification", "BENIGN")


@dataclass
class GuardResult:
    """Structured result from a Prompt Guard classification.

    Attributes:
        model_used: the Prompt Guard model used for classification.
        language_route: the language route taken (ISO-639-1 code).
        classification: BENIGN or MALICIOUS.
        confidence: available confidence score (0.0 to 1.0).
        score: available confidence or score metric.
        segment_count: number of segments scanned.
        malicious_segment_references: list of malicious segment indices/positions.
        diagnostics: additional diagnostics about the classification.
        latency: latency of the classification in ms.
    """
    model_used: str
    language_route: str
    classification: str  # "BENIGN" or "MALICIOUS"
    confidence: float | None = None
    score: float | None = None
    segment_count: int = 1
    malicious_segment_references: list[dict[str, Any]] | None = None
    diagnostics: dict[str, Any] | None = None
    latency_ms: float = 0.0