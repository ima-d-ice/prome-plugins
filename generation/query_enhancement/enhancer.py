"""Query enhancement enhancer.

Optional query enhancement component that uses an injected generation
provider to improve/rewrite the query.

Key constraints (from the specification):
- Default: QUERY_ENHANCEMENT_ENABLED=false
- When enabled:
  * Uses an injected generation provider.
  * Requires explicit provider/model selection.
  * Does NOT call Prompt Guard as the enhancer.
  * Preserves the original query.
  * Returns the enhanced query separately.
  * Never mutates the original user message.
  * Makes the enhanced query available to retrieval without requiring
    retrieval code in this package.
"""

from __future__ import annotations

import os
from typing import Any

from generation.contracts.exceptions import (
    GenerationConfigError,
    GenerationDependencyError,
)
from generation.contracts.models import GenerationRequest, GenerationResponse


class QueryEnhancer:
    """Optional query enhancement component.

    Uses an injected generation provider for enhancement.
    Does not call Prompt Guard as the enhancer.
    Preserves the original query.

    Typical usage:
        enhancer = QueryEnhancer(provider=groq_provider)
        enhanced_request, original = enhancer.enhance(request)
    """

    def __init__(
        self,
        *,
        provider: Any | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the query enhancer.

        Args:
            provider: a generation provider instance (Groq, Ollama, etc.).
                Required for actual enhancement. When None, enhancement
                will return the request unchanged with diagnostics.
            model: explicit model ID for the enhancement call.
                When None, uses the provider's default model.
        """
        self._provider = provider
        self._model = model or os.environ.get("QUERY_ENHANCEMENT_MODEL")

    def enhance(
        self, request: GenerationRequest
    ) -> tuple[GenerationRequest, dict[str, Any]]:
        """Enhance the query in the given request.

        Uses an injected generation provider to produce an enhanced version
        of the query. The original query is preserved and returned separately.

        Args:
            request: the generation request containing the user query to
                enhance.

        Returns:
            (enhanced_request, diagnostics) where:
            - enhanced_request: a new GenerationRequest with the enhanced
              query (original user_input preserved, enhancement applied).
            - diagnostics: dict with enhancement details including
              original_query, enhancement_used, etc.

        Raises:
            GenerationConfigError: if the request is invalid.
            GenerationDependencyError: if no provider is configured and
                enhancement is attempted without one.
        """
        if not request.user_input.strip():
            raise GenerationConfigError("user_input must be a non-empty string")

        # Check if enhancement is enabled
        enhancement_enabled = os.environ.get(
            "QUERY_ENHANCEMENT_ENABLED", "false"
        ).lower() == "true"

        if not enhancement_enabled or self._provider is None:
            # Enhancement disabled or no provider — return request unchanged
            diagnostics = {
                "enhancement_enabled": enhancement_enabled,
                "enhancement_used": False,
                "original_query": request.user_input,
                "reason": "enhancement_disabled_or_no_provider",
            }
            # Return the same request (unchanged) and diagnostics
            # We create a new request to avoid mutating the original,
            # but with the same user_input
            from copy import deepcopy
            new_request = deepcopy(request)
            diagnostics["original_query"] = request.user_input
            return new_request, diagnostics

        # Enhancement enabled with provider — perform enhancement
        try:
            # Build a prompt for the enhancement provider
            # The enhancement provider should improve the query while
            # preserving its intent
            enhancement_prompt = self._build_enhancement_prompt(request.user_input)

            # Make the generation call
            enhancement_request = GenerationRequest(
                user_input=enhancement_prompt,
                messages=request.messages,
                context_items=request.context_items,
                system_instruction=request.system_instruction,
                provider=request.provider,
                model=self._model or request.model,
                language_code=request.language_code,
                generation_parameters=request.generation_parameters,
                input_token_limit=request.input_token_limit,
                output_token_limit=request.output_token_limit,
                enhance_query=False,  # Don't nest enhancement
                safety_config=request.safety_config,
                request_id=request.request_id,
            )

            # Generate using the provider
            response: GenerationResponse = self._provider.generate(enhancement_request)

            # The enhanced query is the generated text
            enhanced_text = response.text

            # Build the enhanced request — preserve original user_input
            # but we'll track both original and enhanced
            from copy import deepcopy
            new_request = deepcopy(request)

            diagnostics = {
                "enhancement_enabled": enhancement_enabled,
                "enhancement_used": True,
                "original_query": request.user_input,
                "enhanced_query": enhanced_text,
                "enhancement_model": response.actual_model,
                "enhancement_provider": response.actual_provider,
                "latency_ms": response.latency,
                "finish_reason": response.finish_reason,
                "token_usage": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                },
            }

            return new_request, diagnostics

        except Exception as e:
            raise GenerationDependencyError(
                f"Query enhancement failed: {e}"
            ) from e

    def _build_enhancement_prompt(self, original_query: str) -> str:
        """Build the prompt for the enhancement generation call.

        Constructs a prompt that asks the model to improve/rewrite the
        query while preserving its intent and not mutating the core meaning.

        Args:
            original_query: the original user query.

        Returns:
            The enhancement prompt string.
        """
        # Prompt that asks the model to enhance the query
        # while preserving intent and not changing the core meaning
        return (
            "You are a query enhancement assistant. Your task is to "
            "improve the clarity and effectiveness of the following user "
            "query for a RAG (Retrieval-Augmented Generation) system. "
            "Guidelines:\n"
            "1. Preserve the original intent and meaning of the query.\n"
            "2. Improve clarity and conciseness where helpful.\n"
            "3. Do NOT change the core question or factual claims.\n"
            "4. Remove any ambiguous or redundant phrasing.\n"
            "5. Output ONLY the enhanced query text, no explanations.\n\n"
            f"Original query: {original_query}\n\n"
            "Enhanced query:"
        )