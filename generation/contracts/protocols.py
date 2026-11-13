"""The generation protocol.

A generation provider turns a :class:`~generation.contracts.models.GenerationRequest`
into a :class:`~generation.contracts.models.GenerationResponse`. The contract is
deliberately small and independent of Ollama, Groq, PyTorch, Transformers, or
any HTTP SDK: it describes the *interface* only.

Implementations must:

- validate the request and fail loudly on invalid input;
- never modify the original request or context items;
- preserve the original request ID and safety configuration;
- return a :class:`~generation.contracts.models.GenerationResponse` with the
  generated text and diagnostics;
- support configurable generation parameters (temperature, top-p, etc.);
- support lazy provider initialization and configurable model selection.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from generation.contracts.models import GenerationRequest, GenerationResponse


@runtime_checkable
class GenerationProvider(Protocol):
    """Interface every Prome generation provider implements.

    The core operation is equivalent to:

        provider(request) -> response

    Implementations must:

    - validate the request and fail loudly on an empty user_input or invalid
      configuration;
    - produce a :class:`~generation.contracts.models.GenerationResponse` with
      the generated text and diagnostics;
    - support configurable generation parameters (temperature, top-p, max_tokens,
      reasoning_effort, stop sequences, structured_output);
    - support lazy model/connection initialization and configurable device placement;
    - report which capabilities it actually supports via :class:`~generation.contracts.models.Capabilities`;
    - support optional streaming where the backend supports it, but the protocol
      method itself is non-blocking and returns a standard GenerationResponse.
    """

    name: str

    @property
    def model_configuration(self) -> ModelConfiguration:
        """Return the model configuration for this provider's default model."""
        ...

    @property
    def capabilities(self) -> Capabilities:
        """Return the capabilities of this provider."""
        ...

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from the given request.

        Args:
            request: the generation request containing user input and options.

        Returns:
            the generation response with text and diagnostics.

        Raises:
            GenerationConfigError: if the request is invalid.
            GenerationDependencyError: if a required optional dependency is missing.
            GenerationInferenceError: if the model fails during inference.
            GenerationTimeoutError: if the operation times out.
        """
        ...