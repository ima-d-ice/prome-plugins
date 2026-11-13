"""Model capability reporting for the generation ecosystem.

Each provider reports which capabilities it actually supports.
This is independent of any specific backend implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Supports(str, Enum):
    """Boolean-like support enumeration."""

    YES = "yes"
    NO = "no"
    MAYBE = "maybe"


@dataclass(frozen=True)
class ModelCapabilityReport:
    """Report from a provider about its capabilities for a specific model."""

    model_id: str
    supports_streaming: Supports
    supports_temperature: Supports
    supports_top_p: Supports
    supports_reasoning_effort: Supports
    supports_structured_output: Supports
    supports_stop_sequences: Supports
    max_context_window: int
    max_output_tokens: int