"""Length counters for chunkers.

Chunkers need a way of measuring "how big is this text". Character counts are
the default and are reported honestly as characters -- they are NOT claimed to
be model token counts. A pluggable ``TokenCounter`` (or any plain callable
``str -> int``) can be supplied for tokenizer-based sizing.

Two built-in counters are provided:

- ``CharacterCounter`` -- counts Unicode code points (``len(text)``). Exact
  for characters; says nothing about tokens.
- ``ApproxTokenCounter`` -- heuristic estimate (~4 characters per token for
  English prose). Intended as a rough budgeting aid ONLY. It is not exact for
  any real tokenizer and must not be presented as one.

For exact model-token budgeting, wrap a real tokenizer, e.g.::

    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    config = FixedSizeConfig(chunk_size=512, overlap=64, tokenizer=enc.encode)

Any callable taking a string and returning an int is accepted; it is adapted
to the ``TokenCounter`` protocol automatically.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenCounter(Protocol):
    """Measures the length of a piece of text in some unit."""

    @property
    def name(self) -> str:
        """Stable identifier for the counter (e.g. ``"character"``)."""
        ...

    @property
    def unit(self) -> str:
        """Unit reported in chunk metadata (e.g. ``"characters"``, ``"tokens"``)."""
        ...

    def count(self, text: str) -> int:
        """Return the length of ``text`` in this counter's unit."""
        ...


@dataclass(frozen=True)
class CharacterCounter:
    """Counts Unicode code points. Honest about what it measures."""

    @property
    def name(self) -> str:
        return "character"

    @property
    def unit(self) -> str:
        return "characters"

    def count(self, text: str) -> int:
        return len(text)


@dataclass(frozen=True)
class ApproxTokenCounter:
    """Very rough ~4-chars-per-token heuristic. NOT a real tokenizer."""

    @property
    def name(self) -> str:
        return "approx_token"

    @property
    def unit(self) -> str:
        return "approx_tokens"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class CallableCounter:
    """Adapts a plain ``str -> int`` callable to the TokenCounter protocol."""

    func: Callable[[str], int]
    counter_name: str = "callable"
    counter_unit: str = "tokens"

    @property
    def name(self) -> str:
        return self.counter_name

    @property
    def unit(self) -> str:
        return self.counter_unit

    def count(self, text: str) -> int:
        value = self.func(text)
        if not isinstance(value, int):
            value = int(value)
        if value < 0:
            raise ValueError(f"tokenizer returned a negative length: {value}")
        return value


CounterLike = TokenCounter | Callable[[str], int]


def ensure_counter(counter: CounterLike | None, *, default_name: str = "character") -> TokenCounter:
    """Normalize ``counter`` to a TokenCounter.

    ``None`` yields the character counter (the documented default). Plain
    callables are wrapped in :class:`CallableCounter`.
    """
    if counter is None:
        return CharacterCounter()
    if isinstance(counter, TokenCounter):
        return counter
    if callable(counter):
        name = getattr(counter, "__name__", default_name)
        return CallableCounter(counter, counter_name=name, counter_unit="tokens")
    raise TypeError(
        f"tokenizer must be None, a TokenCounter or a callable str -> int; got {type(counter).__name__}"
    )
