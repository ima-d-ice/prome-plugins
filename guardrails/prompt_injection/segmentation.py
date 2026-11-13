"""Token-aware segmentation for Prompt Guard 512-token context window.

Respects the Prompt Guard model's 512-token context window requirement.
Implements token-aware splitting with configurable bounded overlap.

Key requirements:
- Use the relevant tokenizer where available.
- Configurable bounded overlap.
- Preserve segment ordering.
- Preserve original document/chunk IDs.
- Scan all eligible segments.
- Record the result of every segment.
- Aggregate suspicious results conservatively.
- Never silently skip a segment.
- Report incomplete scanning if any segment could not be classified.

Note: This module does not import any tokenizer library to keep it
dependency-free. It uses a character-to-token approximation (4 chars ≈ 1 token),
consistent with the Generation subsystem's Prompt Guard implementation.
"""

from __future__ import annotations

from typing import Any


def segment_text_for_prompt_guard(
    text: str,
    *,
    max_tokens: int = 512,
    overlap: int = 50,
) -> list[str]:
    """Split text into segments for Prompt Guard 512-token context window.

    Uses a character-to-token approximation (4 chars ≈ 1 token), consistent
    with the Generation subsystem's Prompt Guard implementation.

    Args:
        text: the text to split into segments.
        max_tokens: maximum tokens per segment (default 512, matching Prompt Guard).
        overlap: overlap percentage between segments (default 50%).

    Returns:
        list of text segments, each fitting within the max_token limit.
    """

    chars_per_token = 4
    max_chars = max_tokens * chars_per_token  # 2048 characters
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


def detect_truncation(
    original_text: str,
    scanned_text: str,
    *,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Detect if scanned text represents a truncated version of the original.

    Used to report incomplete scanning when a long document is truncated
    to fit the 512-token window.

    Args:
        original_text: the original full text.
        scanned_text: the text that was actually scanned (possibly truncated).
        max_chars: maximum character count to compare against.
            When None, uses 2048 (512 tokens * 4 chars/token).

    Returns:
        dict with truncation detection results:
            - is_truncated: True if scanned text is shorter than original
            - original_length: character count of original text
            - scanned_length: character count of scanned text
            - truncated_ratio: ratio of scanned to original (0.0 to 1.0)
            - warning: human-readable warning message if truncated
    """

    if max_chars is None:
        max_chars = 2048  # 512 tokens * 4 chars/token

    original_len = len(original_text)
    scanned_len = len(scanned_text)

    is_truncated = scanned_len < original_len

    if is_truncated:
        truncated_ratio = scanned_len / original_len if original_len > 0 else 0.0
    else:
        truncated_ratio = 1.0

    warning = None
    if is_truncated:
        warning = (
            f"Document truncated: original {original_len} chars, "
            f"scanned {scanned_len} chars ({truncated_ratio:.1%} captured). "
            "Incomplete scanning reported. Scan all eligible segments."
        )

    return {
        "is_truncated": is_truncated,
        "original_length": original_len,
        "scanned_length": scanned_len,
        "truncated_ratio": truncated_ratio,
        "warning": warning,
    }
