"""Token budget tracking for a model's token consumption limits.

Tracks TPM (tokens per minute) and TPD (tokens per day) independently.
Used by the rate limiter and key pool for token accounting.

Contracts are dependency-free: no imports of PyTorch, Transformers, or tokenizers.
Token counting is delegated to the token_counter package.
"""

from __future__ import annotations


class TokenBudget:
    """Token budget for a single key/model combination.

    Tracks TPM (tokens per minute) and TPD (tokens per day) independently.

    The first exhausted applicable limit controls whether a request can proceed.
    """

    def __init__(
        self,
        *,
        tpm: int,
        tpd: int,
    ) -> None:
        """Initialize the token budget.

        Args:
            tpm: tokens per minute limit.
            tpd: tokens per day limit.
        """
        self._tpm = tpm
        self._tpd = tpd

        # Remaining counters (mutable state)
        self._tpm_remaining = tpm
        self._tpd_remaining = tpd

        # Tracking timestamps
        self._tpm_last_reset = __import__("time").monotonic()
        self._tpd_last_reset = __import__("time").monotonic()

        # Daily token counter for TPD tracking
        self._daily_tokens_consumed = 0

    # ------------------------------------------------------------------
    # Budget checking
    # ------------------------------------------------------------------

    def check_budget(
        self,
        estimated_input_tokens: int,
        requested_output_tokens: int,
    ) -> tuple[bool, str | None]:
        """Check if the request fits within the token budgets.

        Args:
            estimated_input_tokens: estimated input token count.
            requested_output_tokens: requested output token count.

        Returns:
            (can_proceed, reason) where reason is None if can_proceed,
            or a string describing the exhausted limit.
        """
        now = __import__("time").monotonic()

        # --- TPM: Tokens Per Minute reset ---
        if now - self._tpm_last_reset >= 60.0:
            self._tpm_remaining = self._tpm
            self._tpm_last_reset = now

        total_estimated = estimated_input_tokens + requested_output_tokens

        if self._tpm_remaining < total_estimated:
            if self._tpm_remaining <= 0:
                return False, "TPM exhausted"
            # Conservative: don't decrement on check; the caller will
            # reconcile after success. Just return False.
            return False, "TPM insufficient"

        # OK for TPM

        # --- TPD: Tokens Per Day reset ---
        if now - self._tpd_last_reset >= 86400.0:
            self._tpd_remaining = self._tpd
            self._tpd_last_reset = now
            self._daily_tokens_consumed = 0

        if self._tpd_remaining < total_estimated and self._tpd_remaining <= 0:
            return False, "TPD exhausted"

        # All budgets OK
        return True, None

    # ------------------------------------------------------------------
    # State reconciliation (called after successful request)
    # ------------------------------------------------------------------

    def record_usage(self, actual_input_tokens: int, actual_output_tokens: int) -> None:
        """Record actual token usage after a successful request.

        Args:
            actual_input_tokens: actual input tokens consumed.
            actual_output_tokens: actual output tokens generated.
        """
        total = actual_input_tokens + actual_output_tokens

        self._tpm_remaining = max(0, self._tpm_remaining - total)
        self._tpd_remaining = max(0, self._tpd_remaining - total)
        self._daily_tokens_consumed = max(0, self._daily_tokens_consumed + total)

    def remaining(self) -> dict[str, int]:
        """Return remaining budgets.

        Returns:
            dict with tpm and tpd keys.
        """
        return {
            "tpm": self._tpm_remaining,
            "tpd": self._tpd_remaining,
        }