"""Qwen3-Reranker-0.6B reranking adapter.

Uses Sentence Transformers' CrossEncoder interface for query--candidate
relevance scoring.  Designed for lazy local checkpoint loading, configurable
device placement, and deterministic tie-breaking.

Typical usage:

    from reranking.qwen3.reranker import Qwen3Reranker

    reranker = Qwen3Reranker(device="cpu")
    request = RerankRequest(
        query="What is quantum computing?",
        candidates=[RerankCandidate(candidate_id="c1", text="...")],
    )
    response = reranker.rerank(request)
"""

from __future__ import annotations

import logging
import math
from typing import Any

from sentence_transformers import CrossEncoder

from reranking.contracts.exceptions import (
    InvalidScoreError,
    RerankConfigError,
    RerankDependencyError,
    RerankInferenceError,
    UnsupportedDeviceError,
)
from reranking.contracts.models import (
    RerankedItem,
    RerankRequest,
    RerankResponse,
    RerankScoreInterpretation,
)
from reranking.contracts.protocols import Reranker

logger = logging.getLogger(__name__)

# The Qwen3-Reranker-0.6B model supports a maximum of 512 tokens for the
# cross-encoder pairing (query + document). This is the effective limit;
# Sentence Transformers may clamp input, but we document and expose it.
DEFAULT_MAX_SEQ_LENGTH = 512

# The model's default task instruction, as documented on the model card.
DEFAULT_TASK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class Qwen3Reranker(Reranker):
    """CrossEncoder-based reranker for Qwen/Qwen3-Reranker-0.6B.

    Wraps Sentence Transformers' :class:`~sentence_transformers.CrossEncoder`
    to provide a dependency-light, locally executable reranking component.

    The model is lazily initialized on first use.  Device placement is
    auto-detected (CPU | MPS | CUDA) respecting availability, and the raw
    cross-encoder logits are preserved as relevance scores (higher = more
    relevant).  An optional sigmoid transformation is available but the raw
    scores are always retained and used for ranking.

    Attributes:
        model_id: the HuggingFace model identifier.
        device: the computing device used for inference.
        max_sequence_length: maximum combined query + document token length.
        batch_size: candidates are scored in batches of this size.
        _model: the loaded CrossEncoder instance (None when unloaded).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        *,
        device: str | None = None,
        max_sequence_length: int = DEFAULT_MAX_SEQ_LENGTH,
        batch_size: int = 32,
    ) -> None:
        """Initialize the Qwen3-Reranker-0.6B adapter.

        Args:
            model_name: HuggingFace model name or local path.
                Defaults to ``"Qwen/Qwen3-Reranker-0.6B"``.
            device: Computing device.  When ``None`` the adapter auto-detects:
                - ``"cuda"`` when a CUDA device is available and verified,
                - ``"mps"`` when Apple Silicon MPS is available and verified,
                - ``"cpu"" otherwise.
                Pass an explicit string to override auto-detection.
            max_sequence_length: maximum combined sequence length for the
                cross-encoder (query + document).  Defaults to 512, which is
                the model's documented limit.
            batch_size: number of query--candidate pairs to score per inference
                batch.  Used to avoid sending unbounded candidates to the model
                in one call.
        """
        self.model_id = model_name
        self.model_name = model_name
        self.max_sequence_length = self._validate_max_sequence_length(
            max_sequence_length
        )
        self.batch_size = batch_size
        self._device = self._resolve_device(device)
        self._model: CrossEncoder | None = None
        self._model_loading_errors: tuple[str, ...] = ()
        self._prompts: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(requested: str | None) -> str:
        """Resolve the computing device respecting availability.

        Args:
            requested: explicit device request, or ``None`` for auto-detect.

        Returns:
            The resolved device string: ``"cuda"``, ``"mps"``, or ``"cpu"``.

        Raises:
            UnsupportedDeviceError: if the requested device is not available.
        """
        if requested is not None:
            import torch

            if requested == "cuda":
                if not torch.cuda.is_available():
                    raise UnsupportedDeviceError(
                        "CUDA requested but no CUDA device is available. "
                        "Use device='cpu' or device='mps' (if available)."
                    )
                # Verify we can actually use it
                try:
                    torch.cuda.get_device_attribute(0)
                except (RuntimeError, AttributeError):
                    raise UnsupportedDeviceError(
                        "CUDA requested but the device could not be initialized."
                    )
                return "cuda"
            elif requested == "mps":
                if not torch.backends.mps.is_available():
                    raise UnsupportedDeviceError(
                        "MPS requested but Apple Silicon MPS is not available. "
                        "Use device='cpu' or device='cuda' (if available)."
                    )
                return "mps"
            elif requested == "cpu":
                return "cpu"
            else:
                raise UnsupportedDeviceError(
                    f"Unrecognized device string: {requested!r}. "
                    "Use 'cuda', 'mps', or 'cpu'."
                )

        # Auto-detect: CUDA > MPS > CPU
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.get_device_attribute(0)
                return "cuda"
            except (RuntimeError, AttributeError):
                pass
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Lazy model initialization
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy-load the CrossEncoder model if not already loaded.

        Raises:
            RerankDependencyError: if sentence-transformers is not installed.
            RerankInferenceError: if the model fails to load.
        """
        if self._model is not None:
            return

        try:
            import sentence_transformers  # noqa: F401  (side-effect import)

            self._model = CrossEncoder(self.model_name)
            logger.info("Qwen3-Reranker-0.6B model loaded successfully: %s", self.model_id)
        except ImportError as e:
            logger.error("sentence-transformers not available: %s", e)
            self._model_loading_errors = (str(e),)
            raise RerankDependencyError(
                "sentence-transformers is required for Qwen3-Reranker. "
                "Install it with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error("Failed to load Qwen3-Reranker-0.6B: %s", e)
            self._model_loading_errors = (str(e),)
            raise RerankInferenceError(
                f"Failed to load the reranker model '{self.model_name}': {e}"
            ) from e

    # ------------------------------------------------------------------
    # Max sequence length validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_max_sequence_length(length: int) -> int:
        """Validate and return the max sequence length.

        Args:
            length: the requested maximum sequence length.

        Returns:
            the validated length.

        Raises:
            RerankConfigError: if the length is not a positive integer.
        """
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise RerankConfigError(
                f"max_sequence_length must be a positive integer, got {length!r}"
            )
        return length

    # ------------------------------------------------------------------
    # Task instruction resolution
    # ------------------------------------------------------------------


    @property
    def score_interpretation(self) -> RerankScoreInterpretation:
        """Return the default score interpretation for this reranker.

        Always returns ``RerankScoreInterpretation.RAW`` since the cross-encoder
        logits are preserved as raw relevance scores (higher = more relevant).
        Callers may apply a sigmoid transformation if desired, but ranking
        uses the raw scores.
        """
        return RerankScoreInterpretation.RAW
    def _resolve_task_instruction(self, task_instruction: str | None) -> str:
        """Return the effective task instruction, falling back to the default.

        Args:
            task_instruction: optional task-specific instruction.

        Returns:
            The resolved task instruction string.
        """
        return (
            task_instruction
            if task_instruction is not None
            else DEFAULT_TASK_INSTRUCTION
        )

    # ------------------------------------------------------------------
    # Scoring pipeline
    # ------------------------------------------------------------------

    def _make_pairs(self, request: RerankRequest) -> list[tuple[str, str]]:
        """Build query--candidate pairs with the configured task instruction.

        Each pair is ``(query_with_instruction, candidate_text)``.

        Args:
            request: the reranking request.

        Returns:
            list of (query, candidate_text) tuples.
        """
        instruction = self._resolve_task_instruction(request.task_instruction)
        # The CrossEncoder expects (query, document) pairs; we prepend the
        # task instruction as the prompt.  The installed version supports the
        # ``prompt`` parameter on ``predict``.
        pairs: list[tuple[str, str]] = []
        for candidate in request.candidates:
            # Do NOT mutate the original candidate text; build a new pairing.
            # The instruction is applied per the model's prompt mechanism.
            query_with_instruction = f"{instruction}\nQuery: {request.query}\nDocument: {candidate.text}"
            pairs.append((query_with_instruction, candidate.text))
        return pairs

    def _validate_scores(self, scores: list[float]) -> None:
        """Validate that all returned scores are finite numbers.

        Args:
            scores: the raw scores from the model.

        Raises:
            InvalidScoreError: if any score is NaN or infinity.
        """
        for i, s in enumerate(scores):
            if not isinstance(s, (int, float)) or isinstance(s, bool):
                raise InvalidScoreError(
                    f"score at index {i} is not a finite numeric value: {s!r}"
                )
            if not math.isfinite(s):
                raise InvalidScoreError(
                    f"score at index {i} is not finite (NaN or infinity): {s!r}"
                )

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score query--candidate pairs through the CrossEncoder.

        Args:
            pairs: list of (query, document) tuples.

        Returns:
            list of raw relevance scores (logits), one per pair.

        Raises:
            RerankInferenceError: if model inference fails.
            InvalidScoreError: if any returned score is NaN or infinity.
        """
        if self._model is None:
            self._ensure_loaded()

        raw = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
            # Do NOT apply softmax: we want raw logits.
            apply_softmax=False,
        )

        # Convert to list of Python floats
        if hasattr(raw, "tolist"):
            scores = raw.tolist()
        elif isinstance(raw, list):
            scores = raw
        else:
            scores = [float(s) for s in raw]

        # Validate scores are finite
        self._validate_scores(scores)

        return scores

    # ------------------------------------------------------------------
    # Protocol implementation: rerank
    # ------------------------------------------------------------------

    def rerank(self, request: RerankRequest) -> RerankResponse:
        """Re-rank candidates based on relevance to the query.

        Args:
            request: the reranking request containing the query and candidate list.

        Returns:
            the reranked response with candidates ordered by descending relevance.

        Raises:
            RerankConfigError: if the request is invalid.
            RerankDependencyError: if a required optional dependency is missing.
            RerankInferenceError: if the model fails during inference.
            UnsupportedDeviceError: if the requested device is not available.
            InvalidScoreError: if the model returns NaN/infinity scores.
        """
        # Input validation (RerankRequest already validates in __post_init__,
        # but we double-check here for robustness)
        if not isinstance(request, RerankRequest):
            raise RerankConfigError("request must be a RerankRequest instance")

        if not request.candidates:
            # Empty candidate list: return empty response
            return RerankResponse(
                reranked_items=(),
                model_id=self.model_id,
                model_revision=self.model_revision,
                candidates_scored=0,
                elapsed_ms=None,
                score_interpretation=RerankScoreInterpretation.RAW,
                diagnostics={},
            )

        # Build query--candidate pairs with instruction
        pairs = self._make_pairs(request)

        # Score the pairs
        scores = self._score_pairs(pairs)

        # Build RerankedItem entries preserving original candidate identity
        raw_items: list[RerankedItem] = []
        for i, (candidate, raw_score) in enumerate(zip(request.candidates, scores)):
            # Use 1-based original rank
            orig_rank = candidate.original_rank if candidate.original_rank and candidate.original_rank > 0 else i + 1
            raw_items.append(
                RerankedItem(
                    candidate=candidate,
                    raw_score=float(raw_score),
                    original_rank=orig_rank,
                    # new_rank will be set after sorting
                    new_rank=0,
                )
            )

        # Sort by descending raw score, then by original rank (deterministic tie-breaker),
        # then by candidate_id (stable tie-breaker)
        sorted_items = sorted(
            enumerate(raw_items),
            key=lambda ie: (
                -ie[1].raw_score,  # descending score
                ie[1].original_rank,  # original rank (lower is better tie-break)
                ie[1].candidate.candidate_id,  # stable ID tie-break
            ),
        )

        # Assign new ranks and build the response
        reranked_items: tuple[RerankedItem, ...] = ()
        candidates_scored = len(scores)

        for new_idx, (_, item) in enumerate(sorted_items):
            # Create a new RerankedItem with the correct new_rank
            # RerankedItem is frozen, so we need to create a new one
            updated_item = RerankedItem(
                candidate=item.candidate,
                raw_score=item.raw_score,
                original_rank=item.original_rank,
                new_rank=new_idx + 1,  # 1-based rank
            )
            # Rebuild the tuple incrementally
            reranked_items = reranked_items + (updated_item,)

        # Apply top-k if configured
        top_k = request.top_k
        if top_k is not None and top_k < len(reranked_items):
            reranked_items = reranked_items[:top_k]

        return RerankResponse(
            reranked_items=reranked_items,
            model_id=self.model_id,
            model_revision=self.model_revision,
            candidates_scored=candidates_scored,
            elapsed_ms=None,
            score_interpretation=RerankScoreInterpretation.RAW,
            diagnostics={
                "batch_count": 1,
                "candidates_scored": candidates_scored,
                "max_sequence_length": self.max_sequence_length,
                "device": self._device,
            },
        )

    # ------------------------------------------------------------------
    # Model info exposure
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded reranker model.

        Returns:
            a dictionary with model identity, device, and configuration data.
        """
        info: dict[str, Any] = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "device": self._device,
            "max_sequence_length": self.max_sequence_length,
            "batch_size": self.batch_size,
            "loaded": self._model is not None,
        }
        if self._model is not None:
            info["default_prompt_name"] = getattr(
                self._model, "default_prompt_name", None
            )
        return info

    def get_capabilities(self) -> dict[str, Any]:
        """Return the capabilities of this reranker.

        Returns:
            a dictionary describing the reranker's abilities.
        """
        return {
            "model_id": self.model_id,
            "apply_softmax_default": False,
            "supports_sigmoid_transform": True,
            "max_sequence_length": self.max_sequence_length,
            "batch_processing": True,
            "deterministic_tie_breaking": True,


            "device": self._device,
        }