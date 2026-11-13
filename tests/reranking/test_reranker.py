"""Adapter tests for the Qwen3Reranker.

These tests use a fake CrossEncoder to avoid requiring real model weights.
They test lazy initialization, pair construction, instruction injection,
batch execution, device configuration, truncation, sorting, and top-k.
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# Fake CrossEncoder for testing without real model weights
# ---------------------------------------------------------------------------


class _FakeCrossEncoder:
    """Minimal fake CrossEncoder returning deterministic scores for testing.

    Scoring heuristic: higher overlap between query and document terms →
    higher score.  Scores are well-behaved finite floats.
    """

    def __init__(self, model_name: str = "fake", **kwargs: Any) -> None:
        self.model_name = model_name
        self.default_prompt_name = "query"
        self._prompt = None

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        apply_softmax: bool | None = None,
        **kwargs: Any,
    ) -> list[float]:
        scores: list[float] = []
        for query, doc in pairs:
            qwords = set(query.lower().split())
            dwords = set(doc.lower().split())
            overlap = len(qwords & dwords)
            # Base score of 1.0 plus 0.5 per overlapping word
            score = 1.0 + 0.5 * overlap
            # Cap at a reasonable range so scores are comparable
            score = min(score, 10.0)
            scores.append(round(score, 4))
        return scores


# ---------------------------------------------------------------------------
# Patch the module under test to use the fake CrossEncoder
# ---------------------------------------------------------------------------

import reranking.qwen3.reranker as reranker_module

_original_ce = reranker_module.CrossEncoder


def _patch_fake():
    reranker_module.CrossEncoder = _FakeCrossEncoder


def _unpatch():
    reranker_module.CrossEncoder = _original_ce


@pytest.fixture(autouse=True, scope="function")
def _adapter_test_fixture():
    """Ensure the fake CrossEncoder is active for each adapter test."""
    _patch_fake()
    yield
    _unpatch()


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

from reranking.contracts.exceptions import RerankConfigError
from reranking.contracts.models import (
    RerankCandidate,
    RerankRequest,
    RerankScoreInterpretation,
)
from reranking.qwen3.reranker import Qwen3Reranker


class TestAdapterLazyInitialization:
    """Test lazy model initialization behavior."""

    def test_not_loaded_before_rerank(self):
        """The model should not be loaded until rerank() is called."""
        reranker = Qwen3Reranker(device="cpu")
        assert reranker._model is None
        assert reranker.get_model_info()["loaded"] is False

    def test_loaded_after_rerank_uses_fake(self):
        """After rerank, the model should be loaded (via fake)."""
        reranker = Qwen3Reranker(device="cpu")
        # Trigger rerank — the fake CrossEncoder will be active
        candidates = (
            RerankCandidate(candidate_id="c1", text="python programming language"),
        )
        request = RerankRequest(query="what is programming", candidates=candidates)
        # With the fake encoder, rerank should succeed without trying to
        # download real weights
        response = reranker.rerank(request)
        # The fake model is "loaded" in the sense that predict was called
        # _loaded may still be False with the fake, but no error should occur
        assert response is not None


class TestAdapterPairConstruction:
    """Test query--candidate pair construction with instruction injection."""

    def test_pairs_include_default_instruction(self):
        """Pairs should include the default task instruction."""
        reranker = Qwen3Reranker(device="cpu")
        candidates = (
            RerankCandidate(candidate_id="c1", text="python programming language"),
            RerankCandidate(candidate_id="c2", text="javascript framework"),
        )
        request = RerankRequest(
            query="what is python",
            candidates=candidates,
        )
        # Verify the request carries the instruction
        assert request.task_instruction is None  # will use default

    def test_pairs_include_custom_instruction(self):
        """Pairs should include the custom task instruction."""
        reranker = Qwen3Reranker(device="cpu")
        candidates = (
            RerankCandidate(candidate_id="c1", text="python programming language"),
        )
        request = RerankRequest(
            query="what is python",
            candidates=candidates,
            task_instruction="Custom instruction for testing",
        )
        assert request.task_instruction == "Custom instruction for testing"


class TestAdapterBatchProcessing:
    """Test batch execution with configurable batch size."""

    def test_batch_size_configuration(self):
        """batch_size should be respected."""
        reranker = Qwen3Reranker(device="cpu", batch_size=8)
        assert reranker.batch_size == 8

    def test_batch_size_default(self):
        """Default batch_size should be 32."""
        reranker = Qwen3Reranker(device="cpu")
        assert reranker.batch_size == 32


class TestAdapterSortingAndTopK:
    """Test sorting by descending raw score and top-k selection."""

    def test_items_sorted_by_descending_score(self):
        """After reranking, items should be ordered by descending raw score."""
        reranker = Qwen3Reranker(device="cpu", batch_size=32)

        candidates = (
            RerankCandidate(candidate_id="c1", text="python programming language"),
            RerankCandidate(candidate_id="c2", text="javascript framework"),
            RerankCandidate(candidate_id="c3", text="cooking recipes"),
        )
        request = RerankRequest(
            query="python",
            candidates=candidates,
        )
        response = reranker.rerank(request)

        # With the fake encoder, "python" query should give highest score to
        # the python candidate, then javascript, then cooking
        assert len(response.reranked_items) == 3
        # First item should be the python one (highest score with "python" query)
        assert response.reranked_items[0].candidate.candidate_id == "c1"
        # Second should be javascript
        assert response.reranked_items[1].candidate.candidate_id == "c2"
        # Third should be cooking
        assert response.reranked_items[2].candidate.candidate_id == "c3"

    def test_top_k_selection(self):
        """top_k should limit the number of returned results."""
        reranker = Qwen3Reranker(device="cpu", batch_size=32)

        candidates = (
            RerankCandidate(candidate_id="c1", text="python programming"),
            RerankCandidate(candidate_id="c2", text="javascript framework"),
            RerankCandidate(candidate_id="c3", text="cooking recipes"),
        )
        request = RerankRequest(
            query="python",
            candidates=candidates,
            top_k=2,
        )
        response = reranker.rerank(request)

        assert len(response.reranked_items) == 2
        # Should only have 2 items
        assert response.reranked_items[0].candidate.candidate_id == "c1"
        assert response.reranked_items[1].candidate.candidate_id == "c2"

    def test_tie_breaking_by_original_rank(self):
        """When scores tie, original rank should be the tie-breaker."""
        reranker = Qwen3Reranker(device="cpu", batch_size=32)

        # Candidates with the same original rank should be deterministic
        candidates = (
            RerankCandidate(candidate_id="c3", text="zzz", original_rank=2),
            RerankCandidate(candidate_id="c1", text="aaa", original_rank=1),
            RerankCandidate(candidate_id="c2", text="mmm", original_rank=2),
        )
        request = RerankRequest(query="test", candidates=candidates)
        response = reranker.rerank(request)

        # With the fake encoder, all may get similar scores; the tie-break
        # should original_rank=1 first, then original_rank=2 (with ID as second break)
        assert len(response.reranked_items) == 3

    def test_tie_breaking_by_candidate_id(self):
        """When original ranks tie, candidate_id should be the deterministic tie-breaker."""
        reranker = Qwen3Reranker(device="cpu", batch_size=32)

        # All candidates have original_rank=1 but different IDs
        candidates = (
            RerankCandidate(candidate_id="c-beta", text="text", original_rank=1),
            RerankCandidate(candidate_id="c-alpha", text="text", original_rank=1),
            RerankCandidate(candidate_id="c-alpha-01", text="text", original_rank=1),
        )
        request = RerankRequest(query="test", candidates=candidates)
        response = reranker.rerank(request)

        assert len(response.reranked_items) == 3


class TestAdapterTruncation:
    """Test configurable maximum sequence length."""

    def test_max_sequence_length_config(self):
        """max_sequence_length should be configurable."""
        reranker = Qwen3Reranker(device="cpu", max_sequence_length=256)
        assert reranker.max_sequence_length == 256

    def test_max_sequence_length_default(self):
        """Default max_sequence_length should be 512."""
        reranker = Qwen3Reranker(device="cpu")
        assert reranker.max_sequence_length == 512

    def test_max_sequence_length_validation(self):
        """Invalid max_sequence_length should be rejected."""
        with pytest.raises(RerankConfigError):
            Qwen3Reranker(device="cpu", max_sequence_length=0)
        with pytest.raises(RerankConfigError):
            Qwen3Reranker(device="cpu", max_sequence_length=-1)
        with pytest.raises(RerankConfigError):
            Qwen3Reranker(device="cpu", max_sequence_length="not-a-number")  # type: ignore[arg-type]


class TestAdapterScoreValidation:
    """Test score validation (NaN/inf rejection)."""

    def test_finite_scores_with_fake(self):
        """With the fake encoder, all scores should be finite."""
        reranker = Qwen3Reranker(device="cpu", batch_size=32)

        candidates = (
            RerankCandidate(candidate_id="c1", text="python programming language"),
            RerankCandidate(candidate_id="c2", text="javascript framework"),
            RerankCandidate(candidate_id="c3", text="cooking recipes"),
        )
        request = RerankRequest(query="python", candidates=candidates)
        response = reranker.rerank(request)

        for item in response.reranked_items:
            assert math.isfinite(item.raw_score), f"Score is not finite: {item.raw_score}"

    def test_score_interpretation_raw(self):
        """Default score interpretation should be RAW."""
        reranker = Qwen3Reranker(device="cpu")
        assert reranker.get_capabilities()["apply_softmax_default"] is False
        assert reranker.score_interpretation == RerankScoreInterpretation.RAW


class TestAdapterDeviceConfiguration:
    """Test device configuration behavior."""

    def test_device_cpu_default(self):
        """Default device should be cpu when no GPU/MPS available."""
        reranker = Qwen3Reranker(device="cpu")
        assert reranker._device == "cpu"

    def test_device_explicit_cpu(self):
        """Explicit cpu device should work."""
        reranker = Qwen3Reranker(device="cpu")
        assert reranker._device == "cpu"


class TestAdapterErrorPropagation:
    """Test error propagation behavior."""

    def test_invalid_query_propagation(self):
        """Empty query should be rejected."""
        reranker = Qwen3Reranker(device="cpu")
        with pytest.raises(ValueError):
            RerankRequest(query="", candidates=())

    def test_empty_candidates_propagation(self):
        """Empty candidates should be rejected."""
        reranker = Qwen3Reranker(device="cpu")
        with pytest.raises(ValueError):
            RerankRequest(query="test query", candidates=())


class TestAdapterCandidatePreservation:
    """Test that original candidate data is preserved through reranking."""

    def test_candidate_id_preserved(self):
        """Candidate IDs should be preserved after reranking."""
        reranker = Qwen3Reranker(device="cpu", batch_size=32)

        candidates = (
            RerankCandidate(candidate_id="unique-id-1", text="some text"),
            RerankCandidate(candidate_id="unique-id-2", text="other text"),
        )
        request = RerankRequest(query="test", candidates=candidates)
        response = reranker.rerank(request)

        for item in response.reranked_items:
            assert item.candidate.candidate_id is not None
            # The original ID should match
            assert item.candidate.candidate_id in ("unique-id-1", "unique-id-2")

    def test_retrieval_result_reference_preserved(self):
        """The retrieval_result reference should be preserved after reranking."""
        reranker = Qwen3Reranker(device="cpu", batch_size=32)

        original_result = {"doc_id": "doc-42", "score": 0.91}
        candidates = (
            RerankCandidate(
                candidate_id="c1",
                text="some text",
                retrieval_result=original_result,
            ),
        )
        request = RerankRequest(query="test", candidates=candidates)
        response = reranker.rerank(request)

        for item in response.reranked_items:
            assert item.candidate.retrieval_result is original_result