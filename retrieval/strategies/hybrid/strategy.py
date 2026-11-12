"""The hybrid retrieval strategy: dense + BM25, fused with RRF.

The strategy runs the two channels, fuses their rankings with the
reciprocal rank function and returns the standardized results.

Two properties are deliberate:

* the channels are **independent**.  Each one applies the same metadata
  filter to its own candidates, so a passage that the filter excludes can
  come from neither of them;
* a channel that fails is **reported**, not hidden.  The default is the
  strict policy: a failing channel makes the whole retrieval fail, because a
  "hybrid" answer that silently lost one of its channels is not a hybrid
  answer.  A caller that prefers a partial answer over an error can ask for
  it explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from retrieval.contracts.capabilities import RetrievalCapabilities
from retrieval.contracts.exceptions import RetrievalConfigError
from retrieval.contracts.models import (
    RetrievedItem,
    RetrievalDiagnostics,
    RetrievalQuery,
    RetrievalResponse,
)
from retrieval.dense.qwen.channel import QwenDenseRetriever
from retrieval.fusion.rrf.rrf import DEFAULT_K, RankedList, fuse
from retrieval.lexical.bm25.sqlite_fts5.retriever import BM25Retriever
from retrieval.postprocessing.adjacent_merge.merger import (
    AdjacentMergeConfig,
    AdjacentMerger,
)
from retrieval.postprocessing.deduplication.deduplicator import deduplicate
from vector_stores.contracts.filters import MetadataFilter

#: The policy when a channel fails.
STRICT = "strict"
"""The retrieval fails when a channel fails."""

PARTIAL = "partial"
"""The retrieval continues with the channels that succeeded."""


@dataclass(frozen=True)
class HybridConfig:
    """Settings of :class:`HybridRetrievalStrategy`.

    Args:
        rrf_k: the damping constant of the reciprocal rank function.
        channel_weights: the weight of each channel in the fused score.
        failure_policy: ``"strict"`` or ``"partial"``.
        deduplicate: whether the fused items are deduplicated.
        adjacent_merge: the settings of the adjacent-chunk merging.
    """

    rrf_k: int = DEFAULT_K
    channel_weights: dict[str, float] = field(default_factory=dict)
    failure_policy: str = STRICT
    deduplicate: bool = True
    adjacent_merge: AdjacentMergeConfig = field(default_factory=AdjacentMergeConfig)

    def __post_init__(self) -> None:
        if self.rrf_k < 1:
            raise RetrievalConfigError(
                f"rrf_k must be >= 1, got {self.rrf_k}"
            )
        if self.failure_policy not in (STRICT, PARTIAL):
            raise RetrievalConfigError(
                f"failure_policy must be {STRICT!r} or {PARTIAL!r}, got "
                f"{self.failure_policy!r}"
            )
        for channel, weight in self.channel_weights.items():
            if weight < 0:
                raise RetrievalConfigError(
                    f"the weight of the channel {channel!r} must be >= 0, got {weight}"
                )


class ChannelFailure(RuntimeError):
    """A retrieval channel failed.

    The failure is always reported, whatever the policy: a caller that asks
    for a partial answer must still be able to see which channel was lost.
    """

    def __init__(self, channel: str, cause: BaseException) -> None:
        self.channel = channel
        self.cause = cause
        super().__init__(f"the {channel!r} channel failed: {cause}")


class HybridRetrievalStrategy:
    """Dense + lexical retrieval, fused with the reciprocal rank function.

    Args:
        dense: the dense channel.
        lexical: the lexical channel.
        config: the strategy settings.
    """

    name = "hybrid"

    def __init__(
        self,
        dense: QwenDenseRetriever,
        lexical: BM25Retriever,
        config: HybridConfig | None = None,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.config = config or HybridConfig()

    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> RetrievalCapabilities:
        """Return the capabilities of the hybrid strategy."""
        return RetrievalCapabilities(
            dense=True,
            lexical=True,
            fusion=True,
            metadata_filtering=True,
            filter_ops=self.dense.capabilities.filter_ops,
            postprocessing=frozenset(
                name
                for name, enabled in (
                    ("deduplication", self.config.deduplicate),
                    ("adjacent_merge", self.config.adjacent_merge.enabled),
                )
                if enabled
            ),
            details={
                "channels": [self.dense.name, self.lexical.name],
                "rrf_k": self.config.rrf_k,
                "failure_policy": self.config.failure_policy,
            },
        )

    # ------------------------------------------------------------------
    def retrieve(self, query: RetrievalQuery) -> RetrievalResponse:
        """Retrieve the passages that answer ``query``."""
        if not isinstance(query, RetrievalQuery):
            raise RetrievalConfigError(
                f"the query must be a RetrievalQuery, got {type(query).__name__}"
            )
        if not query.text.strip():
            raise RetrievalConfigError("the query text must not be empty")

        dense_items, dense_failure = self._run_dense(query)
        lexical_items, lexical_failure = self._run_lexical(query)

        failures: dict[str, str] = {}
        if dense_failure is not None:
            failures[self.dense.name] = f"{type(dense_failure).__name__}: {dense_failure}"
        if lexical_failure is not None:
            failures[self.lexical.name] = (
                f"{type(lexical_failure).__name__}: {lexical_failure}"
            )
        if failures and self.config.failure_policy == STRICT:
            first = dense_failure or lexical_failure
            raise ChannelFailure(
                self.dense.name if dense_failure is not None else self.lexical.name,
                first,
            )

        lists = [
            RankedList(
                channel=self.dense.name,
                items=dense_items,
                weight=self.config.channel_weights.get(self.dense.name, 1.0),
            ),
            RankedList(
                channel=self.lexical.name,
                items=lexical_items,
                weight=self.config.channel_weights.get(self.lexical.name, 1.0),
            ),
        ]
        fused = fuse(lists, k=self.config.rrf_k, strategy=self.name)

        if self.config.deduplicate:
            fused = deduplicate(fused)

        merge_report = None
        if self.config.adjacent_merge.enabled:
            merger = AdjacentMerger(config=self.config.adjacent_merge)
            fused, merge_report = merger.merge(fused)

        fused = fused[: query.limit]

        notes = [f"{channel}: {reason}" for channel, reason in failures.items()]
        if merge_report is not None:
            notes.extend(merge_report.notes)

        diagnostics = RetrievalDiagnostics(
            candidates_searched=len(dense_items) + len(lexical_items),
            channel_candidates={
                self.dense.name: len(dense_items),
                self.lexical.name: len(lexical_items),
            },
            channel_failures=failures,
            notes=tuple(notes),
        )
        return RetrievalResponse(
            results=list(fused),
            query=query,
            strategy=self.name,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    def _run_dense(
        self, query: RetrievalQuery
    ) -> tuple[list[RetrievedItem], BaseException | None]:
        """Run the dense channel, capturing its failure."""
        try:
            response = self.dense.retrieve(query)
            return list(response.results), None
        except Exception as exc:
            return [], exc

    def _run_lexical(
        self, query: RetrievalQuery
    ) -> tuple[list[RetrievedItem], BaseException | None]:
        """Run the lexical channel, capturing its failure."""
        try:
            response = self.lexical.retrieve(query)
            return list(response.results), None
        except Exception as exc:
            return [], exc
