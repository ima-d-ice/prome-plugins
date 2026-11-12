"""The hierarchical (coarse-to-fine) retrieval strategy.

The strategy answers a query in two steps:

1. the **coarse** stage selects the parent sections that look relevant, and
2. the **fine** stage retrieves the child chunks *inside those sections* only.

The two steps are sequential by design: the fine stage needs the ids the
coarse stage selected, so they cannot run in parallel.  The query is embedded
**once** and the same vector is used for both searches.

The result keeps the two levels apart: the matched child is the evidence, the
parent section and the optional neighbouring children are context.
"""

from __future__ import annotations

from dataclasses import dataclass

from chunkers.contracts.models import Chunk
from embeddings.contracts.protocols import EmbeddingModel
from retrieval.contracts.exceptions import RetrievalConfigError
from retrieval.contracts.models import (
    ContextChunk,
    RetrievalDiagnostics,
    RetrievalResponse,
    RetrievalResult,
)
from retrieval.dense.retriever import DenseRetriever
from retrieval.postprocessing.context_expansion.expander import (
    ContextExpansionConfig,
    ContextExpander,
)
from retrieval.strategies.hierarchical.child import ChildStage, ChildStageResult
from retrieval.strategies.hierarchical.config import HierarchicalConfig
from retrieval.strategies.hierarchical.metadata import LEVEL_CHILD
from retrieval.strategies.hierarchical.parent import ParentStage, ParentStageResult
from vector_stores.contracts.filters import eq
from vector_stores.contracts.protocols import VectorStore


@dataclass(frozen=True)
class HierarchicalRetrievalStrategy:
    """Coarse-to-fine retrieval over a hierarchical (parent/child) collection.

    Args:
        retriever: the shared dense retriever (owns the store and the model).
        config: the strategy settings.
    """

    retriever: DenseRetriever
    config: HierarchicalConfig = HierarchicalConfig()

    name = "hierarchical"

    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        store: VectorStore,
        embedding_model: EmbeddingModel,
        *,
        collection: str,
        config: HierarchicalConfig | None = None,
    ) -> HierarchicalRetrievalStrategy:
        """Build the strategy from a store and an embedding model."""
        settings = config or HierarchicalConfig()
        retriever = DenseRetriever(
            store,
            embedding_model,
            collection=collection,
            space=settings.vector_space,
        )
        return cls(retriever=retriever, config=settings)

    # ------------------------------------------------------------------
    @property
    def parent_stage(self) -> ParentStage:
        """Return the coarse stage."""
        return ParentStage(
            self.retriever,
            top_k=self.config.parent_top_k,
            score_threshold=self.config.parent_score_threshold,
            filters=self.config.filters,
        )

    @property
    def child_stage(self) -> ChildStage:
        """Return the fine stage."""
        return ChildStage(
            self.retriever,
            top_k=self.config.child_top_k,
            score_threshold=self.config.child_score_threshold,
            filters=self.config.filters,
        )

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> RetrievalResponse:
        """Retrieve the passages that answer ``query``."""
        if not isinstance(query, str) or not query.strip():
            raise RetrievalConfigError("the query must be a non-empty string")

        # The query is embedded exactly once and the same vector is used by
        # both stages.
        query_vector = self.retriever.embed_query(query)

        parents = self.parent_stage.search(query_vector)
        parent_ids = [parent.chunk_id for parent in parents]
        notes: list[str] = []

        if not parent_ids:
            notes.append("the coarse stage selected no parent section")
            return RetrievalResponse(
                results=[],
                diagnostics=RetrievalDiagnostics(
                    candidates_searched=0,
                    parents_selected=0,
                    selected_parent_ids=(),
                    notes=tuple(notes),
                    _candidates_total=self._total_children,
                ),
            )

        children = self.child_stage.search(query_vector, parent_ids)
        notes.extend(self._integrity_notes(parents, children))

        results = self._build_results(parents, children, notes)

        return RetrievalResponse(
            results=results,
            diagnostics=RetrievalDiagnostics(
                candidates_searched=self._candidate_count(parents),
                parents_selected=len(parents),
                selected_parent_ids=tuple(parent_ids),
                notes=tuple(notes),
                _candidates_total=self._total_children,
            ),
        )

    @staticmethod
    def _candidate_count(parents: list[ParentStageResult]) -> int:
        """Return the number of child chunks the fine stage could consider.

        It is the sum of the child counts of the selected parents, which is
        the size of the candidate set the fine stage actually searched.
        """
        return sum(parent.child_count for parent in parents)

    # ------------------------------------------------------------------
    def _build_results(
        self,
        parents: list[ParentStageResult],
        children: list[ChildStageResult],
        notes: list[str],
    ) -> list[RetrievalResult]:
        """Turn the stage results into the standardized results."""
        parents_by_id = {parent.chunk_id: parent for parent in parents}
        expander = self._expander(parents, children, notes)
        expansion = ContextExpansionConfig(
            include_parent=self.config.include_parent_context,
            include_adjacent=self.config.expand_adjacent,
            max_adjacent=self.config.max_adjacent,
        )

        results: list[RetrievalResult] = []
        for rank, child in enumerate(children[: self.config.final_top_k], start=1):
            parent = parents_by_id.get(child.parent_id)
            context: list[ContextChunk] = []
            if self.config.include_parent_context and parent is not None:
                context.append(
                    ContextChunk(
                        chunk_id=parent.chunk_id,
                        text=parent.text,
                        role="parent",
                        doc_id=parent.doc_id,
                        metadata=dict(parent.metadata),
                        provenance=parent.provenance,
                    )
                )
            if self.config.expand_adjacent and parent is not None:
                expanded = expander.expand(child, config=expansion)
                context.extend(expanded.adjacent)
                notes.extend(expanded.notes)

            metadata = dict(child.metadata)
            metadata["parent_id"] = child.parent_id
            metadata["parent_rank"] = parent.rank if parent is not None else None
            metadata["parent_score"] = parent.score if parent is not None else None
            metadata["child_rank"] = child.rank
            metadata["child_score"] = child.score
            metadata["final_rank"] = rank
            if parent is not None:
                metadata["heading_path"] = list(parent.heading_path)

            results.append(
                RetrievalResult(
                    chunk_id=child.chunk_id,
                    doc_id=child.doc_id,
                    text=child.text,
                    score=child.score,
                    rank=rank,
                    strategy=self.name,
                    metadata=metadata,
                    provenance=child.provenance,
                    context=tuple(context),
                )
            )
        return results

    def _expander(
        self,
        parents: list[ParentStageResult],
        children: list[ChildStageResult],
        notes: list[str],
    ) -> ContextExpander:
        """Build the context expander for the selected parents.

        The neighbouring children are *fetched from the store*: they were not
        part of the fine-stage answer, and the expansion must not invent
        content.
        """
        parent_records: dict[str, Chunk] = {}
        parent_ids = [parent.chunk_id for parent in parents]
        if parent_ids:
            for record in self.retriever.store.get(
                self.retriever.collection, parent_ids
            ):
                parent_records[record.chunk.id] = record.chunk

        # The children of the selected parents, in the order the parent
        # declares them.
        child_records: dict[str, Chunk] = {}
        wanted: list[str] = []
        for parent in parents:
            wanted.extend(parent.child_ids)
        if wanted:
            for record in self.retriever.store.get(self.retriever.collection, wanted):
                child_records[record.chunk.id] = record.chunk

        # The matched children are added as well, so the expander can locate
        # them inside their parent's child list.
        for child in children:
            child_records.setdefault(
                child.chunk_id,
                Chunk(
                    id=child.chunk_id,
                    doc_id=child.doc_id,
                    text=child.text,
                    index=int(child.metadata.get("chunk_index", 0) or 0),
                    count=1,
                    length=len(child.text),
                    length_unit="characters",
                    metadata=dict(child.metadata),
                ),
            )
        return ContextExpander(parents=parent_records, children=child_records)

    @staticmethod
    def _integrity_notes(
        parents: list[ParentStageResult],
        children: list[ChildStageResult],
    ) -> list[str]:
        """Report the relationship problems found between the two stages."""
        notes: list[str] = []
        parent_ids = {parent.chunk_id for parent in parents}
        for child in children:
            if child.parent_id not in parent_ids:
                notes.append(
                    f"the child {child.chunk_id!r} was retrieved for a parent that "
                    "was not selected"
                )
        return notes

    def _total_children(self) -> int:
        """Return the number of child records in the collection."""
        try:
            return self.retriever.store.count(
                self.retriever.collection, eq("chunk_level", LEVEL_CHILD)
            )
        except Exception:
            return 0
