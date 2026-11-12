"""Dense retrieval over a vector store.

The dense retriever is the reusable building block of every dense-backed
retrieval strategy: it embeds a query with an **injected** embedding model and
runs a filtered vector search against a collection.

It never loads a model itself and never downloads weights.  The caller injects
an object that satisfies
:class:`embeddings.contracts.protocols.EmbeddingModel`; in production that is
the Qwen3-Embedding-0.6B adapter.

Two properties matter for the rest of the ecosystem:

* **capabilities are checked, not assumed.**  Before a filtered search the
  retriever verifies that the backend can express the requested filter
  operations and, on backends with a fixed schema, that the filtered fields
  are declared.  It raises
  :class:`~retrieval.contracts.exceptions.UnsupportedRetrievalCapabilityError`
  instead of silently searching the whole collection.
* **the embedding space is checked.**  The query vector must live in the same
  embedding space as the collection's vector space, otherwise the similarity
  numbers are meaningless.
"""

from __future__ import annotations

from collections.abc import Sequence

from embeddings.contracts.models import EmbeddingInput, RepresentationType
from embeddings.contracts.protocols import EmbeddingModel
from retrieval.contracts.exceptions import (
    RetrievalConfigError,
    UnsupportedRetrievalCapabilityError,
)
from vector_stores.contracts.capabilities import StoreCapabilities
from vector_stores.contracts.filters import FIELD_OPS, FilterOp, MetadataFilter
from vector_stores.contracts.indexes import IndexKind
from vector_stores.contracts.models import (
    CollectionInfo,
    ScoreDirection,
    SearchRequest,
    SearchResult,
    VectorSpaceConfig,
)
from vector_stores.contracts.protocols import VectorStore
from vector_stores.contracts.validation import (
    embedding_space_mismatch_reasons,
)


def required_filter_ops(node: MetadataFilter | None) -> set[FilterOp]:
    """Return the *field* operations a filter tree needs.

    Only field operations are collected: the logical operators (``and``,
    ``or``, ``not``) are structural and every backend implements them, while
    the field operations are what
    :attr:`~vector_stores.contracts.capabilities.StoreCapabilities.filter_ops`
    actually advertises.
    """
    if node is None:
        return set()
    found: set[FilterOp] = set()
    if node.op in FIELD_OPS:
        found.add(node.op)
    for child in node.children:
        found |= required_filter_ops(child)
    return found


def filter_fields(node: MetadataFilter | None) -> set[str]:
    """Return the metadata fields a filter tree reads."""
    if node is None:
        return set()
    found: set[str] = set()
    if node.field:
        found.add(node.field)
    for child in node.children:
        found |= filter_fields(child)
    return found


class DenseRetriever:
    """Dense search over one named vector space of one collection.

    Args:
        store: the vector store to search.
        embedding_model: the injected embedding model (never loaded here).
        collection: name of the collection to search.
        space: name of the vector space to search.
    """

    def __init__(
        self,
        store: VectorStore,
        embedding_model: EmbeddingModel,
        *,
        collection: str,
        space: str = "dense",
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.collection = collection
        self.space = space
        self._collection_info: CollectionInfo | None = None

    # ------------------------------------------------------------------
    # capabilities
    # ------------------------------------------------------------------
    @property
    def capabilities(self) -> StoreCapabilities:
        """Return the capabilities of the backing store."""
        return self.store.capabilities

    def collection_info(self) -> CollectionInfo:
        """Describe the collection, caching the answer."""
        if self._collection_info is None:
            self._collection_info = self.store.describe_collection(self.collection)
        return self._collection_info

    def space_config(self) -> VectorSpaceConfig:
        """Return the configuration of the searched vector space."""
        spaces = self.collection_info().vector_spaces
        try:
            return spaces[self.space]
        except KeyError as exc:
            raise RetrievalConfigError(
                f"collection {self.collection!r} has no vector space {self.space!r}; "
                f"known spaces: {sorted(spaces)}"
            ) from exc

    def check_filter_support(self, filter: MetadataFilter | None) -> None:
        """Verify the backend can express ``filter``.

        Raises:
            UnsupportedRetrievalCapabilityError: if the backend lacks a filter
                operation the tree needs, or if the backend requires the
                filtered fields to be declared and they are not.
        """
        if filter is None:
            return
        capabilities = self.capabilities
        missing = required_filter_ops(filter) - set(capabilities.filter_ops)
        if missing:
            raise UnsupportedRetrievalCapabilityError(
                f"the {capabilities.backend!r} backend cannot express the filter "
                f"operation(s) {sorted(op.value for op in missing)}; it supports "
                f"{sorted(op.value for op in capabilities.filter_ops)}"
            )
        if not capabilities.filter_requires_declared_fields:
            return
        declared = {
            index.target
            for index in self.collection_info().indexes
            if index.kind is IndexKind.PAYLOAD
        }
        undeclared = filter_fields(filter) - declared
        if undeclared:
            raise UnsupportedRetrievalCapabilityError(
                f"the {capabilities.backend!r} backend can only filter on the "
                f"fields declared in the collection configuration; "
                f"{sorted(undeclared)} are not declared "
                f"(declared: {sorted(declared)})"
            )

    # ------------------------------------------------------------------
    # embedding
    # ------------------------------------------------------------------
    def embed_query(self, query: str) -> list[float]:
        """Embed ``query`` with the injected model.

        The query is embedded with the model's *query* encoding, not its
        document encoding, and the result is checked against the collection's
        vector space.
        """
        if not isinstance(query, str) or not query.strip():
            raise RetrievalConfigError("the query must be a non-empty string")
        space = self.space_config()
        if space.vector_type is not RepresentationType.DENSE:
            raise RetrievalConfigError(
                f"vector space {self.space!r} is {space.vector_type.value}; the dense "
                "retriever only searches dense spaces"
            )
        # The check happens *before* the inference, so a model of the wrong
        # embedding space is reported without paying for an embedding.
        self.check_embedding_space()
        result = self.embedding_model.embed_query(EmbeddingInput(text=query))
        vector = result.dense_vector
        if not vector:
            raise RetrievalConfigError(
                f"the embedding model {getattr(self.embedding_model, 'name', '?')!r} "
                "returned no dense vector for the query"
            )
        if len(vector) != space.dimensions:
            raise RetrievalConfigError(
                f"the query vector has {len(vector)} dimensions but vector space "
                f"{self.space!r} expects {space.dimensions}"
            )
        return list(vector)

    def check_embedding_space(self) -> None:
        """Verify the injected model and the collection share an embedding space.

        Raises:
            UnsupportedRetrievalCapabilityError: when the model and the
                collection were built for different embedding spaces (a
                different model, revision, width or normalization).
        """
        space = self.space_config()
        model_space = self.embedding_model.get_embedding_space()
        reasons = embedding_space_mismatch_reasons(space.embedding_space, model_space)
        if reasons:
            raise UnsupportedRetrievalCapabilityError(
                f"the embedding model {model_space.model_id!r} does not match the "
                f"embedding space of vector space {self.space!r}: "
                f"{'; '.join(reasons)}"
            )

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def search(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        filter: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Run a dense search and return the hits, best first.

        The score threshold is applied by this layer, not by the backend: the
        vector-store contract carries ``score_threshold`` but only some
        backends implement it, and the meaning of a threshold depends on the
        metric's score direction.
        """
        if limit < 1:
            raise RetrievalConfigError(f"the search limit must be >= 1, got {limit}")
        self.check_filter_support(filter)
        request = SearchRequest(
            collection=self.collection,
            space=self.space,
            limit=limit,
            dense=list(query_vector),
            filter=filter,
            with_payload=True,
        )
        results = self.store.search(request)
        if score_threshold is not None:
            results = [
                result
                for result in results
                if self._passes_threshold(result, score_threshold)
            ]
        return results

    @staticmethod
    def _passes_threshold(result: SearchResult, threshold: float) -> bool:
        """Return whether ``result`` passes ``threshold``.

        A cosine or dot-product score is better when it is larger; a
        euclidean distance is better when it is smaller.  The direction comes
        from the backend, so the same threshold means the same thing
        everywhere.
        """
        direction = result.score_direction
        if direction is ScoreDirection.LOWER_IS_BETTER:
            return result.score <= threshold
        return result.score >= threshold
