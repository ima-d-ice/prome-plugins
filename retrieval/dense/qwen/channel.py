"""The dense retrieval channel, bound to the Qwen3-Embedding-0.6B encoder.

The encoder is always **injected**.  Nothing here loads a model, downloads
weights or talks to a hosted service; constructing the retriever is cheap and
the model is only touched when a query is embedded.
"""

from __future__ import annotations

from embeddings.contracts.models import EmbeddingSpaceDescriptor
from embeddings.contracts.protocols import EmbeddingModel
from retrieval.contracts.capabilities import RetrievalCapabilities
from retrieval.contracts.exceptions import UnsupportedRetrievalCapabilityError
from retrieval.dense.retriever import DenseRetriever
from vector_stores.contracts.protocols import VectorStore

#: The identifier of the production encoder.
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


class QwenDenseRetriever(DenseRetriever):
    """Dense retrieval with the Qwen3-Embedding-0.6B encoder.

    Args:
        store: the vector store to search.
        embedding_model: the injected Qwen adapter (never loaded here).
        collection: name of the collection to search.
        space: name of the vector space to search.
        expected_model_id: the model identifier the collection is expected to
            have been built with.  ``None`` accepts any model.
    """

    name = "dense-qwen"

    def __init__(
        self,
        store: VectorStore,
        embedding_model: EmbeddingModel,
        *,
        collection: str,
        space: str = "dense",
        expected_model_id: str | None = QWEN_MODEL_ID,
    ) -> None:
        super().__init__(
            store,
            embedding_model,
            collection=collection,
            space=space,
        )
        self.expected_model_id = expected_model_id

    @property
    def capabilities(self) -> RetrievalCapabilities:
        """Return the capabilities of the dense channel."""
        store_capabilities = self.store.capabilities
        return RetrievalCapabilities(
            dense=True,
            lexical=False,
            fusion=False,
            metadata_filtering=store_capabilities.dense_search,
            filter_ops=store_capabilities.filter_ops,
            details={
                "backend": store_capabilities.backend,
                "collection": self.collection,
                "vector_space": self.space,
                "embedding_model": self.expected_model_id or "any",
            },
        )

    def check_embedding_space(self) -> None:
        """Verify the collection was built for the expected encoder.

        Raises:
            UnsupportedRetrievalCapabilityError: when the collection carries
                another model or the injected model does not match the
                collection's embedding space.
        """
        space = self.space_config()
        if (
            self.expected_model_id
            and space.embedding_space.model_id != self.expected_model_id
        ):
            raise UnsupportedRetrievalCapabilityError(
                f"vector space {self.space!r} of collection {self.collection!r} was "
                f"built for {space.embedding_space.model_id!r}, but the dense "
                f"channel expects {self.expected_model_id!r}"
            )
        super().check_embedding_space()

    def expected_embedding_space(self) -> EmbeddingSpaceDescriptor:
        """Return the embedding space the collection is expected to declare."""
        return self.embedding_model.get_embedding_space()


def qwen_dense_retriever(
    store: VectorStore,
    embedding_model: EmbeddingModel,
    *,
    collection: str,
    space: str = "dense",
) -> QwenDenseRetriever:
    """Build the dense channel for ``collection``.

    A convenience wrapper; the retriever can also be constructed directly.
    """
    return QwenDenseRetriever(
        store,
        embedding_model,
        collection=collection,
        space=space,
    )
