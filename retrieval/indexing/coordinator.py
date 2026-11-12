"""Coordination of the dense and the lexical index.

The hybrid engine needs the *same* chunks in two places: the vector store,
for the dense channel, and the lexical index, for the BM25 channel.  The
coordinator writes to both and reports exactly what happened.

There is no distributed transaction between the two storage systems.  The
coordinator therefore never reports a success it cannot back up:

* every operation returns a report that names the backend that succeeded
  and the backend that failed;
* a failure in one backend does not roll back the other one, it is
  *reported*, and the failed side can be reconciled afterwards with
  :meth:`IndexingCoordinator.reconcile_lexical`.

The coordinator never loads an embedding model.  The caller either supplies
the vectors or injects an embedding model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chunkers.contracts.models import Chunk
from embeddings.contracts.models import EmbeddingInput
from embeddings.contracts.protocols import EmbeddingModel
from retrieval.contracts.exceptions import RetrievalConfigError
from retrieval.lexical.bm25.sqlite_fts5.index import LexicalRecord, SQLiteFTS5Index
from vector_stores.contracts.models import (
    SpaceVectors,
    StoredRecord,
    VectorSpaceConfig,
)
from vector_stores.contracts.protocols import VectorStore


@dataclass
class IndexingReport:
    """What an indexing operation did.

    Attributes:
        operation: the name of the operation.
        vector_store_written: the number of records written to the vector
            store.
        lexical_written: the number of records written to the lexical index.
        vector_store_error: the error of the vector store, when it failed.
        lexical_error: the error of the lexical index, when it failed.
        chunk_ids: the ids of the chunks the operation touched.
    """

    operation: str
    vector_store_written: int = 0
    lexical_written: int = 0
    vector_store_error: str | None = None
    lexical_error: str | None = None
    chunk_ids: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        """Return whether every backend was updated."""
        return self.vector_store_error is None and self.lexical_error is None

    @property
    def partial(self) -> bool:
        """Return whether exactly one of the two backends was updated."""
        return not self.succeeded and (
            self.vector_store_error is None or self.lexical_error is None
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the report."""
        return {
            "operation": self.operation,
            "succeeded": self.succeeded,
            "partial": self.partial,
            "vector_store_written": self.vector_store_written,
            "lexical_written": self.lexical_written,
            "vector_store_error": self.vector_store_error,
            "lexical_error": self.lexical_error,
            "chunk_ids": list(self.chunk_ids),
        }


class IndexingCoordinator:
    """Writes chunks into the dense and the lexical index.

    Args:
        vector_store: the vector store of the dense channel.
        lexical_index: the lexical index of the BM25 channel.
        collection: the collection of the vector store.
        vector_space: the named dense vector space of the collection.
        embedding_model: the optional embedding model, used when the caller
            does not supply the vectors.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        lexical_index: SQLiteFTS5Index,
        *,
        collection: str,
        vector_space: str = "dense",
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.collection = collection
        self.vector_space = vector_space
        self.embedding_model = embedding_model

    # ------------------------------------------------------------------
    def upsert(
        self,
        chunks: Sequence[Chunk],
        *,
        vectors: Sequence[Sequence[float]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IndexingReport:
        """Write ``chunks`` into both indexes.

        Args:
            chunks: the chunks to write.
            vectors: the precomputed dense vectors.  When they are not
                given, the injected embedding model computes them.
            metadata: the metadata stored next to every chunk.  It is the
                same metadata in both indexes, so a filter means the same
                thing in the dense and in the lexical channel.

        Returns:
            A report of what each backend did.
        """
        if not chunks:
            return IndexingReport(operation="upsert")
        chunk_ids = tuple(chunk.id for chunk in chunks)

        if vectors is None:
            vectors = self._embed(chunks)
        if len(vectors) != len(chunks):
            raise RetrievalConfigError(
                f"got {len(vectors)} vectors for {len(chunks)} chunks"
            )

        report = IndexingReport(operation="upsert", chunk_ids=chunk_ids)

        records = self._records(chunks, vectors, metadata)
        try:
            report.vector_store_written = self.vector_store.upsert(
                self.collection, records
            )
        except Exception as exc:
            report.vector_store_error = f"{type(exc).__name__}: {exc}"

        lexical_records = [
            LexicalRecord.from_chunk(chunk, metadata=metadata)
            for chunk in chunks
        ]
        try:
            report.lexical_written = self.lexical_index.upsert(lexical_records)
        except Exception as exc:
            report.lexical_error = f"{type(exc).__name__}: {exc}"

        return report

    def delete(self, chunk_ids: Sequence[str]) -> IndexingReport:
        """Delete the given chunks from both indexes."""
        report = IndexingReport(
            operation="delete", chunk_ids=tuple(chunk_ids)
        )
        if not chunk_ids:
            return report
        try:
            self.vector_store.delete(self.collection, list(chunk_ids))
            report.vector_store_written = len(chunk_ids)
        except Exception as exc:
            report.vector_store_error = f"{type(exc).__name__}: {exc}"
        try:
            report.lexical_written = self.lexical_index.delete(list(chunk_ids))
        except Exception as exc:
            report.lexical_error = f"{type(exc).__name__}: {exc}"
        return report

    def delete_documents(self, doc_ids: Sequence[str]) -> IndexingReport:
        """Delete every chunk of the given documents from both indexes."""
        report = IndexingReport(operation="delete_documents")
        if not doc_ids:
            return report
        try:
            report.vector_store_written = self.vector_store.delete_by_document(
                self.collection, doc_ids[0]
            )
            for doc_id in doc_ids[1:]:
                report.vector_store_written += self.vector_store.delete_by_document(
                    self.collection, doc_id
                )
        except Exception as exc:
            report.vector_store_error = f"{type(exc).__name__}: {exc}"
        try:
            report.lexical_written = self.lexical_index.delete_documents(
                list(doc_ids)
            )
        except Exception as exc:
            report.lexical_error = f"{type(exc).__name__}: {exc}"
        return report

    def reconcile_lexical(
        self, chunks: Sequence[Chunk], *, metadata: dict[str, Any] | None = None
    ) -> int:
        """Write ``chunks`` into the lexical index again.

        This is the safe retry of a failed lexical write: the vector store
        is left alone, so a half-finished upsert can be completed without
        touching the dense index.
        """
        records = [
            LexicalRecord.from_chunk(chunk, metadata=metadata) for chunk in chunks
        ]
        return self.lexical_index.upsert(records)

    # ------------------------------------------------------------------
    def _embed(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed ``chunks`` with the injected model."""
        if self.embedding_model is None:
            raise RetrievalConfigError(
                "no vectors were supplied and no embedding model was injected; "
                "the coordinator never loads a model itself"
            )
        inputs = [
            EmbeddingInput(text=chunk.text, doc_id=chunk.doc_id, chunk_id=chunk.id)
            for chunk in chunks
        ]
        results = self.embedding_model.embed_documents(inputs)
        if len(results) != len(inputs):
            raise RetrievalConfigError(
                f"the embedding model returned {len(results)} results for "
                f"{len(inputs)} inputs"
            )
        vectors: list[list[float]] = []
        for result in results:
            vector = result.dense_vector
            if not vector:
                raise RetrievalConfigError(
                    "the embedding model returned no dense vector for a chunk"
                )
            vectors.append(list(vector))
        return vectors

    def _records(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        metadata: dict[str, Any] | None,
    ) -> list[StoredRecord]:
        """Build the vector-store records of ``chunks``."""
        descriptor = self.embedding_model.get_embedding_space() if self.embedding_model else None
        if descriptor is None:
            # Without a model the space descriptor is taken from the
            # collection, so the records stay compatible with it.
            descriptor = self._collection_space().embedding_space
        space = self.vector_space
        records: list[StoredRecord] = []
        for chunk, vector in zip(chunks, vectors):
            chunk_metadata = dict(chunk.metadata)
            if metadata:
                chunk_metadata.update(metadata)
            records.append(
                StoredRecord(
                    chunk=chunk,
                    vectors={
                        space: SpaceVectors(
                            space=space,
                            embedding_space=descriptor,
                            dense=list(vector),
                        )
                    },
                    metadata=dict(metadata or {}),
                )
            )
        return records

    def _collection_space(self) -> VectorSpaceConfig:
        """Return the configuration of the targeted vector space."""
        info = self.vector_store.describe_collection(self.collection)
        try:
            return info.vector_spaces[self.vector_space]
        except KeyError as exc:
            raise RetrievalConfigError(
                f"collection {self.collection!r} has no vector space "
                f"{self.vector_space!r}"
            ) from exc
