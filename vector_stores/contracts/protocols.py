"""The vector-store protocol every backend implements.

The contract is deliberately small and backend independent: a store manages
collections of chunk records and answers vector queries.  It never embeds
text, never loads a model and never talks to a hosted service.

Implementations must:

- validate the vector configuration before accepting data, and refuse vectors
  that belong to a different model or embedding space;
- preserve the public chunk id and document id of every record;
- report unsupported capabilities explicitly instead of degrading silently;
- raise :class:`~vector_stores.contracts.exceptions.VectorStoreError`
  subclasses instead of leaking backend exceptions.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from vector_stores.contracts.capabilities import StoreCapabilities
from vector_stores.contracts.filters import MetadataFilter
from vector_stores.contracts.indexes import IndexConfig, IndexDescription
from vector_stores.contracts.models import (
    CollectionConfig,
    CollectionInfo,
    ScrollPage,
    SearchRequest,
    SearchResult,
    StoredRecord,
)


@runtime_checkable
class VectorStore(Protocol):
    """Interface every Prome vector store implements."""

    name: str
    """Backend identifier."""

    @property
    def capabilities(self) -> StoreCapabilities:
        """Return the capabilities of this backend."""

        ...

    def create_collection(
        self, config: CollectionConfig, *, exist_ok: bool = False
    ) -> CollectionInfo:
        """Create a collection from ``config``.

        Raises:
            CollectionExistsError: if the collection exists and ``exist_ok`` is False.
        """

        ...

    def describe_collection(self, name: str) -> CollectionInfo:
        """Describe an existing collection.

        Raises:
            CollectionNotFoundError: if the collection does not exist.
        """

        ...

    def list_collections(self) -> list[CollectionInfo]:
        """List the collections of the backend."""

        ...

    def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True when a collection was deleted."""

        ...

    def upsert(self, collection: str, records: Sequence[StoredRecord]) -> int:
        """Insert or replace ``records``. Returns the number of records written."""

        ...

    def upsert_one(self, collection: str, record: StoredRecord) -> int:
        """Insert or replace a single record."""

        ...

    def get(self, collection: str, ids: Sequence[str]) -> list[StoredRecord]:
        """Fetch records by their public chunk ids.

        Ids that do not exist are skipped (the result may be shorter than the
        request).
        """

        ...

    def delete(self, collection: str, ids: Sequence[str]) -> int:
        """Delete records by their public chunk ids. Returns the number deleted."""

        ...

    def delete_by_document(self, collection: str, doc_id: str) -> int:
        """Delete every record of a document. Returns the number deleted."""

        ...

    def delete_by_filter(self, collection: str, filter: MetadataFilter) -> int:
        """Delete every record matching ``filter``. Returns the number deleted."""

        ...

    def count(self, collection: str, filter: MetadataFilter | None = None) -> int:
        """Count the records of a collection, optionally filtered."""

        ...

    def scroll(
        self,
        collection: str,
        *,
        limit: int = 100,
        offset: int | str | None = None,
        filter: MetadataFilter | None = None,
        with_vectors: bool = False,
    ) -> ScrollPage:
        """Return one page of records.

        ``offset`` is an opaque cursor: pass back the ``next_offset`` of the
        previous page unchanged.
        """

        ...

    def search(self, request: SearchRequest) -> list[SearchResult]:
        """Run a vector search and return the hits, best first."""

        ...

    def create_index(self, collection: str, index: IndexConfig) -> IndexDescription:
        """Create an index on an existing collection."""

        ...

    def delete_index(self, collection: str, name: str) -> bool:
        """Delete an index by name. Returns True when an index was deleted."""

        ...

    def list_indexes(self, collection: str) -> list[IndexDescription]:
        """List the indexes of a collection."""

        ...

    def close(self) -> None:
        """Release the backend resources held by this store."""

        ...


def iter_records(
    store: VectorStore,
    collection: str,
    *,
    page_size: int = 100,
    filter: MetadataFilter | None = None,
    with_vectors: bool = False,
) -> Iterator[StoredRecord]:
    """Iterate over every record of a collection.

    This is a thin helper over :meth:`VectorStore.scroll` so that backends
    only have to implement pagination, not iteration.
    """
    offset: int | None = None
    while True:
        page = store.scroll(
            collection,
            limit=page_size,
            offset=offset,
            filter=filter,
            with_vectors=with_vectors,
        )
        yield from page.records
        if page.next_offset is None:
            return
        offset = page.next_offset


def collect_records(
    store: VectorStore,
    collection: str,
    *,
    page_size: int = 100,
    filter: MetadataFilter | None = None,
    with_vectors: bool = False,
) -> list[StoredRecord]:
    """Return every record of a collection as a list."""
    return list(
        iter_records(
            store,
            collection,
            page_size=page_size,
            filter=filter,
            with_vectors=with_vectors,
        )
    )
