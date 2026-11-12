# Qdrant Adapter

The first vector-store backend of the Prome ecosystem.  It accepts
**precomputed embeddings only**: it never embeds text and never loads a
model.

## Installation

```bash
pip install 'prome-rag-plugins[qdrant]'
# or
pip install 'qdrant-client>=1.12'
```

`qdrant-client` is an optional dependency and is imported lazily: importing
`vector_stores.qdrant` works without it, and the error
(`VectorStoreDependencyError`) is raised only when the backend is actually
used.

## Deployment modes

```python
from vector_stores.qdrant import QdrantVectorStore

store = QdrantVectorStore.memory()          # in-process, not persistent
store = QdrantVectorStore.local("/tmp/qdrant")   # embedded, on disk
store = QdrantVectorStore.server("http://localhost:6333")   # running server
store = QdrantVectorStore(client=my_qdrant_client)          # bring your own
```

Docker, Qdrant Cloud and any other hosted service are never required or
used.

## Quick start

```python
from vector_stores.contracts import CollectionConfig, SearchRequest, VectorSpaceConfig
from vector_stores.qdrant import QdrantVectorStore

store = QdrantVectorStore.local("/tmp/qdrant")
store.create_collection(config)              # a CollectionConfig
store.upsert("docs", [record, record])        # StoredRecord objects
hits = store.search(SearchRequest(collection="docs", space="dense", dense=vector))
store.close()
```

## How the chunk id is stored

A Qdrant point id is an unsigned integer or a UUID, while a Prome chunk id
is a free-form string (`{digest}-{strategy}-{index}`, e.g.
`a1b2c3d4e5f6-fixed-0000`).  The adapter therefore maps the chunk id to a
**deterministic UUIDv5** (fixed namespace, SHA-1 based):

- the same chunk id always lands on the same point, in every process and on
  every machine;
- the mapping is collision resistant for the number of chunks a store
  realistically holds;
- the **original chunk id is never replaced**: it is stored in the payload
  and returned by every read operation.

## Payload layout

```text
{
  "chunk_id": "a1b2c3d4e5f6-fixed-0000",   # the public chunk id
  "doc_id": "report.pdf",                   # the public document id
  "tenant": "acme",                         # user metadata, filterable
  "_prome": {                               # the contract fields
      "text": "...", "index": 0, "count": 8,
      "length": 512, "length_unit": "characters",
      "heading": "Introduction",
      "block_refs": [...], "chunk_metadata": {...},
      "provenance": {...},
      "embedding_spaces": {"dense": {...}}
  }
}
```

User metadata stays at the top level so it can be filtered and indexed
directly (`filter(eq("tenant", "acme"))`); the contract fields live under
the reserved `_prome` key.  A metadata key that collides with a reserved
name is rejected.

## Vectors

| Space kind | Qdrant representation |
|---|---|
| dense | named vector `{"dense": [...]}` |
| sparse | named sparse vector `{"sparse": SparseVector(indices, values)}` |
| multi-vector | named multivector `{"late": [[...], [...]]}` with the `max_sim` comparator |

Sparse vectors keep their indices and weights (they are never densified) and
multi-vectors keep the full token matrix (they are never averaged).

## Indexes

| Kind | How it is configured |
|---|---|
| exact | `params={"exact": True}` on the search request |
| HNSW | `HnswIndexConfig(m=..., ef_construct=..., full_scan_threshold=...)` in the vector space; `params={"hnsw_ef": ...}` at query time |
| sparse inverted | built by Qdrant for every sparse vector; `SparseIndexConfig(full_scan_threshold=...)` |
| payload | `PayloadIndexSpec(field, field_type)` in the collection config, or `create_index()` afterwards |
| multi-vector | built by Qdrant with the `max_sim` comparator |

The default HNSW parameters are conservative (`m=16`, `ef_construct=100`).
Low-level settings are not exposed through the common API; backend-specific
options live in `backend_options`.

## Filters

The portable filter language is mapped onto Qdrant conditions:

| Portable | Qdrant |
|---|---|
| `eq` / `ne` | `MatchValue` (floats use an exact `Range`) |
| `in` / `not-in` | `MatchAny` |
| `gt` / `gte` / `lt` / `lte` | `Range` |
| `exists` | `is_empty` |
| `and` / `or` / `not` | `must` / `should` / `must_not` |

## Limitations of the in-process modes

The `memory` and `local` modes of the Qdrant client are **not** the server:

- every query is a brute-force scan - `exact` and `hnsw_ef` have no effect,
  and the "approximate" path is exact as well;
- payload indexes are accepted but do not speed up filtering (the client
  prints a warning);
- the search latency measured in those modes is the latency of a full scan,
  not of HNSW.

Use the `server` mode for the real HNSW index and for payload indexes.

## Error handling

| Exception | Cause |
|---|---|
| `VectorStoreDependencyError` | `qdrant-client` is not installed |
| `CollectionNotFoundError` | the collection does not exist |
| `CollectionExistsError` | the collection already exists |
| `VectorStoreConfigError` | invalid collection configuration or search request |
| `DimensionMismatchError` | the vector width differs from the declared one |
| `EmbeddingSpaceMismatchError` | the vector belongs to another model / revision / normalization |
| `InvalidRecordError` | malformed record, unknown vector space, reserved metadata key |
| `UnsupportedFilterOperationError` | the filter operation cannot be expressed |
| `UnsupportedIndexOperationError` | the index kind cannot be created after the collection exists |
| `BackendConnectionError` | the backend could not be reached |

## License

MIT
