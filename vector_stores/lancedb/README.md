# LanceDB Adapter

LanceDB is an embedded, local vector database: the data lives in a
directory, no server is needed and no hosted service is ever used.  The
adapter accepts precomputed embeddings only - it never embeds text and never
loads a model.

## Installation

```bash
pip install 'prome-rag-plugins[lancedb]'
# or
pip install 'lancedb>=0.41'
```

`lancedb` is an optional dependency and is imported lazily: importing
`vector_stores.lancedb` works without it, and the error
(`VectorStoreDependencyError`) is raised only when the backend is used.

## Quick start

```python
from vector_stores.contracts import CollectionConfig, SearchRequest
from vector_stores.lancedb import LanceDBVectorStore

store = LanceDBVectorStore.local("/tmp/lancedb")   # persistent directory
store = LanceDBVectorStore.memory()                 # ephemeral directory
store.create_collection(config)
store.upsert("docs", records)
hits = store.search(
    SearchRequest(collection="docs", space="dense", dense=query_vector)
)
store.close()
```

A collection is a LanceDB **table**; the chunk id is the merge key, so the
public chunk id is never replaced by a generated identifier.

## Table layout

| column | content |
|---|---|
| `chunk_id` | the public chunk id (the primary key of the merge) |
| `doc_id` | the public document id |
| `v_<space>` | dense vectors (fixed size list of float32) |
| `s_<space>_indices` / `s_<space>_values` | sparse indices and weights (storage only) |
| `m_<space>` | multi-vector matrices (token level vectors) |
| `metadata` | struct with the **declared** filterable fields |
| `metadata_json` | the full metadata, lossless |
| `payload` | the contract fields (JSON) |

The declared collection configuration is stored in the `_prome_collections`
table, so the "foreign model" check still applies after a restart.

## Search

| Scenario | How |
|---|---|
| exact dense | `params={"exact": True}` (bypasses the vector index) |
| approximate dense | the native IVF index (`IVF_FLAT`, `IVF_PQ`, `IVF_HNSW_FLAT`, `IVF_HNSW_PQ`) |
| multi-vector | native late interaction: the distance is the sum over the query vectors of the distance to the best matching stored vector (max-sim style) |
| sparse | **not supported** - the vectors are stored but never searched |

The score is reported in the convention of the metric: LanceDB returns
distances (`1 - similarity` for cosine and dot, the squared euclidean
distance for `l2`), and the adapter converts them so that `metric` and
`score_direction` always agree.

## Indexes

| Kind | LanceDB |
|---|---|
| exact | no index (`params={"exact": True}`) |
| HNSW | `IVF_HNSW_FLAT` (IVF partitions + an HNSW graph inside each partition) |
| IVF | `IVF_FLAT`, `IVF_PQ`, `IVF_HNSW_PQ` |
| payload | scalar index (`BTREE`, `BITMAP`, `LABEL_LIST`) on a declared metadata field |

```python
store.create_index(
    "docs",
    IndexConfig(kind=IndexKind.HNSW, target="dense",
                build_params={"num_partitions": 16, "m": 16, "ef_construction": 100}),
)
```

Query-time parameters: `nprobes`, `refine_factor`, `exact`.

## Filters

The portable filter language is mapped onto the Lance SQL dialect:
`=` / `<>` / `IN` / `NOT IN`, the range operators, `IS NULL` / `IS NOT NULL`
and `AND` / `OR` / `NOT`.  String values are escaped, field names are
validated identifiers.

**Only the fields declared in the collection configuration can be filtered
on**: a LanceDB table has a fixed schema, so a filter on a field that does
not exist in it cannot be pushed down.  Such a filter raises
`UnsupportedFilterOperationError` instead of being ignored.

## Limitations

- No sparse vector search (LanceDB 0.41 has no sparse index and no sparse
  type).
- The metadata struct only holds the declared fields; the other metadata
  values are kept in `metadata_json` and cannot be filtered on.
- `NE` follows SQL semantics: the records where the field is missing are
  not returned.

## License

MIT
