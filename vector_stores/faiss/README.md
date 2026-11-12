# FAISS Adapter

FAISS is a similarity search library, not a document database.  This adapter
therefore combines two stores:

- **FAISS** holds the dense vectors and answers the nearest neighbour
  queries (`IndexFlat` for exact search, `IndexHNSWFlat` and `IndexIVFFlat`
  for approximate search);
- **SQLite** holds the records: the public chunk id, the document id, the
  text, the metadata, the provenance, the embedding-space descriptor and
  the collection configuration.

SQLite is the **authoritative** store: the FAISS index is a derived artifact
and is rebuilt automatically when the two disagree.

## Installation

```bash
pip install 'prome-rag-plugins[faiss]'
# or
pip install 'faiss-cpu>=1.16'
```

`faiss-cpu` is an optional dependency and is imported lazily.

## Quick start

```python
from vector_stores.contracts import CollectionConfig, SearchRequest
from vector_stores.faiss import FaissVectorStore

store = FaissVectorStore.local("/tmp/faiss")   # persistent directory
store = FaissVectorStore.memory()              # ephemeral directory
store.create_collection(config)
store.upsert("docs", records)
hits = store.search(
    SearchRequest(collection="docs", space="dense", dense=query_vector)
)
store.close()
```

## Layout

```text
<root>/<collection>/
├── records.sqlite3   # the authoritative records + the collection configuration
└── index-dense.faiss # the derived vector index
```

SQLite holds the chunk id, the document id, the contract payload, the
metadata (as a JSON document) and the vector itself (as float32 bytes), so
the index can always be rebuilt.  The metadata fields declared in the
collection configuration get a SQLite expression index
(`json_extract(metadata, '$.field')`).

## Indexes

| Kind | FAISS |
|---|---|
| exact | `IndexFlatIP` (cosine / dot) or `IndexFlatL2` (euclidean) |
| HNSW | `IndexHNSWFlat` (`m`, `efConstruction`, `efSearch`) |
| IVF | `IndexIVFFlat` (`lists`) - needs at least `lists` vectors to train |
| payload | a SQLite expression index on a metadata field |

```python
store.create_index(
    "docs",
    IndexConfig(kind=IndexKind.HNSW, target="dense",
                build_params={"m": 16, "ef_construct": 100}),
)
```

Query-time parameters: `hnsw_ef` (HNSW), `nprobe` (IVF), `exact`.

## Consistency and recovery

FAISS renumbers the ids when a vector is removed, so the adapter never
removes a vector from the index:

- an upsert of an existing chunk appends a new vector and leaves the old
  one in the index; the old id is not live any more and the search filters
  it out;
- a deletion only removes the row from SQLite;
- the index is **rebuilt** (and the ids renumbered) by
  `store.rebuild_index(collection)`, or automatically when the adapter
  detects that the index and the store disagree (a missing, stale or
  corrupt index file).

A filtered search is **exact over the matching records**: the filter is
evaluated by SQLite and the scores are computed over exactly that subset.
The adapter never returns fewer results than there are matching records,
and it never post-filters a small set of ANN candidates.

## Limitations

- Dense vectors only: a sparse or multi-vector space is rejected at
  collection creation, and nothing is ever densified or averaged.
- The vectors are stored twice (in SQLite and in the FAISS index); that is
  what makes the rebuild possible.
- The deleted records leave their vector in the index until the next
  rebuild, which makes the index grow.
- **FAISS and torch cannot share a process on macOS.**  The `faiss-cpu`
  wheel links its own copy of `libomp` and so does torch; when the second
  OpenMP runtime initializes, the interpreter aborts with `OMP: Error #15`
  (a hard abort, not a catchable exception).  The adapter detects the
  common case - torch already loaded - and raises
  `VectorStoreDependencyError` instead of letting the process die.  Use the
  FAISS adapter in a process that does not import torch (the FAISS test
  suite is therefore run on its own: `pytest tests/vector_stores/faiss`).

## License

MIT
