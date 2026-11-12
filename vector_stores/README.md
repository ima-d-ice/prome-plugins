# Vector Store Ecosystem

Backend-independent contracts for storing and searching chunk embeddings,
plus the adapter implementations.  The first adapter is **Qdrant**
(`vector_stores/qdrant/`).

The contracts are dependency free: importing `vector_stores` never imports a
vector database client.

## What a vector store does here

A vector store manages **collections** of chunk records and answers vector
queries.  It never embeds text and never loads a model: embeddings are
always passed in, precomputed.

```text
parsers/  →  chunkers/  →  embeddings/  →  vector_stores/  →  (retrieval, later)
```

## Reused contracts (no duplicates)

| Concern | Source of truth |
|---|---|
| The stored chunk (id, doc id, text, index/count, metadata, provenance) | `chunkers.contracts.models.Chunk` / `BlockReference` |
| Embedding-space identity (model, revision, dimensions, normalization) | `embeddings.contracts.models.EmbeddingSpaceDescriptor` |
| Vector kind (dense / sparse / multi-vector) | `embeddings.contracts.models.RepresentationType` |
| Normalization | `embeddings.contracts.models.Normalization` |

The vector store adds only what the storage side needs: distance metrics,
score direction, collection configuration, filters, index descriptions and
search requests/results.

## Package layout

```text
vector_stores/
├── contracts/
│   ├── models.py        # CollectionConfig, VectorSpaceConfig, StoredRecord,
│   │                    # SearchRequest, SearchResult, ChunkProvenance, ...
│   ├── filters.py       # portable metadata filter language
│   ├── indexes.py       # index kinds, index configuration/description
│   ├── capabilities.py  # capability descriptor of a backend
│   ├── protocols.py     # the VectorStore protocol
│   ├── payload.py       # backend independent encoding of the record payload
│   ├── validation.py    # the portable validation rules
│   └── exceptions.py    # the error hierarchy
├── qdrant/              # the Qdrant adapter (optional dependency)
├── lancedb/             # the LanceDB adapter (optional dependency)
├── faiss/               # the FAISS adapter (optional dependency)
└── pgvector/            # the PostgreSQL adapter (optional dependency)
```

## Collections and vector spaces

A **collection** is a logical group of stored chunks.  It declares one or
more **named vector spaces**; a space fixes the vector kind, the width, the
distance metric and the embedding space (model + revision) that may be
stored in it.

```python
from embeddings.contracts.models import EmbeddingSpaceDescriptor, Normalization, RepresentationType
from vector_stores.contracts import (
    CollectionConfig, DistanceMetric, HnswIndexConfig,
    PayloadFieldType, PayloadIndexSpec, VectorSpaceConfig,
)

dense = VectorSpaceConfig(
    name="dense",
    vector_type=RepresentationType.DENSE,
    embedding_space=EmbeddingSpaceDescriptor(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        model_revision="main",
        dimensions=1024,
        representation_type=RepresentationType.DENSE,
        normalization=Normalization.L2,
    ),
    dimensions=1024,
    distance=DistanceMetric.COSINE,
    hnsw=HnswIndexConfig(m=16, ef_construct=100),
)

config = CollectionConfig(
    name="docs",
    vector_spaces={"dense": dense},
    payload_indexes=(PayloadIndexSpec("tenant", PayloadFieldType.KEYWORD),),
)
```

Rules the contracts enforce:

- a collection declares at least one vector space; a dense or multi-vector
  space declares its width; a sparse space must not;
- **matching dimensions never establish compatibility**: the model id, the
  model revision, the width and the normalization must match as well;
- one named space holds one representation kind - a dense space never
  receives a sparse vector, and a multi-vector space never receives an
  averaged vector;
- metadata fields are only indexed when the user asks for it.

## Stored records

A record keeps the whole `Chunk` and adds the vectors of each space:

```python
from vector_stores.contracts import SpaceVectors, StoredRecord

record = StoredRecord(
    chunk=chunk,                     # a chunkers.contracts.models.Chunk
    vectors={
        "dense": SpaceVectors(
            space="dense",
            embedding_space=dense.embedding_space,
            dense=[0.01] * 1024,
        )
    },
    metadata={"tenant": "acme"},     # filterable payload fields
)
```

`StoredRecord.from_embedding_result(chunk, result)` bridges the two
ecosystems: it takes an `embeddings.contracts.models.EmbeddingResult` and
picks the representation that matches the target space.

## Metadata filters

The filter language is portable and small:

```python
from vector_stores.contracts import and_, eq, exists, gt, is_in, ne, not_, or_

eq("tenant", "acme")          # equality
ne("tenant", "acme")          # inequality
is_in("lang", ["en", "fr"])   # IN / NOT IN
gt("score", 2.0)              # > >= < <=
exists("tenant")              # field is present (or missing)
and_(eq("a", 1), or_(eq("b", 2), not_(eq("c", 3))))
```

Each backend declares the operations it supports in
`store.capabilities.filter_ops` and raises
`UnsupportedFilterOperationError` for the others - it never silently
returns a different result set.

## Searching

```python
from vector_stores.contracts import SearchRequest

results = store.search(
    SearchRequest(
        collection="docs",
        space="dense",
        dense=query_vector,
        limit=10,
        filter=eq("tenant", "acme"),
        score_threshold=0.5,
        params={"hnsw_ef": 128},   # backend search parameters
    )
)
```

A result keeps the original chunk id, the document id, the text and the
metadata (when the payload was requested), the provenance, the raw backend
score, the metric and the score direction:

```python
result.chunk_id, result.doc_id, result.score
result.metric            # DistanceMetric.COSINE
result.score_direction   # ScoreDirection.HIGHER_IS_BETTER
```

Scores are **not** comparable across metrics or across models: a cosine
similarity (higher is better) and a euclidean distance (lower is better) are
different quantities.

## Indexes

Vector index algorithms and metadata indexes are usually managed by the
backend itself, so the contracts describe them instead of implementing them:

| Kind | Meaning | Portable parameters |
|---|---|---|
| `exact` | brute-force scan, exact results | query: `exact` |
| `hnsw` | approximate dense index | build: `m`, `ef_construct`, `full_scan_threshold`; query: `hnsw_ef` |
| `sparse-inverted` | inverted index over sparse dimensions | build: `full_scan_threshold` |
| `payload` | metadata index over one field | build: `field`, `field_type` |
| `multi-vector` | late interaction over the token matrix | - |

The same name is **not** a portability promise: `store.capabilities.index_kinds`
lists, per kind, the parameters the backend actually honours.

## Adapters

| Backend | Status | Dense | ANN | Sparse | Multi-vector | Persistence |
|---|---|---|---|---|---|---|
| Qdrant | implemented | yes | HNSW | yes (native) | yes (native max-sim) | memory / local / server |
| LanceDB | implemented | yes | IVF (incl. IVF_HNSW) | stored, **not searchable** | yes (native max-sim) | memory / local |
| FAISS | implemented | yes | HNSW / IVF | no | no | memory / local (FAISS + SQLite) |
| pgvector | implemented | yes | HNSW / IVFFlat | yes (pgvector >= 0.7) | stored, **not searchable** | server (PostgreSQL) |

Every adapter implements the same `VectorStore` protocol and the same
conformance test suite (`tests/vector_stores/conformance.py`); the
differences above are the *real* differences between the backends, not a
smoothed-over claim.

Each adapter has its own README:

- `vector_stores/qdrant/README.md`
- `vector_stores/lancedb/README.md`
- `vector_stores/faiss/README.md`
- `vector_stores/pgvector/README.md`

## Roadmap

The three planned adapters (LanceDB, FAISS, pgvector) are implemented.  The
next steps are:

1. more index kinds per backend (IVF / PQ for LanceDB, quantization for
   Qdrant);
2. a shared "retrieval" layer that consumes the stores (not part of the
   vector-store milestone).

Not implemented in this milestone: IVF/PQ/quantization on every backend,
custom ANN algorithms, BM25 / full-text retrieval.

## License

MIT
