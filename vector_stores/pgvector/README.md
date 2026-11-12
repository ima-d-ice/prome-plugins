# PostgreSQL (pgvector) Adapter

The adapter stores precomputed embeddings in a PostgreSQL table and searches
them with the pgvector operators.  It needs a PostgreSQL server with the
`vector` extension installed (`CREATE EXTENSION vector;`).

The adapter never creates a database, never talks to a hosted service and
never requires a running server for the unit tests that do not touch it.

## Installation

```bash
pip install 'prome-rag-plugins[pgvector]'
# or
pip install 'psycopg[binary]>=3.2' 'pgvector>=0.4'
```

`psycopg` (v3) and `pgvector` are optional dependencies and are imported
lazily.

## Quick start

```python
from vector_stores.contracts import CollectionConfig, SearchRequest
from vector_stores.pgvector import PgVectorVectorStore

store = PgVectorVectorStore.server("postgresql://user@host/db")
store.create_collection(config)
store.upsert("docs", records)
hits = store.search(
    SearchRequest(collection="docs", space="dense", dense=query_vector)
)
store.close()
```

A collection is a **table** in the configured schema.  The declared
configuration is stored in `<schema>.prome_collection_config`, so the
"foreign model" check still applies after a restart.

## Table layout

| column | content |
|---|---|
| `chunk_id` | the public chunk id (primary key) |
| `doc_id` | the public document id |
| `<space>_vector` | dense vectors (`vector(dim)`) |
| `<space>_sparse` | sparse vectors (`sparsevec`) |
| `<space>_multivector` | multi-vector matrices (`jsonb`, storage only) |
| `payload` | the contract fields (`jsonb`) |
| `metadata` | the user metadata (`jsonb`) |
| `embedding_space` | the embedding-space descriptors (`jsonb`) |

## Metrics and operators

| Metric | Operator | Reported score |
|---|---|---|
| cosine | `<=>` | `1 - distance` (a similarity, higher is better) |
| euclidean | `<->` | the distance (lower is better) |
| dot | `<#>` | `-distance` (the inner product, higher is better) |

pgvector has no operator for the manhattan metric: the adapter rejects it.

## Indexes

| Kind | PostgreSQL |
|---|---|
| exact | no index; the planner is prevented from using the index (`params={"exact": True}`) |
| HNSW | `CREATE INDEX ... USING hnsw (col vector_cosine_ops) WITH (m, ef_construction)` |
| IVF | `CREATE INDEX ... USING ivfflat (col vector_cosine_ops) WITH (lists)` |
| payload | a B-tree on the document id, a GIN index on the metadata and an expression index on every declared metadata field |

```python
store.create_index(
    "docs",
    IndexConfig(kind=IndexKind.HNSW, target="dense",
                build_params={"m": 16, "ef_construction": 64}),
)
```

Query-time parameters: `hnsw_ef_search`, `ivfflat_probes`, `exact`.

## Filters

The portable filter language is compiled into parameterized SQL.  The
metadata is read as `jsonb` (`metadata->'field'`), so the comparison works
for every JSON type.  **Every value is bound as a parameter** and every
identifier is validated, so neither a field name nor a filter value can
change the meaning of a statement.

## Sparse vectors

`sparsevec` needs pgvector 0.7 or newer; the installed version is detected
at connection time and reported in `store.extension_version` and in the
capability descriptor.  The indices of the embedding contract and of
pgvector both count the dimensions from 0, so they are used as they are -
the values are never densified.

pgvector only compares two `sparsevec` values of the same width, and the
embedding contract does not carry that width: the adapter fixes it with the
first batch of a collection and rejects a record that would need more
dimensions.

There is **no inverted index for `sparsevec`**: the sparse search is always
an exact scan.

## Multi-vectors

PostgreSQL has no late interaction operator.  The adapter stores the token
matrix as `jsonb` and **rejects a multi-vector search** with
`UnsupportedVectorTypeError`; it never averages the token vectors.

## Transactions

Every write is a single statement in one transaction that is committed
before the call returns; a failed statement therefore leaves the collection
untouched.

## License

MIT
