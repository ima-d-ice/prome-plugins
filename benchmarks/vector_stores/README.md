# Vector Store Benchmarks

Reproducible benchmark of the vector-store adapters on a **deterministic
synthetic corpus**.  No embedding model is downloaded and no inference is
run: the vectors come from a seeded pseudo random generator, so two runs on
the same machine produce the same corpus - and the *same* corpus for every
backend, which makes the numbers comparable.

```bash
python benchmarks/vector_stores/benchmark.py --backend qdrant  --mode memory
python benchmarks/vector_stores/benchmark.py --backend qdrant  --mode server --url http://localhost:6333
python benchmarks/vector_stores/benchmark.py --backend lancedb --mode local --path /tmp/lancedb-bench
python benchmarks/vector_stores/benchmark.py --backend faiss    --mode local --path /tmp/faiss-bench
python benchmarks/vector_stores/benchmark.py --backend pgvector --mode server --dsn postgresql://user@host/db
```

## What is measured

| Scenario | What it measures |
|---|---|
| `dense_exact` | brute-force dense search (`params={"exact": True}`) - the correctness baseline |
| `dense_approximate` | the native ANN path of the backend and its **recall@k against the exact baseline** |
| `sparse` | sparse vector search (only when the backend has one) |
| `multi_vector` | late interaction search (only when the backend has one) |
| `dense_filtered` | dense search with a metadata filter |
| `batch_upsert` | insert throughput (records per second) |
| `index_build` | the time to build the native approximate index |
| `storage_footprint` | the on-disk size of the collection (bytes, bytes per vector) |

Latency is reported as mean, p50 and p95 in milliseconds.  Recall@k is the
average fraction of the exact top-k that the approximate top-k contains (and
its minimum over the queries).

**No combined score is produced.**  Latency and recall are separate
measurements; they trade off against each other and a single number would
hide that.

## Mode matters

| Mode | HNSW / ANN | Payload indexes | Meaning of the numbers |
|---|---|---|---|
| Qdrant `memory` / `local` | not used (full scan) | accepted but not used | adapter overhead, not index performance |
| Qdrant `server` | real HNSW index | real payload indexes | what a deployment would see |
| LanceDB `local` | real IVF index | real scalar indexes | the embedded engine |
| FAISS `local` | real HNSW / IVF index | SQLite expression indexes | the embedded engine |
| pgvector `server` | real HNSW / IVFFlat index | real PostgreSQL indexes | what a deployment would see |

In the in-process Qdrant modes the "approximate" path is exact as well, so
the recall is 1.0 by construction.  That is a property of the Qdrant
client, not a measurement of HNSW.

The pgvector HNSW index is built by the server in the background: the
planner only uses it once it is ready, which is why the recall of a small
corpus can be lower than the exact baseline.

## Reports

`reports/<backend>-<mode>/` holds the JSON and CSV reports of the last run
of each backend.  Previous reports are copied to `reports/<backend>-<mode>/archive/`
before being overwritten (unless `--no-archive` is passed).

The numbers describe this synthetic corpus on the machine that produced
them; they are regression indicators, not production baselines.
