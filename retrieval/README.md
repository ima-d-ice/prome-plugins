# Prome RAG Retrieval Ecosystem

Retrieval turns a query into ranked passages.  The ecosystem builds on the
existing Prome contracts and never duplicates them:

| Concern | Contract |
|---|---|
| identity and provenance of a passage | `chunkers.contracts.models.Chunk` |
| embeddings | `embeddings.contracts.protocols.EmbeddingModel` |
| storage and search | `vector_stores.contracts.protocols.VectorStore` |

The embedding model is always **injected**.  Nothing in this package loads a
model, downloads weights or talks to a hosted service.

## Contents

| Module | What it does |
|---|---|
| `retrieval/contracts/` | The shared contract: `RetrievalQuery`, `RetrievedItem`, `RetrievalResponse`, `RetrievalDiagnostics`, `Retriever`, `RetrievalCapabilities`. |
| `retrieval/dense/` | `DenseRetriever`: embed a query, run a filtered vector search, check the backend's capabilities first. |
| `retrieval/dense/qwen/` | The production dense channel, bound to Qwen3-Embedding-0.6B. |
| `retrieval/lexical/bm25/sqlite_fts5/` | BM25 over a persistent SQLite FTS5 index (relational content + triggers). |
| `retrieval/fusion/rrf/` | Reciprocal Rank Fusion, default `k=60`. |
| `retrieval/postprocessing/deduplication/` | Merge repeated chunk ids, keep the best-ranked channel. |
| `retrieval/postprocessing/adjacent_merge/` | Merge consecutive chunks of the same document. |
| `retrieval/strategies/hybrid/` | The hybrid strategy: dense + BM25, fused with RRF. |
| `retrieval/strategies/hierarchical/` | The hierarchical (coarse-to-fine) strategy. |
| `retrieval/postprocessing/context_expansion/` | Attaches the parent section and the neighbouring children to a match. |
| `retrieval/indexing/` | `IndexingCoordinator`: writes the same chunks into both indexes. Reports per-backend failures and can reconcile the lexical side. |

The hybrid strategy is the general-purpose engine; the hierarchical strategy
is a coarse-to-fine specialization for large documents. Both expose the same
`retrieve(query) -> RetrievalResponse` interface.

## The hybrid strategy

The hybrid strategy runs the dense and the lexical channels in sequence,
then fuses their rankings with the reciprocal rank function.

```
USER QUERY
    |
    v
Query preparation
    |
    +-------------------------+
    |                         |
    v                         v
Dense retrieval          BM25 retrieval
    |                         |
Qwen query embedding     SQLite FTS5 index
    |                         |
Vector store search      Lexical search
    |                         |
    +------------+------------+
                 |
                 v
             RRF fusion
                 |
                 v
       Metadata/filter validation
                 |
                 v
          Deduplication
                 |
                 v
        Adjacent chunk merging
                 |
                 v
         Standardized results
```

```python
from embeddings.qwen3 import Qwen3EmbeddingModel
from retrieval.strategies.hybrid import HybridRetrievalStrategy
from retrieval.dense.qwen import QwenDenseRetriever
from retrieval.lexical.bm25.sqlite_fts5 import SQLiteFTS5Index, BM25Retriever
from vector_stores.qdrant import QdrantVectorStore

model = Qwen3EmbeddingModel()
store = QdrantVectorStore.local("/tmp/qdrant")

# (index the corpus into the store and the lexical index first)
index = SQLiteFTS5Index("/tmp/lexical.db").open()

dense = QwenDenseRetriever(store, model, collection="docs")
lexical = BM25Retriever(index)
strategy = HybridRetrievalStrategy(dense, lexical)
for result in strategy.retrieve("how are the signing keys rotated?").results:
    print(result.rank, result.text)
```

### The two channels

* **Dense** (`retrieval/dense/qwen/`) — Qwen3-Embedding-0.6B, injected, never
  loaded by the channel itself.
* **BM25** (`retrieval/lexical/bm25/sqlite_fts5/`) — the built-in BM25 of
  SQLite FTS5 over a persistent index.  The index is persistent and holds the
  chunk text plus its metadata and provenance.

### Fusion

Only the **ranks** are used; the raw scores are never compared.
`RRF_score(d) = sum over the channels that retrieved d of 1 / (k + rank(d))`,
with `k=60` and `rank` starting at 1.  A chunk found by only one channel
keeps its contribution.

### Failure handling

The default is **strict**: if either channel fails, the whole retrieval
fails.  Pass `HybridConfig(failure_policy="partial")` if you prefer a
partial answer; the failing channel is then recorded in
`response.diagnostics.channel_failures`.

## The hierarchical strategy

The hierarchical strategy answers a query in two steps:

```
USER QUERY
    |
    v
Qwen query embedding                (once)
    |
    v
PARENT-LEVEL DENSE RETRIEVAL        chunk_level == "parent"
    |
    v
Top relevant parent sections
    |
    v
CHILD-LEVEL DENSE RETRIEVAL         chunk_level == "child"
restricted to the selected parents  AND parent_id IN (...)
    |
    v
Fine-grained evidence chunks
    |
    v
Optional adjacent-child expansion
    |
    v
Parent context reconstruction
    |
    v
Final retrieval results
```

The coarse stage finds *which sections* are relevant; the fine stage finds
*which passages* inside them carry the answer.  The fine stage is restricted
to the children of the selected parents **in the store itself**, so the
backend never scores the passages of an unrelated section.

Both levels live in the **same collection** and the **same named dense vector
space**.  What distinguishes them is the `chunk_level` metadata field.

### Indexing

```python
from chunkers.hierarchical import HierarchicalChunker, HierarchicalConfig
from embeddings.qwen3 import Qwen3EmbeddingModel
from retrieval.strategies.hierarchical import HierarchicalIndexer
from vector_stores.qdrant import QdrantVectorStore

model = Qwen3EmbeddingModel()
store = QdrantVectorStore.local("/tmp/qdrant")

indexer = HierarchicalIndexer(store, model, collection="docs")
indexer.ensure_collection()

chunker = HierarchicalChunker(HierarchicalConfig(parent_size=2000, child_size=500))
indexer.index_result(chunker.chunk(document))
```

Parents and children are embedded **independently** with the same encoder: the
embedding of a parent is never reused for its children.  Indexing is
idempotent -- re-indexing a document replaces its records instead of
duplicating them.

### Retrieving

```python
from retrieval.strategies.hierarchical import (
    HierarchicalConfig, HierarchicalRetrievalStrategy,
)

strategy = HierarchicalRetrievalStrategy.create(
    store, model, collection="docs",
    config=HierarchicalConfig(
        parent_top_k=8, child_top_k=40, final_top_k=10,
        include_parent_context=True, expand_adjacent=True, max_adjacent=1,
    ),
)
response = strategy.retrieve("how are the signing keys rotated?")

for result in response.results:
    print(result.rank, result.chunk_id, result.score)
    print("  evidence:", result.text)
    if result.parent_context:
        print("  section:", result.parent_context.text)
    for adjacent in result.adjacent_context:
        print("  nearby :", adjacent.text)
```

### The result

The result keeps the evidence and the context apart:

* `result.chunk_id` / `result.text` are the **evidence** -- the child chunk
  that matched;
* `result.parent_context` is the **section** the evidence sits in;
* `result.adjacent_context` are the **neighbours** of the evidence inside
  that section.

The parent text is never substituted for the evidence text, and a parent is
never returned as if it were a precise match.

### Configuration

| Option | Meaning |
|---|---|
| `parent_top_k` | how many parent sections the coarse stage may select |
| `child_top_k` | how many child chunks the fine stage may retrieve |
| `final_top_k` | how many results are returned |
| `parent_score_threshold` | minimum score a parent must reach |
| `child_score_threshold` | minimum score a child must reach |
| `filters` | additional metadata filter, applied to both stages |
| `include_parent_context` | whether the parent section is attached |
| `expand_adjacent` | whether neighbouring children are attached |
| `max_adjacent` | how many neighbours are attached in total |
| `vector_space` | the named dense vector space to search |

## Evaluation and benchmarks

```bash
python benchmarks/retrieval/hybrid/evaluate.py                 # hybrid quality
python benchmarks/retrieval/hybrid/evaluate.py --embedding qwen
python benchmarks/retrieval/hybrid/benchmark.py                # hybrid performance
python benchmarks/retrieval/hierarchical/evaluate.py            # hierarchical quality
python benchmarks/retrieval/hierarchical/evaluate.py --embedding qwen
python benchmarks/retrieval/hierarchical/benchmark.py          # hierarchical performance
```

See `benchmarks/retrieval/hybrid/README.md` and
`benchmarks/retrieval/hierarchical/README.md` for the methodology.

## Capability handling

The dense retriever checks what the backend can do **before** it searches:

* a filter operation the backend does not support raises
  `UnsupportedRetrievalCapabilityError` instead of being silently dropped;
* on a backend with a fixed schema (LanceDB) the filtered fields must be
  declared in the collection configuration, and the collection declares
  `chunk_level` and `parent_id` for that reason;
* the query embedding must live in the same embedding space as the
  collection, otherwise the search fails loudly.

The fine stage never falls back to searching every child: a broken
restriction is an error, not a silent widening of the search.

## Evaluation and benchmark

```bash
python benchmarks/retrieval/hierarchical/evaluate.py            # quality
python benchmarks/retrieval/hierarchical/evaluate.py --embedding qwen
python benchmarks/retrieval/hierarchical/benchmark.py            # performance
```

See `benchmarks/retrieval/hierarchical/README.md` for the methodology.

## Limitations

* The hierarchy is **one level deep**.  There is no tree and no recursive
  summarisation.
* The hybrid strategy keeps only the ranks; it does not calibrate the raw
  scores.
* The lexical channel requires SQLite with FTS5 (available in every standard
  Python build); if FTS5 is missing, the channel reports the capability
  instead of degrading to an unrelated implementation.
* The context expansion **never rewrites** the source text and never
  summarizes it.  It attaches the original chunks.
* The hierarchical coarse stage costs a second vector search.  On a backend
  whose filter evaluation is a full scan (the in-process Qdrant modes) that
  roughly doubles the latency; on a backend with payload indexes the fine
  stage is much cheaper than the coarse one.
* The hybrid engine writes to two storage systems.  There is no distributed
  transaction, so the coordinator reports per-backend failures and offers a
  reconciliation step; a partial write is always visible in the report.
