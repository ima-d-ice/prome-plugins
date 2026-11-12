# Hybrid Retrieval Evaluation & Benchmark

`evaluate.py` measures **quality** on a labeled corpus of 18 queries across
two documents (an operations runbook and an API reference). The corpus
deliberately mixes the two classes of questions a production system faces:

* **exact technical queries** — error codes (`4291`), API names
  (`X-RateLimit-Remaining`), metric names
  (`platform_key_rotation_current_age_seconds`), commands, header names and
  flag pairs. These are the bread and butter of a lexical channel.
* **natural-language questions** — paraphrases of rate-limiting, session
  refresh, webhook retries and cache eviction. These are the bread and
  butter of a dense channel.

Three systems are compared on the same corpus, the same chunks and the same
embeddings:

* `dense` — the dense channel alone (Qwen3-Embedding-0.6B for `--embedding
  qwen`, or the deterministic double otherwise);
* `bm25`  — the lexical channel alone (SQLite FTS5 BM25);
* `hybrid` — both channels fused with Reciprocal Rank Fusion (`k=60`).

The ground truth is expressed on the **sentence** that carries the answer, so
the labels are valid no matter how the corpus is chunked.

## Metrics

Recall at 1/3/5/10, hit rate at 1/3/5/10, MRR, and p50/p95 latency.

## Usage

```bash
python benchmarks/retrieval/hybrid/evaluate.py                # deterministic double
python benchmarks/retrieval/hybrid/evaluate.py --embedding qwen
python benchmarks/retrieval/hybrid/evaluate.py --json report.json
python benchmarks/retrieval/hybrid/benchmark.py                # performance
python benchmarks/retrieval/hybrid/benchmark.py --embedding qwen --scaling
```

## Measured results (this environment)

Qwen3-Embedding-0.6B, 3 documents, 9 chunks, 18 queries:

| system | R@1 | R@3 | R@5 | R@10 | MRR | p50 ms | p95 ms |
|--------|-----|-----|-----|------|-----|--------|--------|
| dense  | 0.833 | 0.889 | 1.000 | 1.000 | 0.880 | 56.2 | 411.2 |
| bm25   | 0.444 | 0.444 | 0.444 | 0.444 | 0.444 | 0.3 | 1.9 |
| hybrid | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 53.9 | 68.4 |

Per-query breakdown with Qwen: dense wins every paraphrase query
(q09–q18), BM25 wins every exact-identifier query (q01–q08), and the hybrid
engine wins **all 18**. The corpus is small (18 queries, 9 chunks); the
numbers are suggestive, not statistically significant. The deterministic
double is itself lexical, so hybrid matches dense exactly there (0.722 R@1 /
0.889 R@3 / 1.000 R@10 / MRR 0.821).

Benchmark (synthetic corpus, Qdrant memory, 50–200 documents):

* indexing throughput: ~6 900 records/s;
* query latency p50/p95 — dense 0.69/0.84 ms, BM25 0.09/0.17 ms,
  hybrid 0.79/0.94 ms (50 documents);
* latency grows roughly linearly with corpus size (see `--scaling`).

## Limitations

* pgvector and LanceDB have not been tested with this engine yet.
* BM25 alone is weak on paraphrase-style queries (as expected for a lexical
  index); it is reported as measured.
* The corpus is hand-written and small. For production-grade evaluation,
  point the harness at your own documents and labeled queries.
