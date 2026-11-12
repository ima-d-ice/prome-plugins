# Hierarchical Retrieval Evaluation & Benchmark

Two harnesses, both reproducible and both free of model downloads.

| Script | What it answers |
|---|---|
| `evaluate.py` | does the hierarchy keep the quality, and how much of the corpus does it skip? |
| `benchmark.py` | what does the hierarchy cost? |

## `evaluate.py` -- quality

Three systems run on the **same corpus, the same chunks and the same
embeddings**:

* `flat` -- a dense search over **every** child chunk of the corpus;
* `parent` -- the coarse stage alone: a dense search over the parent
  sections;
* `hierarchical` -- the full strategy: the coarse stage selects the parents,
  the fine stage retrieves the children inside them.

### The corpus

`corpus.py` holds three hand-written documents with the ground truth
attached:

* `platform-operations.md` -- a long technical manual with seven sections;
* `billing-guide.md` -- multi-section documentation;
* `retrieval-study.md` -- a research-style paper (abstract, method, results,
  threats to validity).

The corpus deliberately contains the situations the hierarchy is supposed to
handle: repeated terminology across sections, sections that share their
vocabulary, and queries whose answer is a single sentence in the middle of a
long section.

### The labels

A query names the **sentences** that answer it, not the chunk ids.  The
harness resolves those sentences to chunk ids *after* chunking, so the labels
stay valid when the chunking configuration changes:

* a relevant **child** is the chunk that contains a labelled sentence;
* a relevant **parent** is the parent of that child, and additionally every
  parent whose heading is named by the query.

### The metrics

`metrics.py` computes the standard ones: recall at 1/3/5, hit rate at 1/3/5,
MRR, nDCG at 5 and 10, and the p50/p95 latency.

A metric that cannot be computed (no relevant id) is reported as `n/a` and
excluded from the average -- never as a zero.

The **candidate reduction ratio** is the fraction of the candidate space the
fine stage did *not* score: `1 - candidates_searched / candidates_total`.
For the flat search it is `0` by construction, because it scores every child.

### Usage

```bash
python benchmarks/retrieval/hierarchical/evaluate.py                # deterministic
python benchmarks/retrieval/hierarchical/evaluate.py --embedding qwen
python benchmarks/retrieval/hierarchical/evaluate.py --json report.json
```

The deterministic model (`embeddings.deterministic`) is a hashed
bag-of-words projection: reproducible and dependency free, but **not
semantic**.  Its numbers describe the behaviour of the *architecture*, not
the quality of a production encoder.  Use `--embedding qwen` for that.

## `benchmark.py` -- performance

The benchmark measures:

* the **indexing throughput** (records written per second);
* the **query latency** (p50/p95/mean) of the three systems;
* the **candidate reduction** the hierarchy buys;
* optionally, how the latency and the reduction **scale** with the corpus
  size (`--scaling`).

The corpus is synthetic and the vectors come from a seeded generator, so two
runs on the same machine produce the same numbers.  With `--embedding qwen`
the real encoder is used and the corpus is embedded for real.

```bash
python benchmarks/retrieval/hierarchical/benchmark.py
python benchmarks/retrieval/hierarchical/benchmark.py --embedding qwen --scaling
```

### Reading the latency

The in-process Qdrant modes (`memory`, `local`) evaluate a payload filter
with a full scan over the collection, so the cost of a search is dominated by
the number of records and not by the number of hits.  The hierarchical
strategy therefore performs **two** filtered scans where the flat search
performs one, and the latency is roughly twice as high in those modes.

A backend with real payload indexes (a Qdrant server) pushes the restriction
into the index, and the fine stage then costs much less than the coarse one.
The benchmark does not hide this: it reports the raw latency of each system.

## Reports

Both scripts accept `--json <path>` and write the full report, so a run can
be archived and compared.  The numbers quoted in the final report of the
milestone were produced by these scripts.
