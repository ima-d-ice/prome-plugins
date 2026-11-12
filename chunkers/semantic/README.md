# Semantic Chunker

Meaning-aware chunk boundaries: instead of splitting on character counts or
heading levels, the semantic chunker compares **what consecutive pieces of
text actually say** and cuts where the topic changes.

The comparison is done with cosine similarity over embeddings from a
pluggable embedding model. No LLM is involved, and no model is loaded
internally — the model is injected.

## When to use it

Use semantic chunking when:

- the document has **no reliable structure** (plain prose, transcripts, PDFs
  whose headings were not extracted) but still covers several topics;
- you want chunks that are **topically coherent**, so an embedded chunk
  matches a query about *that* topic rather than a mixture of topics;
- you accept the **extra cost** of embedding every candidate segment
  (measured and reported separately in the benchmark).

Use the **structure-aware** chunker when the parser produced good headings
— heading-based boundaries are free and deterministic, while semantic
boundaries cost an embedding pass. The two can be combined by using the
structure-aware chunker for parents and the semantic chunker for children.

## How it works

1. The document is flattened into blocks with character offsets.
2. **Candidate segments** are extracted per block: sentences for prose
   blocks, whole blocks for structured ones (tables, lists, code, records),
   so structured data is never cut mid-row.
3. A segment larger than `max_chunk_size` is split further with the shared
   recursive splitter (atomic blocks are split on line boundaries, so rows
   stay intact).
4. All segments are embedded **in batches** with the injected model's
   `embed_documents()`.
5. **Adjacent** segments are compared with cosine similarity.
6. A boundary is placed wherever the similarity falls below the threshold,
   at a heading, or where the size limit would be exceeded.
7. Chunks below `min_chunk_size` are merged into a neighbour in the same
   section, when that still fits `max_chunk_size`.
8. `overlap` of trailing context is carried into the next chunk.

Content is never dropped and block references stay accurate: every chunk is
an exact slice of the flattened document text.

## Threshold interpretation

`similarity_threshold` is the **minimum adjacent cosine similarity**
required for two consecutive segments to stay in the same chunk.

> A similarity **below** the threshold marks a topic change and therefore a
> chunk boundary.

- Range: `[-1.0, 1.0]` (cosine similarity range).
- **Higher** → cuts more aggressively → smaller, more topically focused
  chunks.
- **Lower** → keeps more text together → larger, mixed-topic chunks.
- Default: `0.5`, a conservative "clearly a different topic" cut.

It is *not* a distance threshold, and it is *not* a percentile over the
document's similarity distribution: it is an absolute, reproducible cosine
value. The interpretation is recorded in every chunk's metadata as
`metadata["semantic_threshold_kind"] == "minimum_adjacent_cosine_similarity"`.

Headings are **not** compared by similarity — a heading is a title, not body
text, so its embedding says nothing about topical coherence. A heading is
glued to the content that follows it instead.

## Configuration

```python
from chunkers.semantic import SemanticChunker, SemanticConfig
from embeddings.deterministic import DeterministicEmbeddingModel

config = SemanticConfig(
    embedding_model=DeterministicEmbeddingModel(dimension=64),
    similarity_threshold=0.5,   # minimum adjacent cosine similarity
    max_chunk_size=1000,        # hard upper bound for a chunk
    min_chunk_size=None,        # optional lower bound (chunks below it are merged)
    overlap=0,                  # trailing context carried into the next chunk
    tokenizer=None,             # None = characters; pass a TokenCounter or `str -> int`
    batch_size=32,              # segments per embed_documents() call
    respect_structure=True,     # headings start a new chunk and never merge sections
)
```

Sizes are measured with `tokenizer` and reported in `Chunk.length` /
`Chunk.length_unit`, like the other chunkers. Invalid configuration raises
`ChunkerConfigError` (threshold out of range, overlap not smaller than the
chunk size, minimum larger than the maximum, non-positive batch size, ...).

## The embedding model

The semantic chunker depends only on the shared interface in
[`embeddings/contracts`](../../embeddings/contracts):

```python
class EmbeddingModel(Protocol):
    def embed_documents(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]: ...
    def embed_query(self, input: EmbeddingInput) -> EmbeddingResult: ...
    def embed_queries(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]: ...
```

Each candidate segment is wrapped in an `EmbeddingInput` and sent through
`embed_documents`; the `dense_vector` of the returned `EmbeddingResult` is
used for the cosine comparison. Any object implementing the interface can be
injected — there is no hardcoded provider and no model loading inside the
chunker.

Adapters available in the package:

| Implementation | Use |
|---|---|
| `embeddings.deterministic.DeterministicEmbeddingModel` | Dependency-free, reproducible, no weights. For tests and algorithm benchmarks. |
| `embeddings.local.LocalTransformerEmbeddingModel` | A locally cached transformer checkpoint. **Never downloads**; raises `EmbeddingConfigError` if the model is not in the local cache. |
| `embeddings.Qwen3EmbeddingModel` | The production embedding encoder (Qwen3-Embedding-0.6B only). Needs `sentence-transformers`. |

### Failure handling

The chunker **never** silently falls back to fixed or recursive chunking:

- a missing model or a model without `embed_documents()` → `ChunkerConfigError`;
- a backend error during inference → `SemanticChunkerError` with the
  original exception preserved as `__cause__` — the run fails loudly;
- inconsistent vector dimensions, a wrong number of returned results, a
  dimension that disagrees with `EmbeddingResult.dimensions`, a missing
  dense vector or a non-dense representation → `SemanticChunkerError`.

## Structure preservation

- Headings and section boundaries are respected: a heading opens a new chunk
  and is glued to the content that follows it.
- Sections are never merged just because their embeddings happen to be
  similar (`respect_structure=True`).
- Lists and tables are kept intact where possible; an oversized one is split
  on row boundaries rather than mid-row.
- Source block order is preserved, and block references stay accurate.
- No content is silently discarded.
- A coherent paragraph is not split unnecessarily: its sentences have a high
  adjacent similarity and stay together.

## Provenance and metadata

Besides the standard fields (`doc_id`, `block_refs`, `metadata["pages"]`,
`metadata["char_start"]`/`["char_end"]`), every chunk records what produced
it, for reproducibility:

```python
chunk.metadata["embedding_model"]          # e.g. "sentence-transformers/all-MiniLM-L6-v2"
chunk.metadata["embedding_dimension"]      # e.g. 384
chunk.metadata["semantic_config"]          # full configuration snapshot
chunk.metadata["semantic_threshold"]       # 0.5
chunk.metadata["semantic_threshold_kind"]  # "minimum_adjacent_cosine_similarity"
chunk.metadata["semantic_max_chunk_size"]
chunk.metadata["semantic_min_chunk_size"]
chunk.metadata["semantic_overlap"]
chunk.metadata["semantic_batch_size"]
chunk.metadata["semantic_respect_structure"]
chunk.metadata["section_index"]            # section the chunk belongs to
chunk.metadata["segment_count"]           # candidate segments in the chunk
```

## Example

```python
from parsers.pdf.pymupdf import PyMuPDFParser
from chunkers.semantic import SemanticChunker, SemanticConfig
from embeddings.local import LocalTransformerEmbeddingModel

doc = PyMuPDFParser().parse("report.pdf")
chunker = SemanticChunker(
    SemanticConfig(
        embedding_model=LocalTransformerEmbeddingModel(),
        similarity_threshold=0.55,
        max_chunk_size=800,
    )
)
for chunk in chunker.chunk(doc):
    print(chunk.index, chunk.length, chunk.heading, chunk.metadata["segment_count"])
```

## Known limitations

- **Cost.** Every candidate segment is embedded, so semantic chunking is
  markedly slower than the size-based strategies. The embedding time is
  reported separately by the benchmark so the cost is never hidden.
- **Depends on the embedding quality.** A weak embedding model produces
  noisy similarity values and therefore arbitrary boundaries. Use a real
  sentence-embedding model in production.
- **Sentence granularity.** Boundaries are detected between candidate
  segments, which are sentences for prose. A topic change *inside* a single
  sentence is not detected.
- **No cross-document reasoning.** Boundaries are computed within one
  document only; adjacent chunks from different documents are never compared.
- **Threshold is absolute.** A single threshold must work across the whole
  corpus; corpora with very different similarity scales may need different
  values per document type.
- The chunker is deterministic only if the embedding model is deterministic.
- No vector store, retrieval, reranking or generation is included.
