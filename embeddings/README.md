# Prome RAG Embedding Ecosystem

Modular, plug-and-play embedding module supporting one local embedding model:

**Qwen3-Embedding-0.6B** — General-purpose all-rounder

## Installation

```bash
# Qwen3-Embedding-0.6B is the production embedding model
pip install "prome-rag-plugins[qwen3]"
# or install sentence-transformers directly:
pip install sentence-transformers
```

The adapter uses `sentence-transformers` and will raise a clear `ImportError` if it is not installed.

## Quick Start

```python
from embeddings import Qwen3EmbeddingModel
from embeddings.contracts.models import EmbeddingInput

# Initialize model (lazy — loads on first embed call)
model = Qwen3EmbeddingModel(output_dimension=1024)

# Embed a single query
result = model.embed_query(EmbeddingInput(
    text="what is this about",
    doc_id="doc1",
    chunk_id="chunk1"
))
print(f"Query embedding: {len(result.dense_vector)}-dim, model={result.model_id}")

# Embed documents
documents = [
    EmbeddingInput(text="this is a document about dogs", doc_id="doc1", chunk_id="c1"),
    EmbeddingInput(text="another document about cats", doc_id="doc1", chunk_id="c2"),
]
results = model.embed_documents(documents)
for r in results:
    print(f"doc_id={r.doc_id}, chunk_id={r.chunk_id}, dim={r.dimensions}")
```

## Local Inference

All embedding models run locally — no hosted APIs, no cloud fallback.

```python
from embeddings import Qwen3EmbeddingModel
from embeddings.contracts.models import EmbeddingInput

model = Qwen3EmbeddingModel(output_dimension=1024)
result = model.embed_query(EmbeddingInput(text="what is this", doc_id="d1", chunk_id="c1"))
print(f"Query embedding dim: {result.dimensions}")
```

## Device Compatibility

The adapter auto-detects the best available device:

- **MPS** (macOS Metal): preferred if available
- **CUDA**: if NVIDIA GPU available
- **CPU**: fallback if no GPU acceleration available

## Document Embedding

```python
from embeddings import Qwen3EmbeddingModel
from embeddings.contracts.models import EmbeddingInput

model = Qwen3EmbeddingModel(output_dimension=1024)
inputs = [
    EmbeddingInput(text="document about dogs", doc_id="doc1", chunk_id="chunk1"),
    EmbeddingInput(text="document about cats", doc_id="doc1", chunk_id="chunk2"),
]
results = model.embed_documents(inputs)
for r in results:
    print(f"doc_id={r.doc_id}, chunk_id={r.chunk_id}, dim={r.dimensions}")
```

## Online Query Embedding

```python
from embeddings import Qwen3EmbeddingModel
from embeddings.contracts.models import EmbeddingInput

model = Qwen3EmbeddingModel(output_dimension=1024)
result = model.embed_query(EmbeddingInput(text="what about dogs", doc_id="doc1", chunk_id="chunk1"))
print(f"Query embedding dim: {result.dimensions}")
```

Also supports batch query embedding via `embed_queries()`.

## Batch Processing

All adapters support batch inference with configurable batch size (default 32):

```python
results = model.embed_documents(inputs, batch_size=16)
results = model.embed_queries(inputs, batch_size=16)
```

## Model Selection

The production embedding model is **Qwen3-Embedding-0.6B**.

Switching embedding models requires re-embedding the relevant document corpus. Vectors from different models are not compatible in the same ordinary dense-vector collection.

## Known Limitations

- **Embedding-space isolation**: Vectors from different models are not compatible. If you switch embedding models, re-embed the relevant document corpus.

- **Qwen3-Embedding-0.6B**: General-purpose model. May not perform as well as specialized models on specific domains (code, multilingual, etc.).

## References

- Qwen3-Embedding-0.6B: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)