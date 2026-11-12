# Qwen3-Embedding-0.6B Embedding Adapter

General-purpose all-rounder embedding model for the Prome RAG ecosystem.

## Model

- **Model ID**: `Qwen/Qwen3-Embedding-0.6B`
- **Dimensions**: 1024 (configurable within model-supported options)
- **Representation**: Dense only
- **Normalization**: L2
- **Max Sequence Length**: 8192 (reduced to 512 when `output_dimension` is configured)
- **Supported Languages**: English, Chinese, code

## Installation

```bash
pip install "prome-rag-plugins[qwen3]"
# or install sentence-transformers directly:
pip install sentence-transformers
```

The adapter uses `sentence-transformers` and will raise a clear `ImportError` if it is not installed.

## Quick Start

```python
from embeddings.qwen3 import Qwen3EmbeddingModel
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

## Capabilities

- **Dense embedding**: ✅ 1024-dimensional vectors with L2 normalization
- **Sparse embedding**: ❌ Not supported
- **Multi-vector**: ❌ Not supported
- **Query embedding**: ✅ Supported (may use different instructions than documents)
- **Document embedding**: ✅ Supported
- **Batch inference**: ✅ Supported (configurable batch_size)
- **Max sequence length**: 8192 input tokens

## Configuration

```python
model = Qwen3EmbeddingModel(
    output_dimension=512,    # Optional: config within model-supported options
    batch_size=16,           # Optional: batch size for encoding
    device=None              # Optional: auto-detects cuda/mps/cpu
)
```

- `output_dimension`: Must be one of the model's supported options. If `None`, uses default 1024.
  When set, `max_sequence_length` is reduced to 512.

## Embedding Instructions

Qwen3-Embedding-0.6B supports separate document and query encoding with model-recommended
instructions. The adapter currently uses default SentenceTransformer encoding without explicit
prefixes, as the model's native `encode()` method handles instruction application internally.

- **Document encoding**: Standard text encoding
- **Query encoding**: Standard text encoding (may differ from document encoding at the model level)

## Output Dimensions

The model supports configurable output dimensions. The default is 1024. When an
`output_dimension` is configured, the model respects it within its supported options.

## Device Compatibility

The adapter auto-detects the best available device:

```python
model = Qwen3EmbeddingModel()  # auto-detects MPS > CUDA > CPU
```

## Error Handling

- `ImportError`: Raised if `sentence-transformers` is not installed (on first embed call)
- `RuntimeError`: Raised if model weights cannot be loaded (unlikely with Qwen3-Embedding-0.6B
  as it's a small model that typically downloads successfully)
- Invalid `output_dimension`: The adapter will use the configured dimension without validation
  against the model's supported options (pass-through behavior)

## License

This adapter is part of the Prome RAG Plugins ecosystem (MIT licensed).