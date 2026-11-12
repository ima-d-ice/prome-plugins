# Qwen3-Reranker-0.6B Reranking Adapter

Wraps Sentence Transformers' CrossEncoder for query--candidate relevance
scoring using the Qwen/Qwen3-Reranker-0.6B model.

## Model

- **Model ID**: `Qwen/Qwen3-Reranker-0.6B`
- **Task**: Query–document relevance scoring (cross-encoder)
- **Output**: Raw relevance logits (higher = more relevant)
- **Default prompt**: "Given a web search query, retrieve relevant passages that answer the query"
- **Max sequence length**: 512 tokens (combined query + document)
- **Batch processing**: Configurable batch size to avoid unbounded inference

## Installation

```bash
pip install "prome-rag-plugins[reranking-qwen3]"
# or install dependencies directly:
pip install sentence-transformers torch
```

The adapter depends on `sentence-transformers` and `torch`.  Optional
`mps` support requires Apple Silicon; `cuda` support requires a compatible GPU.

## Quick Start

```python
from reranking.qwen3.reranker import Qwen3Reranker
from reranking.contracts.models import RerankCandidate, RerankRequest

# Initialize reranker (lazy — loads on first rerank call)
reranker = Qwen3Reranker(device="cpu")

# Build candidates
candidates = (
    RerankCandidate(candidate_id="c1", text="Quantum computing uses qubits."),
    RerankCandidate(candidate_id="c2", text="The capital of France is Paris."),
    RerankCandidate(candidate_id="c3", text="Basketball is a team sport."),
)

# Create reranking request
request = RerankRequest(
    query="What is quantum computing?",
    candidates=candidates,
)

# Run reranking
response = reranker.rerank(request)
for item in response.reranked_items:
    print(f"rank {item.new_rank}: score={item.raw_score:.3f}, "
          f"candidate_id={item.candidate.candidate_id}, "
          f"text='{item.candidate.text[:60]}...'")
```

## Configuration

```python
model = Qwen3Reranker(
    model_name="Qwen/Qwen3-Reranker-0.6B",
    device="cpu",          # or "mps" (Apple Silicon) or "cuda" (NVIDIA GPU)
    max_sequence_length=512,
    batch_size=16,         # score 16 candidates per inference batch
)
```

### Task instruction

The default task instruction is:

> Given a web search query, retrieve relevant passages that answer the query

Override it per-request via ``RerankRequest(task_instruction=...)`` or
per-reranker-instance by subclassing or configuring the adapter.

## API

### :class:`~reranking.qwen3.reranker.Qwen3Reranker`

- **``__init__``**: configure model name, device, max sequence length, batch size
- **``rerank``**: score and reorder candidates (`RerankRequest` -> `RerankResponse`)
- **``get_model_info``**: retrieve metadata about the loaded model
- **``get_capabilities``**: describe the reranker's abilities

### :class:`~reranking.contracts.models.RerankRequest`

- **``query``**: the user query text
- **``candidates``**: tuple of `RerankCandidate`
- **``top_k``**: optional limit on the number of results to return
- **``task_instruction``**: optional override of the model's default prompt

### :class:`~reranking.contracts.models.RerankCandidate`

- **``candidate_id``**: stable identifier
- **``text``**: the chunk/text content to score
- **``original_rank``**: pre-rerank rank (used for tie-breaking)
- **``metadata``**: optional metadata preserved through reranking
- **``source_provenance``**: optional provenance from the original retrieval
- **``retrieval_result``**: optional reference to the original retrieval result

### :class:`~reranking.contracts.models.RerankedItem`

- **``candidate``**: the original rerank candidate
- **``raw_score``**: the raw relevance logit from the model
- **``original_rank``**: the candidate's rank before reranking
- **``new_rank``**: the candidate's 1-based rank in the reranked result

### :class:`~reranking.contracts.models.RerankResponse`

- **``reranked_items``**: the candidates ordered by relevance (best first)
- **``model_id``**: the identity of the reranker model
- **``model_revision``**: the model revision / tag when available
- **``candidates_scored``**: how many candidates were scored
- **``elapsed_ms``**: optional wall-clock time for scoring
- **``score_interpretation``**: how to interpret the raw scores (`"raw"` or `"sigmoid"`)
- **``diagnostics``**: additional diagnostics

## Device compatibility

- **CPU**: always supported, no extra dependencies
- **MPS**: supported on Apple Silicon when `torch.backends.mps.is_available()`
- **CUDA**: supported when NVIDIA GPU and `torch.cuda.is_available()`

## Error handling

- `RerankDependencyError`: raised if `sentence-transformers` is not installed
- `UnsupportedDeviceError`: raised if an unavailable device is requested
- `RerankConfigError`: raised for invalid configuration values
- `RerankInferenceError`: raised if model inference fails
- `InvalidScoreError`: raised if the model returns NaN or infinity scores

## License

This adapter is part of the Prome RAG Plugins ecosystem (MIT licensed).