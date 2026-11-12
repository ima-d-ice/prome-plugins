# Prome RAG Reranking Ecosystem

Modular reranking component for reordering retrieved candidates in RAG systems.

## Overview

The reranker receives a user query and a list of candidate chunks, scores each
query–candidate pair using the Qwen/Qwen3-Reranker-0.6B model, and returns the
candidates reordered by relevance.

It is deliberately small, independent, and locally executable. It does not
depend on Qdrant, BM25, RRF, or any retrieval strategy.

## Quick Start

```python
from reranking import Qwen3Reranker
from reranking.contracts.models import RerankCandidate, RerankRequest

reranker = Qwen3Reranker(device="cpu")

candidates = (
    RerankCandidate(candidate_id="c1", text="some chunk text"),
    RerankCandidate(candidate_id="c2", text="another chunk text"),
)

request = RerankRequest(
    query="your query here",
    candidates=candidates,
)

response = reranker.rerank(request)
for item in response.reranked_items:
    print(f"rank {item.new_rank}: score={item.raw_score:.3f}, "
          f"id={item.candidate.candidate_id}")
```

## Architecture

- **Contracts** (`reranking/contracts/`): dependency-free data models and protocol
  (`RerankCandidate`, `RerankRequest`, `RerankedItem`, `RerankResponse`, `Reranker`)
- **Qwen3 adapter** (`reranking/qwen3/`): CrossEncoder-based implementation
  of the `Reranker` protocol
- **Tests** (`tests/reranking/`): unit, adapter, and integration test layers
- **Benchmarks** (`benchmarks/reranking/`): performance and quality metrics

## Directory Layout

```text
reranking/
    __init__.py
    contracts/
        __init__.py
        models.py       # RerankCandidate, RerankRequest, RerankedItem, RerankResponse, Reranker protocol
        protocols.py    # Reranker protocol
        exceptions.py   # RerankError, RerankConfigError, RerankDependencyError, RerankInferenceError, InvalidScoreError, UnsupportedDeviceError
    qwen3/
        __init__.py
        reranker.py     # Qwen3Reranker implementation
        README.md
    README.md
```

## Optional Dependencies

Add ``reranking-qwen3`` extra to install the reranker dependencies:

```bash
pip install "prome-rag-plugins[reranking-qwen3]"
```

This installs:
- `sentence-transformers`
- `torch`
- and its transitive dependencies

The core reranking contracts are dependency-free.

## License

This project is MIT licensed.