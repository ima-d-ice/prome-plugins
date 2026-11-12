"""Integration tests with the real Qwen3-Embedding-0.6B encoder.

The weights are in the local Hugging Face cache, but ``sentence-transformers``
is an optional dependency; the tests skip when it — or the weights — are
missing.  No weights are downloaded during the test session.
"""

from __future__ import annotations

import pytest

from retrieval.contracts.models import RetrievalQuery
from retrieval.dense.qwen import QwenDenseRetriever
from retrieval.lexical.bm25.sqlite_fts5 import SQLiteFTS5Index
from retrieval.strategies.hybrid import HybridRetrievalStrategy
from tests.retrieval.support import make_chunk


def qwen_model():
    """Return the real Qwen adapter, skipping when it cannot be used."""
    pytest.importorskip("sentence_transformers")
    from embeddings.qwen3 import Qwen3EmbeddingModel

    try:
        model = Qwen3EmbeddingModel()
        model.get_model_info()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"the Qwen3 model is not usable here: {exc}")
    return model


@pytest.fixture()
def corpus():
    """Return a small corpus that distinguishes the two channels."""
    return [
        make_chunk(
            "c0", "auth.md",
            "API keys are rotated every thirty days by the key rotation "
            "scheduler. A key that is not rotated within the grace period "
            "of seven days is revoked automatically.",
            index=0, count=2,
        ),
        make_chunk(
            "c1", "billing.md",
            "Invoices are issued on the first day of every month. An invoice "
            "for a partial month is pro-rated by day.",
            index=0, count=2,
        ),
    ]


def test_the_real_dense_channel_retrieves(store, corpus):
    """The dense channel works with the real encoder."""
    model = qwen_model()

    from embeddings.contracts.models import RepresentationType
    from vector_stores.contracts.models import (
        CollectionConfig,
        DistanceMetric,
        SpaceVectors,
        StoredRecord,
        VectorSpaceConfig,
    )

    space = VectorSpaceConfig(
        name="dense",
        vector_type=RepresentationType.DENSE,
        embedding_space=model.get_embedding_space(),
        dimensions=model.capabilities.embedding_dimensions,
        distance=DistanceMetric.COSINE,
    )
    store.create_collection(
        CollectionConfig(name="docs", vector_spaces={"dense": space})
    )

    from embeddings.contracts.models import EmbeddingInput

    records = []
    for chunk in corpus:
        result = model.embed_documents([EmbeddingInput(text=chunk.text)])[0]
        records.append(
            StoredRecord(
                chunk=chunk,
                vectors={
                    "dense": SpaceVectors(
                        space="dense",
                        embedding_space=model.get_embedding_space(),
                        dense=list(result.dense_vector),
                    )
                },
            )
        )
    store.upsert("docs", records)

    retriever = QwenDenseRetriever(store, model, collection="docs")
    response = retriever.retrieve(RetrievalQuery(text="How often are API keys rotated?"))

    assert response.results
    assert response.results[0].chunk_id == "c0"


def test_the_real_hybrid_strategy_retrieves(store, corpus, tmp_path):
    """The hybrid strategy works with the real encoder and a real SQLite index."""
    model = qwen_model()

    from embeddings.contracts.models import EmbeddingInput, RepresentationType
    from vector_stores.contracts.models import (
        CollectionConfig,
        DistanceMetric,
        SpaceVectors,
        StoredRecord,
        VectorSpaceConfig,
    )

    space = VectorSpaceConfig(
        name="dense",
        vector_type=RepresentationType.DENSE,
        embedding_space=model.get_embedding_space(),
        dimensions=model.capabilities.embedding_dimensions,
        distance=DistanceMetric.COSINE,
    )
    store.create_collection(
        CollectionConfig(name="docs", vector_spaces={"dense": space})
    )

    records = []
    for chunk in corpus:
        result = model.embed_documents([EmbeddingInput(text=chunk.text)])[0]
        records.append(
            StoredRecord(
                chunk=chunk,
                vectors={
                    "dense": SpaceVectors(
                        space="dense",
                        embedding_space=model.get_embedding_space(),
                        dense=list(result.dense_vector),
                    )
                },
            )
        )
    store.upsert("docs", records)

    index = SQLiteFTS5Index(tmp_path / "lexical.db")
    index.open()
    from retrieval.lexical.bm25.sqlite_fts5 import LexicalRecord

    index.upsert([LexicalRecord.from_chunk(chunk) for chunk in corpus])

    dense = QwenDenseRetriever(store, model, collection="docs")
    from retrieval.lexical.bm25.sqlite_fts5 import BM25Retriever

    lexical = BM25Retriever(index)
    strategy = HybridRetrievalStrategy(dense, lexical)

    response = strategy.retrieve(
        RetrievalQuery(text="How often are the signing keys rotated?")
    )

    assert response.results
    # The auth chunk answers the question; it should be the first result.
    assert response.results[0].chunk_id == "c0"
    assert response.results[0].doc_id == "auth.md"

    index.close()
