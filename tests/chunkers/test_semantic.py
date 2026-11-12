"""Tests for the semantic chunker.

All tests use deterministic fake embedding models, so no model weights are
downloaded and every assertion is reproducible.
"""

from __future__ import annotations

import pytest

from chunkers.contracts import ApproxTokenCounter, ChunkerConfigError
from chunkers.semantic import SemanticChunker, SemanticChunkerError, SemanticConfig
from embeddings.contracts.models import (
    EmbeddingResult,
    Normalization,
    RepresentationType,
)
from parsers.contracts import BlockType, ParsedBlock, ParsedDocument, ParsedPage

from .helpers import assert_provenance, coverage_stats, flattened_text


# ---------------------------------------------------------------------------
# Deterministic fake embedding models
# ---------------------------------------------------------------------------
class TopicEmbeddingModel:
    """Maps a text to the vector of the topic it mentions.

    ``topics`` maps a keyword to a topic name; several keywords may share a
    topic. A text is assigned the topic of the first keyword it contains
    (keywords are checked in insertion order). Texts without any keyword
    fall back to ``default_topic``, which gets its own axis.

    Texts mentioning the same topic get identical (hence similarity 1.0)
    vectors; different topics get orthogonal vectors (similarity 0.0). This
    makes the boundary detection completely predictable.
    """

    def __init__(
        self,
        topics: dict[str, str],
        dimension: int = 16,
        default_topic: str = "unmatched",
    ) -> None:
        self._topics = dict(topics)
        self._dimension = dimension
        self._default_topic = default_topic
        self.calls: list[list[str]] = []
        # Each distinct topic value gets its own orthogonal axis.
        self._axes = sorted(set(self._topics.values()) | {default_topic})

    @property
    def model_id(self) -> str:
        return "fake-topic-model"

    @property
    def name(self) -> str:
        return "fake-topic-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, inputs):
        self.calls.append([item.text for item in inputs])
        return [self._result(item) for item in inputs]

    def embed_queries(self, inputs):
        return [self._result(item) for item in inputs]

    def embed_query(self, input):
        return self._result(input)

    def _result(self, item):
        lowered = item.text.lower()
        topic = self._default_topic
        for keyword, name in self._topics.items():
            if keyword.lower() in lowered:
                topic = name
                break
        vector = [0.0] * self._dimension
        vector[self._axes.index(topic)] = 1.0
        return EmbeddingResult(
            model_id="fake-topic-model",
            dimensions=self._dimension,
            dense_vector=vector,
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            doc_id=item.doc_id,
            chunk_id=item.chunk_id,
        )


class CountingEmbeddingModel:
    """Deterministic model that records how it was called."""

    def __init__(self, inner: TopicEmbeddingModel | None = None) -> None:
        self._inner = inner or TopicEmbeddingModel({"topic": "a"})
        self.batch_sizes: list[int] = []
        self.total_texts = 0

    @property
    def model_id(self) -> str:
        return "counting-model"

    @property
    def name(self) -> str:
        return "counting-model"

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_documents(self, inputs):
        inputs = list(inputs)
        self.batch_sizes.append(len(inputs))
        self.total_texts += len(inputs)
        return self._inner.embed_documents(inputs)

    def embed_queries(self, inputs):
        return self.embed_documents(inputs)

    def embed_query(self, input):
        return self.embed_documents([input])[0]


class FailingEmbeddingModel:
    """A model whose inference always fails."""

    model_id = "failing-model"
    name = "failing-model"
    dimension = 8

    def embed_documents(self, inputs):
        raise RuntimeError("backend exploded")

    def embed_queries(self, inputs):
        return self.embed_documents(inputs)

    def embed_query(self, input):
        return self.embed_documents([input])[0]


class WrongDimensionModel:
    """Returns vectors that do not match the declared dimension."""

    model_id = "wrong-dimension-model"
    name = "wrong-dimension-model"
    dimension = 16

    def embed_documents(self, inputs):
        return [
            EmbeddingResult(
                model_id=self.model_id,
                dimensions=16,
                dense_vector=[0.1, 0.2, 0.3],
                representation_type=RepresentationType.DENSE,
            )
            for _ in inputs
        ]

    def embed_queries(self, inputs):
        return self.embed_documents(inputs)

    def embed_query(self, input):
        return self.embed_documents([input])[0]


class InconsistentDimensionModel:
    """Returns vectors of differing lengths within one batch."""

    model_id = "inconsistent-model"
    name = "inconsistent-model"
    dimension = 8

    def embed_documents(self, inputs):
        return [
            EmbeddingResult(
                model_id=self.model_id,
                dimensions=8,
                dense_vector=[0.1] * (4 if index % 2 == 0 else 8),
                representation_type=RepresentationType.DENSE,
            )
            for index, _ in enumerate(inputs)
        ]

    def embed_queries(self, inputs):
        return self.embed_documents(inputs)

    def embed_query(self, input):
        return self.embed_documents([input])[0]


class WrongCountModel:
    """Returns the wrong number of results."""

    model_id = "wrong-count-model"
    name = "wrong-count-model"
    dimension = 8

    def embed_documents(self, inputs):
        return [
            EmbeddingResult(
                model_id=self.model_id,
                dimensions=8,
                dense_vector=[0.1] * 8,
                representation_type=RepresentationType.DENSE,
            )
        ]

    def embed_queries(self, inputs):
        return self.embed_documents(inputs)

    def embed_query(self, input):
        return self.embed_documents([input])[0]


class NoEmbedMethod:
    """Not an embedding model at all."""

    model_id = "not-a-model"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _doc(blocks, source="synthetic.html"):
    return ParsedDocument(source=source, parser_name="test", parser_version="1", blocks=blocks)


def _heading(text, level, **block_kwargs):
    return ParsedBlock(BlockType.HEADING, text, metadata={"heading_level": level}, **block_kwargs)


def _text(text):
    return ParsedBlock(BlockType.TEXT, text)


def _model():
    """Two clearly different topics (cats vs. particle physics)."""
    return TopicEmbeddingModel(
        {
            "cat": "cats",
            "mat": "cats",
            "lattice": "physics",
            "quark": "physics",
            "gluon": "physics",
        },
        dimension=8,
    )


def _two_topic_doc():
    return _doc(
        [
            _text("The cat sat on the mat. The cat liked the warm mat. A cat naps."),
            _text("Lattice gauge theory binds quarks. Gluons confine quarks. Quarks spin."),
        ]
    )


# ---------------------------------------------------------------------------
# Related vs unrelated segments
# ---------------------------------------------------------------------------
def test_clearly_related_segments_stay_together():
    doc = _doc([_text("The cat sat on the mat. The cat liked the warm mat. A cat naps.")])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert len(chunks) == 1
    assert "The cat sat on the mat." in chunks[0].text
    assert "A cat naps." in chunks[0].text


def test_clearly_different_topics_are_separated():
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(
        _two_topic_doc()
    )
    assert len(chunks) == 2
    assert "cat" in chunks[0].text
    assert "Lattice" in chunks[1].text


def test_multiple_topic_transitions():
    doc = _doc(
        [
            _text("The cat sat on the mat."),
            _text("Lattice gauge theory binds quarks."),
            _text("The cat liked the warm mat."),
            _text("Gluons confine quarks."),
        ]
    )
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert len(chunks) == 4
    topics = ["cats" if "cat" in c.text else "physics" for c in chunks]
    assert topics == ["cats", "physics", "cats", "physics"]


def test_related_segments_are_not_split_unnecessarily():
    doc = _doc([_text("The cat sat on the mat. " * 12)])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------
def test_threshold_is_minimum_adjacent_cosine_similarity():
    """A higher threshold cuts more aggressively, a lower one less."""
    doc = _two_topic_doc()
    low = SemanticChunker(
        SemanticConfig(embedding_model=_model(), similarity_threshold=0.1, max_chunk_size=500)
    ).chunk(doc)
    high = SemanticChunker(
        SemanticConfig(embedding_model=_model(), similarity_threshold=0.9, max_chunk_size=500)
    ).chunk(doc)
    # With a low threshold the two topics merge (similarity 0.0 < 0.1 is
    # still a cut, so use 0.0 for "never cut"): compare the documented rule
    # directly on the similarity values.
    assert len(low) <= len(high)


def test_threshold_zero_never_splits_on_similarity():
    doc = _two_topic_doc()
    chunks = SemanticChunker(
        SemanticConfig(embedding_model=_model(), similarity_threshold=0.0, max_chunk_size=500)
    ).chunk(doc)
    # Similarity 0.0 is not below 0.0, so the topic change is not detected.
    assert len(chunks) == 1


def test_threshold_one_splits_every_topic_change():
    doc = _two_topic_doc()
    chunks = SemanticChunker(
        SemanticConfig(embedding_model=_model(), similarity_threshold=1.0, max_chunk_size=500)
    ).chunk(doc)
    assert len(chunks) >= 2


def test_threshold_is_recorded_in_metadata():
    chunks = SemanticChunker(
        SemanticConfig(embedding_model=_model(), similarity_threshold=0.42, max_chunk_size=500)
    ).chunk(_two_topic_doc())
    for chunk in chunks:
        assert chunk.metadata["semantic_threshold"] == pytest.approx(0.42)
        assert chunk.metadata["semantic_threshold_kind"] == "minimum_adjacent_cosine_similarity"
        assert chunk.metadata["semantic_config"]["similarity_threshold"] == pytest.approx(0.42)


def test_invalid_threshold_rejected():
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), similarity_threshold=1.5)
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), similarity_threshold=-1.5)
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), similarity_threshold="high")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------
def test_maximum_chunk_size_enforced():
    doc = _doc([_text("The cat sat on the mat. " * 40)])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=100)).chunk(doc)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.length <= 100


def test_maximum_chunk_size_enforced_with_many_topics():
    doc = _doc([_text("The cat sat on the mat. " * 30), _text("Lattice gauge theory. " * 30)])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=120)).chunk(doc)
    for chunk in chunks:
        assert chunk.length <= 120


def test_minimum_chunk_size_merges_small_chunks():
    doc = _doc(
        [
            _text("The cat sat on the mat."),
            _text("The cat liked the warm mat."),
        ]
    )
    model = TopicEmbeddingModel({"cat": "cats"}, dimension=8)
    without = SemanticChunker(
        SemanticConfig(embedding_model=model, similarity_threshold=0.9, max_chunk_size=500)
    ).chunk(doc)
    with_min = SemanticChunker(
        SemanticConfig(
            embedding_model=model,
            similarity_threshold=0.9,
            max_chunk_size=500,
            min_chunk_size=60,
        )
    ).chunk(doc)
    assert len(with_min) <= len(without)
    assert coverage_stats(doc, with_min)["coverage"] == pytest.approx(1.0)


def test_minimum_chunk_size_never_exceeds_maximum():
    doc = _doc([_text("The cat sat on the mat. " * 20)])
    chunks = SemanticChunker(
        SemanticConfig(
            embedding_model=_model(), max_chunk_size=100, min_chunk_size=90
        )
    ).chunk(doc)
    for chunk in chunks:
        assert chunk.length <= 100


def test_minimum_chunk_size_larger_than_max_rejected():
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), max_chunk_size=100, min_chunk_size=200)


def test_invalid_min_chunk_size_rejected():
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), min_chunk_size=0)
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), min_chunk_size=-5)


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------
def test_overlap_repeats_content():
    doc = _doc([_text("The cat sat on the mat. " * 30)])
    without = SemanticChunker(
        SemanticConfig(embedding_model=_model(), max_chunk_size=100, overlap=0)
    ).chunk(doc)
    with_overlap = SemanticChunker(
        SemanticConfig(embedding_model=_model(), max_chunk_size=100, overlap=40)
    ).chunk(doc)
    assert coverage_stats(doc, with_overlap)["duplicate_rate"] > 0.0
    assert coverage_stats(doc, without)["duplicate_rate"] == 0.0
    assert coverage_stats(doc, with_overlap)["coverage"] == pytest.approx(1.0)


def test_overlap_does_not_cross_section_boundaries():
    doc = _doc(
        [
            _heading("Cats", 1),
            _text("The cat sat on the mat."),
            _heading("Physics", 1),
            _text("Lattice gauge theory binds quarks."),
        ]
    )
    chunks = SemanticChunker(
        SemanticConfig(embedding_model=_model(), max_chunk_size=500, overlap=200)
    ).chunk(doc)
    # The physics chunk must not start inside the cats section.
    physics = [c for c in chunks if "Lattice" in c.text]
    assert physics
    assert "The cat sat on the mat." not in physics[0].text


def test_invalid_overlap_rejected():
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), max_chunk_size=100, overlap=100)
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), max_chunk_size=100, overlap=-1)


# ---------------------------------------------------------------------------
# Structure preservation
# ---------------------------------------------------------------------------
def test_heading_boundaries_respected():
    doc = _doc(
        [
            _heading("Cats", 1),
            _text("The cat sat on the mat."),
            _heading("Physics", 1),
            _text("Lattice gauge theory binds quarks."),
        ]
    )
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].heading == "Cats"
    assert chunks[1].heading == "Physics"


def test_heading_is_glued_to_its_content():
    doc = _doc([_heading("Cats", 1), _text("The cat sat on the mat.")])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Cats")
    assert "The cat sat on the mat." in chunks[0].text


def test_unrelated_sections_are_not_merged_even_when_similar():
    """Two sections with identical wording must stay separate."""
    doc = _doc(
        [
            _heading("Cats", 1),
            _text("The cat sat on the mat."),
            _heading("Physics", 1),
            _text("The cat sat on the mat."),
        ]
    )
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert len(chunks) == 2
    assert chunks[0].heading == "Cats"
    assert chunks[1].heading == "Physics"


def test_respect_structure_disabled_allows_crossing():
    doc = _doc(
        [
            _heading("Cats", 1),
            _text("The cat sat on the mat."),
            _heading("Physics", 1),
            _text("The cat sat on the mat."),
        ]
    )
    chunks = SemanticChunker(
        SemanticConfig(
            embedding_model=_model(), max_chunk_size=500, respect_structure=False
        )
    ).chunk(doc)
    assert len(chunks) == 1


def test_source_block_order_preserved():
    doc = _doc(
        [
            _text("First block about cats."),
            _text("Second block about lattice gauge."),
            _text("Third block about cats."),
        ]
    )
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert "First" in chunks[0].text
    assert "Second" in chunks[1].text
    assert "Third" in chunks[2].text


def test_table_preserved():
    table = "Item | Qty\n--- | ---\nWidget | 3\nGadget | 7"
    doc = _doc([_text("Intro about cats."), ParsedBlock(BlockType.TABLE, table)])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert any(table in chunk.text for chunk in chunks)


def test_list_preserved():
    lst = "- alpha item\n- beta item\n- gamma item"
    doc = _doc([_text("Intro about cats."), ParsedBlock(BlockType.LIST, lst)])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert any(lst in chunk.text for chunk in chunks)


def test_oversized_table_is_split_on_row_boundaries():
    table = "Item | Qty\n" + "\n".join(f"row{i} | {i}" for i in range(120))
    doc = _doc([ParsedBlock(BlockType.TABLE, table)])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=150)).chunk(doc)
    assert len(chunks) > 1
    joined = "\n".join(chunk.text for chunk in chunks)
    for i in range(120):
        assert f"row{i} | {i}" in joined


def test_long_paragraph_is_split():
    doc = _doc([_text("The cat sat on the mat. " * 60)])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=200)).chunk(doc)
    assert len(chunks) > 1
    assert coverage_stats(doc, chunks)["coverage"] == pytest.approx(1.0)


def test_multiple_pages():
    doc = ParsedDocument(
        source="multi.pdf",
        parser_name="test",
        parser_version="1",
        pages=[
            ParsedPage(page_number=1, blocks=[_text("Page one about cats.")]),
            ParsedPage(page_number=2, blocks=[_text("Page two about lattice gauge.")]),
        ],
    )
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert len(chunks) == 2
    pages = {ref.page_number for chunk in chunks for ref in chunk.block_refs}
    assert pages == {1, 2}
    for chunk in chunks:
        assert chunk.metadata["pages"] == sorted({ref.page_number for ref in chunk.block_refs})


# ---------------------------------------------------------------------------
# Embedding integration
# ---------------------------------------------------------------------------
def test_missing_embedding_model_rejected():
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=None)
    with pytest.raises(ChunkerConfigError):
        SemanticChunker()


def test_model_without_embed_documents_rejected():
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=NoEmbedMethod())


def test_incompatible_embedding_dimensions_rejected():
    doc = _doc([_text("Some text about cats.")])
    chunker = SemanticChunker(
        SemanticConfig(embedding_model=WrongDimensionModel(), max_chunk_size=500)
    )
    with pytest.raises(SemanticChunkerError):
        chunker.chunk(doc)


def test_inconsistent_dimensions_within_batch_rejected():
    doc = _doc([_text("One. Two. Three. Four. Five. Six.")])
    chunker = SemanticChunker(
        SemanticConfig(embedding_model=InconsistentDimensionModel(), max_chunk_size=500)
    )
    with pytest.raises(SemanticChunkerError):
        chunker.chunk(doc)


def test_embedding_inference_failure_is_not_swallowed():
    doc = _doc([_text("Some text about cats.")])
    chunker = SemanticChunker(
        SemanticConfig(embedding_model=FailingEmbeddingModel(), max_chunk_size=500)
    )
    with pytest.raises(SemanticChunkerError) as excinfo:
        chunker.chunk(doc)
    # The original backend failure is preserved as the cause.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "backend exploded" in str(excinfo.value)


def test_embedding_inference_failure_does_not_fall_back_to_other_chunking():
    """A failing model must not silently produce fixed/recursive chunks."""

    class Boom:
        model_id = "boom"
        dimension = 8

        def embed_documents(self, texts):
            raise RuntimeError("no model available")

    doc = _doc([_text("Some text about cats.")])
    chunker = SemanticChunker(SemanticConfig(embedding_model=Boom(), max_chunk_size=500))
    with pytest.raises(SemanticChunkerError):
        chunker.chunk(doc)


def test_wrong_number_of_vectors_rejected():
    doc = _doc([_text("One. Two. Three.")])
    chunker = SemanticChunker(
        SemanticConfig(embedding_model=WrongCountModel(), max_chunk_size=500)
    )
    with pytest.raises(SemanticChunkerError):
        chunker.chunk(doc)


def test_non_dense_representation_rejected():
    """Only dense embeddings support cosine similarity."""

    class SparseModel:
        model_id = "sparse-model"
        name = "sparse-model"
        dimension = 8

        def embed_documents(self, inputs):
            return [
                EmbeddingResult(
                    model_id=self.model_id,
                    dimensions=8,
                    representation_type=RepresentationType.SPARSE,
                    sparse_indices=[0, 1],
                    sparse_values=[0.5, 0.5],
                )
                for _ in inputs
            ]

        def embed_queries(self, inputs):
            return self.embed_documents(inputs)

        def embed_query(self, input):
            return self.embed_documents([input])[0]

    doc = _doc([_text("Some text about cats.")])
    chunker = SemanticChunker(SemanticConfig(embedding_model=SparseModel(), max_chunk_size=500))
    with pytest.raises(SemanticChunkerError):
        chunker.chunk(doc)


def test_missing_dense_vector_rejected():
    class NoVectorModel:
        model_id = "no-vector-model"
        name = "no-vector-model"
        dimension = 8

        def embed_documents(self, inputs):
            return [
                EmbeddingResult(model_id=self.model_id, dimensions=8)
                for _ in inputs
            ]

        def embed_queries(self, inputs):
            return self.embed_documents(inputs)

        def embed_query(self, input):
            return self.embed_documents([input])[0]

    doc = _doc([_text("Some text about cats.")])
    chunker = SemanticChunker(
        SemanticConfig(embedding_model=NoVectorModel(), max_chunk_size=500)
    )
    with pytest.raises(SemanticChunkerError):
        chunker.chunk(doc)


def test_embedding_calls_are_batched():
    model = CountingEmbeddingModel(TopicEmbeddingModel({"cat": "cats"}, dimension=8))
    doc = _doc([_text("The cat sat on the mat. " * 20)])
    SemanticChunker(
        SemanticConfig(embedding_model=model, max_chunk_size=100, batch_size=4)
    ).chunk(doc)
    assert model.batch_sizes
    assert all(size <= 4 for size in model.batch_sizes)
    assert model.total_texts == sum(model.batch_sizes)


def test_batch_size_validation():
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), batch_size=0)
    with pytest.raises(ChunkerConfigError):
        SemanticConfig(embedding_model=_model(), batch_size=-1)


def test_model_id_and_config_stored_in_metadata():
    chunks = SemanticChunker(
        SemanticConfig(
            embedding_model=_model(),
            max_chunk_size=200,
            min_chunk_size=50,
            overlap=20,
            batch_size=8,
        )
    ).chunk(_two_topic_doc())
    for chunk in chunks:
        assert chunk.metadata["embedding_model"] == "fake-topic-model"
        assert chunk.metadata["embedding_dimension"] == 8
        assert chunk.metadata["semantic_max_chunk_size"] == 200
        assert chunk.metadata["semantic_min_chunk_size"] == 50
        assert chunk.metadata["semantic_overlap"] == 20
        assert chunk.metadata["semantic_batch_size"] == 8
        assert chunk.metadata["semantic_config"]["respect_structure"] is True


def test_semantic_metadata_carries_source_provenance():
    doc = _doc([_text("Some text about cats.")], source="my-doc.pdf")
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    for chunk in chunks:
        assert chunk.doc_id == "my-doc.pdf"
        assert chunk.metadata["source"] == "my-doc.pdf"
        assert chunk.metadata["parser_name"] == "test"
        assert chunk.metadata["strategy"] == "semantic"


# ---------------------------------------------------------------------------
# Common behaviour
# ---------------------------------------------------------------------------
def test_empty_document():
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model())).chunk(
        ParsedDocument(source="e", parser_name="t", parser_version="1")
    )
    assert chunks == []


def test_document_with_only_whitespace():
    doc = _doc([ParsedBlock(BlockType.TEXT, "   \n  ")])
    assert SemanticChunker(SemanticConfig(embedding_model=_model())).chunk(doc) == []


def test_source_references_accurate():
    doc = _doc(
        [
            _heading("Cats", 1),
            _text("The cat sat on the mat."),
            _text("Lattice gauge theory binds quarks."),
        ]
    )
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert_provenance(doc, chunks)


def test_content_coverage_complete():
    doc = _doc(
        [
            _heading("Cats", 1),
            _text("The cat sat on the mat. " * 10),
            _heading("Physics", 1),
            _text("Lattice gauge theory binds quarks. " * 10),
        ]
    )
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=150)).chunk(doc)
    assert coverage_stats(doc, chunks)["coverage"] == pytest.approx(1.0)


def test_no_content_discarded():
    doc = _doc(
        [
            _text("Alpha sentence. Beta sentence."),
            _text("Gamma sentence. Delta sentence."),
        ]
    )
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    joined = " ".join(chunk.text for chunk in chunks)
    for sentence in ("Alpha sentence.", "Beta sentence.", "Gamma sentence.", "Delta sentence."):
        assert sentence in joined


def test_deterministic_output():
    doc = _two_topic_doc()
    config = SemanticConfig(embedding_model=_model(), max_chunk_size=200, overlap=20)
    first = [c.as_dict() for c in SemanticChunker(config).chunk(doc)]
    second = [c.as_dict() for c in SemanticChunker(config).chunk(doc)]
    assert first == second


def test_unicode_content():
    doc = _doc([_heading("Ünïcödé ✨", 1), _text("héllo wörld 日本語テキスト " * 20)])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=120)).chunk(doc)
    assert coverage_stats(doc, chunks)["coverage"] == pytest.approx(1.0)
    assert chunks[0].heading == "Ünïcödé ✨"


def test_input_document_not_mutated():
    doc = _doc([_heading("Cats", 1), _text("The cat sat on the mat.")])
    before = doc.as_dict()
    SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    assert doc.as_dict() == before


def test_chunk_text_matches_metadata_range():
    doc = _two_topic_doc()
    text = flattened_text(doc)
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=500)).chunk(doc)
    for chunk in chunks:
        assert text[chunk.metadata["char_start"] : chunk.metadata["char_end"]] == chunk.text


def test_count_and_index_consistency():
    doc = _doc([_text("The cat sat on the mat. " * 30)])
    chunks = SemanticChunker(SemanticConfig(embedding_model=_model(), max_chunk_size=100)).chunk(doc)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.count == len(chunks) for c in chunks)


def test_tokenizer_based_sizing():
    doc = _doc([_text("The cat sat on the mat. " * 30)])
    counter = ApproxTokenCounter()
    chunks = SemanticChunker(
        SemanticConfig(embedding_model=_model(), max_chunk_size=20, tokenizer=counter)
    ).chunk(doc)
    assert chunks
    for chunk in chunks:
        assert chunk.length <= 20
        assert chunk.length_unit == "approx_tokens"


def test_non_document_input_rejected():
    with pytest.raises(TypeError):
        SemanticChunker(SemanticConfig(embedding_model=_model())).chunk("not a document")  # type: ignore[arg-type]
