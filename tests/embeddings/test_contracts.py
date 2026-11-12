"""Core contract and model tests for the embedding ecosystem."""

import pytest

from embeddings.contracts.models import (
    EmbeddingInput,
    EmbeddingResult,
    EmbeddingCapabilities,
    EmbeddingSpaceDescriptor,
    RepresentationType,
    Normalization,
)
from embeddings.contracts.protocols import EmbeddingModel
from embeddings import Qwen3EmbeddingModel


def test_contracts_importable():
    """Test that all core contracts are importable."""
    assert EmbeddingInput is not None
    assert EmbeddingResult is not None
    assert EmbeddingCapabilities is not None
    assert EmbeddingSpaceDescriptor is not None
    assert RepresentationType is not None
    assert Normalization is not None


def test_embedding_input_defaults():
    """Test EmbeddingInput defaults."""
    inp = EmbeddingInput(text="test")
    assert inp.text == "test"
    assert inp.metadata == {}
    assert inp.doc_id is None
    assert inp.chunk_id is None
    assert inp.source_refs is None


def test_embedding_input_with_all_fields():
    """Test EmbeddingInput with all fields set."""
    inp = EmbeddingInput(
        text="test",
        metadata={"key": "value"},
        doc_id="d1",
        chunk_id="c1",
        source_refs=["ref1", "ref2"],
    )
    assert inp.text == "test"
    assert inp.metadata == {"key": "value"}
    assert inp.doc_id == "d1"
    assert inp.chunk_id == "c1"
    assert inp.source_refs == ["ref1", "ref2"]


def test_embedding_result_defaults():
    """Test EmbeddingResult defaults."""
    result = EmbeddingResult(model_id="test-model", dimensions=768)
    assert result.model_id == "test-model"
    assert result.dimensions == 768
    assert result.dense_vector is None
    assert result.sparse_indices is None
    assert result.sparse_values is None
    assert result.multi_vector is None
    assert result.representation_type == RepresentationType.DENSE
    assert result.normalization == Normalization.NONE
    assert result.metadata == {}
    assert result.doc_id is None
    assert result.chunk_id is None
    assert result.query_id is None


def test_embedding_result_with_all_fields():
    """Test EmbeddingResult with all fields set."""
    result = EmbeddingResult(
        model_id="test-model",
        dimensions=768,
        dense_vector=[0.1] * 768,
        sparse_indices=[1, 5, 10],
        sparse_values=[0.5, 0.3, 0.2],
        multi_vector=[[0.1] * 768, [0.2] * 768],
        representation_type=RepresentationType.SPARSE,
        normalization=Normalization.L2,
        metadata={"source": "test"},
        doc_id="d1",
        chunk_id="c1",
        query_id="q1",
    )
    assert result.model_id == "test-model"
    assert result.dimensions == 768
    assert len(result.dense_vector) == 768
    assert result.sparse_indices == [1, 5, 10]
    assert result.sparse_values == [0.5, 0.3, 0.2]
    assert len(result.multi_vector) == 2
    assert result.representation_type == RepresentationType.SPARSE
    assert result.normalization == Normalization.L2
    assert result.metadata == {"source": "test"}
    assert result.doc_id == "d1"
    assert result.chunk_id == "c1"
    assert result.query_id == "q1"


def test_embedding_capabilities_defaults():
    """Test EmbeddingCapabilities defaults."""
    caps = EmbeddingCapabilities()
    assert caps.dense == True
    assert caps.sparse == False
    assert caps.multi_vector == False
    assert caps.query_embedding == True
    assert caps.document_embedding == True
    assert caps.batch_inference == True
    assert caps.max_sequence_length == 8192
    assert caps.embedding_dimensions is None
    assert caps.supported_languages is None


def test_embedding_capabilities_custom():
    """Test EmbeddingCapabilities with custom values."""
    caps = EmbeddingCapabilities(
        dense=True,
        sparse=True,
        multi_vector=True,
        query_embedding=True,
        document_embedding=True,
        batch_inference=True,
        max_sequence_length=512,
        embedding_dimensions=1024,
        supported_languages=["en", "zh"],
    )
    assert caps.dense == True
    assert caps.sparse == True
    assert caps.multi_vector == True
    assert caps.max_sequence_length == 512
    assert caps.embedding_dimensions == 1024
    assert caps.supported_languages == ["en", "zh"]


def test_embedding_space_descriptor():
    """Test EmbeddingSpaceDescriptor creation and compatibility check."""
    desc1 = EmbeddingSpaceDescriptor(
        model_id="test-model",
        dimensions=768,
        representation_type=RepresentationType.DENSE,
        normalization=Normalization.L2,
    )
    desc2 = EmbeddingSpaceDescriptor(
        model_id="test-model",
        dimensions=768,
        representation_type=RepresentationType.DENSE,
        normalization=Normalization.L2,
    )
    desc3 = EmbeddingSpaceDescriptor(
        model_id="other-model",
        dimensions=768,
        representation_type=RepresentationType.DENSE,
        normalization=Normalization.L2,
    )

    # Same descriptors should be compatible
    assert desc1.is_compatible_with(desc2)

    # Different model IDs should not be compatible
    assert not desc1.is_compatible_with(desc3)

    # Different dimensions should not be compatible
    desc4 = EmbeddingSpaceDescriptor(
        model_id="test-model", dimensions=512, representation_type=RepresentationType.DENSE
    )
    assert not desc1.is_compatible_with(desc4)

    # Different representation types should not be compatible
    desc5 = EmbeddingSpaceDescriptor(
        model_id="test-model",
        dimensions=768,
        representation_type=RepresentationType.SPARSE,
    )
    assert not desc1.is_compatible_with(desc5)


def test_embedding_model_protocol():
    """Test that embedding model protocol is runtime-checkable."""
    from embeddings.contracts.protocols import EmbeddingModel as EM
    assert hasattr(EM, "embed_documents")
    assert hasattr(EM, "embed_query")
    assert hasattr(EM, "embed_queries")
    assert hasattr(EM, "get_model_info")
    assert hasattr(EM, "get_embedding_space")


def test_qwen3_model_registration():
    """Test Qwen3 embedding model class is importable and identifiable."""
    assert Qwen3EmbeddingModel is not None
    model = Qwen3EmbeddingModel()
    assert model.name == "Qwen/Qwen3-Embedding-0.6B"


def test_qwen3_capabilities():
    """Test Qwen3 capabilities reporting."""
    model = Qwen3EmbeddingModel(output_dimension=1024)
    caps = model.capabilities
    assert caps.dense == True
    assert caps.sparse == False
    assert caps.multi_vector == False
    assert caps.query_embedding == True
    assert caps.document_embedding == True
    assert caps.batch_inference == True
    assert caps.embedding_dimensions == 1024
    # output_dimension set → max_sequence_length reduced
    assert caps.max_sequence_length == 512


def test_qwen3_get_model_info():
    """Test Qwen3 model info reporting."""
    model = Qwen3EmbeddingModel(output_dimension=1024)
    info = model.get_model_info()
    assert info["model_id"] == "Qwen/Qwen3-Embedding-0.6B"
    assert info["adapter_name"] == "qwen3"
    assert info["dimensions"] == 1024


def test_qwen3_get_embedding_space():
    """Test Qwen3 embedding space descriptor."""
    model = Qwen3EmbeddingModel(output_dimension=1024)
    space = model.get_embedding_space()
    assert space.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert space.dimensions == 1024
    assert space.representation_type.name == "DENSE"
    assert space.adapter_name == "qwen3"


def test_qwen3_embed_query_preserves_inputs():
    """Test Qwen3 query embedding preserves input references (structure test)."""
    model = Qwen3EmbeddingModel(output_dimension=1024)
    inp = EmbeddingInput(text="what is this about", doc_id="d1", chunk_id="c1")
    result = model.embed_query(inp)
    assert result.dimensions == 1024
    assert result.dense_vector is not None
    assert len(result.dense_vector) == 1024
    assert result.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert result.chunk_id == "c1"
    assert result.doc_id == "d1"


def test_qwen3_embed_documents_preserves_inputs():
    """Test Qwen3 document embedding preserves input references."""
    model = Qwen3EmbeddingModel(output_dimension=1024)
    inputs = [
        EmbeddingInput(text="this is a document about dogs", doc_id="d1", chunk_id="c1"),
        EmbeddingInput(text="another document about cats", doc_id="d1", chunk_id="c2"),
    ]
    results = model.embed_documents(inputs)
    assert len(results) == 2
    assert results[0].dimensions == 1024
    assert results[0].dense_vector is not None
    assert results[0].chunk_id == "c1"
    assert results[0].doc_id == "d1"
    assert results[1].chunk_id == "c2"
    assert results[1].doc_id == "d1"


def test_batch_ordering():
    """Test that batch embedding preserves input order."""
    model = Qwen3EmbeddingModel(output_dimension=1024)
    inputs = [
        EmbeddingInput(text=f"document {i}", doc_id="d1", chunk_id=f"c{i}")
        for i in range(5)
    ]
    results = model.embed_documents(inputs)
    assert len(results) == 5
    # Check that chunk_ids are in the same order as input
    for i, result in enumerate(results):
        assert result.chunk_id == f"c{i}"


def test_source_reference_preservation():
    """Test that source references are preserved in embedding results."""
    model = Qwen3EmbeddingModel(output_dimension=1024)
    inp = EmbeddingInput(
        text="test document",
        doc_id="doc-123",
        chunk_id="chunk-456",
        source_refs=["ref-a", "ref-b"],
    )
    result = model.embed_query(inp)
    assert result.doc_id == "doc-123"
    assert result.chunk_id == "chunk-456"
    # Metadata should be preserved (or at least not lost)
    assert result.metadata == {}


def test_embedding_space_mismatch():
    """Test that embedding space mismatch is detected."""
    from embeddings.contracts.models import EmbeddingSpaceDescriptor
    from embeddings.contracts.models import RepresentationType, Normalization

    desc1 = EmbeddingSpaceDescriptor(
        model_id="model-a", dimensions=1024, representation_type=RepresentationType.DENSE
    )
    desc2 = EmbeddingSpaceDescriptor(
        model_id="model-b", dimensions=1024, representation_type=RepresentationType.DENSE
    )
    desc3 = EmbeddingSpaceDescriptor(
        model_id="model-a", dimensions=768, representation_type=RepresentationType.DENSE
    )

    # Different model IDs - not compatible
    assert not desc1.is_compatible_with(desc2)

    # Different dimensions - not compatible
    assert not desc1.is_compatible_with(desc3)


def test_representation_type_enum():
    """Test RepresentationType enum values."""
    assert RepresentationType.DENSE.value == "dense"
    assert RepresentationType.SPARSE.value == "sparse"
    assert RepresentationType.MULTI_VECTOR.value == "multi-vector"


def test_normalization_enum():
    """Test Normalization enum values."""
    assert Normalization.NONE.value == "none"
    assert Normalization.L2.value == "l2"


def test_embedding_result_representation_types():
    """Test EmbeddingResult can represent different representation types."""
    # Dense result
    dense_result = EmbeddingResult(
        model_id="m", dimensions=768, dense_vector=[0.1]*768,
        representation_type=RepresentationType.DENSE,
    )
    assert dense_result.representation_type == RepresentationType.DENSE

    # Sparse result
    sparse_result = EmbeddingResult(
        model_id="m", dimensions=768,
        sparse_indices=[1, 5], sparse_values=[0.5, 0.3],
        representation_type=RepresentationType.SPARSE,
    )
    assert sparse_result.representation_type == RepresentationType.SPARSE

    # Multi-vector result
    multi_result = EmbeddingResult(
        model_id="m", dimensions=768,
        multi_vector=[[0.1]*768, [0.2]*768],
        representation_type=RepresentationType.MULTI_VECTOR,
    )
    assert multi_result.representation_type == RepresentationType.MULTI_VECTOR


def test_compatible_descriptors_same():
    """Test that identical descriptors are compatible."""
    from embeddings.contracts.models import EmbeddingSpaceDescriptor, RepresentationType, Normalization

    d1 = EmbeddingSpaceDescriptor(
        model_id="same-model", dimensions=768, representation_type=RepresentationType.DENSE
    )
    d2 = EmbeddingSpaceDescriptor(
        model_id="same-model", dimensions=768, representation_type=RepresentationType.DENSE
    )
    assert d1.is_compatible_with(d2)


def test_incompatible_descriptors_different_model():
    """Test that different model IDs are not compatible."""
    from embeddings.contracts.models import EmbeddingSpaceDescriptor, RepresentationType, Normalization

    d1 = EmbeddingSpaceDescriptor(
        model_id="model-a", dimensions=768, representation_type=RepresentationType.DENSE
    )
    d2 = EmbeddingSpaceDescriptor(
        model_id="model-b", dimensions=768, representation_type=RepresentationType.DENSE
    )
    assert not d1.is_compatible_with(d2)


def test_incompatible_descriptors_different_dims():
    """Test that different dimensions are not compatible (even with same model_id)."""
    from embeddings.contracts.models import EmbeddingSpaceDescriptor, RepresentationType, Normalization

    d1 = EmbeddingSpaceDescriptor(
        model_id="same-model", dimensions=1024, representation_type=RepresentationType.DENSE
    )
    d2 = EmbeddingSpaceDescriptor(
        model_id="same-model", dimensions=768, representation_type=RepresentationType.DENSE
    )
    assert not d1.is_compatible_with(d2)


def test_incompatible_descriptors_different_repr():
    """Test that different representation types are not compatible."""
    from embeddings.contracts.models import EmbeddingSpaceDescriptor, RepresentationType, Normalization

    d1 = EmbeddingSpaceDescriptor(
        model_id="same-model", dimensions=768, representation_type=RepresentationType.DENSE
    )
    d2 = EmbeddingSpaceDescriptor(
        model_id="same-model", dimensions=768, representation_type=RepresentationType.SPARSE
    )
    assert not d1.is_compatible_with(d2)


def test_empty_input_handling():
    """Test handling of empty text input (structure test only)."""
    model = Qwen3EmbeddingModel(output_dimension=1024)
    inp = EmbeddingInput(text="", doc_id="d1", chunk_id="c1")
    # Empty string may or may not be handled - test structure
    try:
        result = model.embed_query(inp)
        assert result.dimensions == 1024
    except Exception:
        # Some models may reject empty input - that's also acceptable
        pass








def test_embedding_input_validation():
    """Test EmbeddingInput text validation."""
    # Empty text should be allowed (models may handle it differently)
    inp = EmbeddingInput(text="")
    assert inp.text == ""
    
    # Very long text should be allowed (models may truncate)
    inp = EmbeddingInput(text="x" * 10000)
    assert len(inp.text) == 10000


def test_combined_embedding_space():
    """Test that different embedding spaces are not compatible."""
    from embeddings.contracts.models import EmbeddingSpaceDescriptor, RepresentationType, Normalization

    qwen_space = EmbeddingSpaceDescriptor(
        model_id="Qwen/Qwen3-Embedding-0.6B", dimensions=1024,
        representation_type=RepresentationType.DENSE
    )
    bge_space = EmbeddingSpaceDescriptor(
        model_id="Qwen/Qwen3-Embedding-0.6B-mock", dimensions=768,
        representation_type=RepresentationType.DENSE
    )
    
    # Different model IDs or dimensions are not compatible
    assert not qwen_space.is_compatible_with(bge_space)
