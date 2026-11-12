"""Qwen3-Embedding-0.6B embedding adapter.

General-purpose all-rounder embedding model.
Model: Qwen/Qwen3-Embedding-0.6B
"""

from __future__ import annotations

import logging
from typing import Any

from embeddings.contracts.models import EmbeddingInput, EmbeddingResult
from embeddings.contracts.protocols import EmbeddingModel
from embeddings.contracts.models import EmbeddingCapabilities, EmbeddingSpaceDescriptor, RepresentationType, Normalization

logger = logging.getLogger(__name__)


def _import_sentence_transformers():
    """Import sentence-transformers, raising a clear error if unavailable."""
    try:
        import sentence_transformers

        return sentence_transformers
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for Qwen3-Embedding-0.6B. "
            "Install it with: pip install sentence-transformers"
        ) from e


class Qwen3EmbeddingModel(EmbeddingModel):
    """Qwen3-Embedding-0.6B embedding adapter.

    General-purpose all-rounder supporting document and query encoding,
    configurable output dimensions, batch inference, and normalization.
    """

    name: str = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        output_dimension: int | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ):
        """Initialize the Qwen3-Embedding-0.6B adapter.

        Args:
            model_name: HuggingFace model name.
            output_dimension: Desired output dimension. Must be one of the
                model's supported options (None means default 1024).
            device: Computing device ('cuda', 'mps', or 'cpu'). None means
                auto-detect.
            batch_size: Batch size for encoding.
        """
        self.model_name = model_name
        self.output_dimension = output_dimension
        self.device = device or self._detect_device()
        self.batch_size = batch_size
        self._model = None
        self._capabilities: EmbeddingCapabilities | None = None

    @staticmethod
    def _detect_device() -> str:
        """Detect the best available device."""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def _loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None

    def _ensure_loaded(self) -> None:
        """Lazy-load the model if not already loaded."""
        if self._loaded:
            return
        try:
            st = _import_sentence_transformers()
            self._model = st.SentenceTransformer(self.model_name)
            logger.info("Qwen3-Embedding-0.6B model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Qwen3-Embedding-0.6B: {e}")
            self._model = None
            raise

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """Return the capabilities of this model."""
        if self._capabilities is not None:
            return self._capabilities

        dim = self.output_dimension or 1024
        caps = EmbeddingCapabilities(
            dense=True,
            sparse=False,
            multi_vector=False,
            query_embedding=True,
            document_embedding=True,
            batch_inference=True,
            max_sequence_length=8192,
            embedding_dimensions=dim,
            supported_languages=["en", "zh", "code"],
        )

        if self.output_dimension is not None:
            caps.max_sequence_length = min(caps.max_sequence_length, 512)

        self._capabilities = caps
        return caps

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the model."""
        if not self._loaded:
            self._ensure_loaded()

        return {
            "model_id": self.model_name,
            "adapter_name": "qwen3",
            "dimensions": self.capabilities.embedding_dimensions,
            "representation_type": "dense",
            "normalization": "l2",
            "max_sequence_length": self.capabilities.max_sequence_length,
            "supported_languages": self.capabilities.supported_languages,
            "output_dimension": self.output_dimension,
        }

    def get_embedding_space(self) -> EmbeddingSpaceDescriptor:
        """Return the embedding space descriptor."""
        caps = self.capabilities
        return EmbeddingSpaceDescriptor(
            model_id=self.model_name,
            model_revision=None,
            adapter_name="qwen3",
            dimensions=caps.embedding_dimensions,
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            query_instruction=None,
            document_instruction=None,
            supported_languages=caps.supported_languages,
        )

    def embed_documents(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed a list of document inputs.

        Each input's source_doc_id and chunk_id are retained in the result.
        """
        if not self._loaded:
            self._ensure_loaded()

        texts = [inp.text for inp in inputs]
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        results = []
        for i, (inp, vec) in enumerate(zip(inputs, embeddings)):
            result = EmbeddingResult(
                model_id=self.model_name,
                dimensions=len(vec),
                dense_vector=vec.tolist(),
                representation_type=RepresentationType.DENSE,
                normalization=Normalization.L2,
                metadata={},
                doc_id=inp.doc_id,
                chunk_id=inp.chunk_id,
                query_id=None,
            )
            results.append(result)

        return results

    def embed_query(self, input: EmbeddingInput) -> EmbeddingResult:
        """Embed a single query input.

        Query embedding may use different instructions or prefixes than
        document embedding.
        """
        if not self._loaded:
            self._ensure_loaded()

        texts = [input.text]
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vec = embeddings[0]
        result = EmbeddingResult(
            model_id=self.model_name,
            dimensions=len(vec),
            dense_vector=vec.tolist(),
            representation_type=RepresentationType.DENSE,
            normalization=Normalization.L2,
            metadata={},
            doc_id=input.doc_id,
            chunk_id=input.chunk_id,
            query_id=input.doc_id or input.chunk_id,
        )
        return result

    def embed_queries(self, inputs: list[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed a list of query inputs.

        Each query may use different instructions or prefixes than document
        embedding.
        """
        if not self._loaded:
            self._ensure_loaded()

        texts = [inp.text for inp in inputs]
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        results = []
        for i, (inp, vec) in enumerate(zip(inputs, embeddings)):
            result = EmbeddingResult(
                model_id=self.model_name,
                dimensions=len(vec),
                dense_vector=vec.tolist(),
                representation_type=RepresentationType.DENSE,
                normalization=Normalization.L2,
                metadata={},
                doc_id=inp.doc_id,
                chunk_id=inp.chunk_id,
                query_id=None,
            )
            results.append(result)

        return results