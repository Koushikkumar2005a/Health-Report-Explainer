"""Shared multilingual E5 embeddings for indexing and retrieval."""

from __future__ import annotations

import threading
from collections.abc import Sequence

MODEL_NAME = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
EMBEDDING_DIMENSION = 768

_model = None
_model_lock = threading.Lock()


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding model cannot produce valid vectors."""


def get_embedding_model():
    """Load the sentence-transformer once per process."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise EmbeddingError(
                        "sentence-transformers is required for medical reference retrieval."
                    ) from exc

                _model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
    return _model


def _encode(texts: Sequence[str], prefix: str) -> list[list[float]]:
    if not texts:
        return []

    try:
        import numpy as np
    except ImportError as exc:
        raise EmbeddingError("numpy is required for medical reference embeddings.") from exc

    model = get_embedding_model()
    vectors = np.asarray(
        model.encode(
            [f"{prefix}{text}" for text in texts],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.shape != (len(texts), EMBEDDING_DIMENSION):
        raise EmbeddingError(
            "The embedding model returned vectors with an unexpected dimension."
        )
    if not np.isfinite(vectors).all():
        raise EmbeddingError("The embedding model returned a non-finite vector.")

    return vectors.tolist()


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed knowledge-base passages with the E5 passage prefix."""
    return _encode(texts, "passage: ")


def embed_queries(queries: Sequence[str]) -> list[list[float]]:
    """Embed retrieval queries with the E5 query prefix."""
    return _encode(queries, "query: ")


class E5EmbeddingFunction:
    """Chroma-compatible embedding function backed by the shared E5 model."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "multilingual_e5_base"

    @staticmethod
    def build_from_config(config: dict) -> "E5EmbeddingFunction":
        del config
        return E5EmbeddingFunction()

    def get_config(self) -> dict:
        return {}

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine"]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return embed_documents(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return embed_queries(input)
