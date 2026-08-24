from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class SparseEmbedding:
    indices: list[int]
    values: list[float]

    def __post_init__(self):
        if len(self.indices) != len(self.values):
            raise ValueError("Indices and values must have the same length.")


class SparseEmbedder(Protocol):
    def embed_query(self, text: str) -> SparseEmbedding: ...
    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]: ...


class TextSingleVectorEmbedder(Protocol):
    """Embeddings that return a single vector per text."""

    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


class TextMultiVectorEmbedder(Protocol):
    """Late interaction / ColBERT-style - embeddings that return multiple vectors per text."""

    vector_dimension: int  # size of single vector in the multi-vector embedding

    def embed_texts(self, texts: list[str]) -> list[list[list[float]]]: ...
    def embed_query(self, query: str) -> list[list[float]]: ...


class VisualSingleVectorEmbedder(Protocol):
    """Embeddings that return a single vector per image."""

    dimension: int

    def embed_images(self, image_paths: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


class VisualMultiVectorEmbedder(Protocol):
    """Embeddings that return multiple vectors per image."""

    vector_dimension: int  # size of single vector in the multi-vector embedding

    def embed_images(self, image_paths: list[str]) -> list[list[list[float]]]: ...
    def embed_query(self, query: str) -> list[list[float]]: ...
