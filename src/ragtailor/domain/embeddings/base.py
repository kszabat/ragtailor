from dataclasses import dataclass
from typing import Protocol

from __future__ import annotations


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


class DenseEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class VisualEmbedder(Protocol):
    patch_dimension: int

    def embed_query(self, text: str) -> list[list[float]]: ...
    def embed_pages(self, image_paths: list[str]) -> list[list[list[float]]]: ...
