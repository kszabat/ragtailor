from __future__ import annotations

from typing import TYPE_CHECKING

from ragtailor.domain.embeddings.base import SparseEmbedder
from ragtailor.domain.embeddings.base import SparseEmbedding as SparseVector

if TYPE_CHECKING:
    import torch


class SpladeSparseEmbedder(SparseEmbedder):
    def __init__(
        self, model_name: str = "naver/splade-v3", device: str | None = None
    ) -> None:
        from sentence_transformers import SparseEncoder

        self._model = SparseEncoder(model_name, device=device)

    def embed_query(self, text: str) -> SparseVector:
        embedding_tensor = self._model.encode_query(
            inputs=[text], convert_to_sparse_tensor=True
        )
        embedding_vector = _sparse_tensor_to_vectors(embedding_tensor)[0]
        
        return embedding_vector

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        embedding_tensors = self._model.encode_document(
            inputs=texts, convert_to_sparse_tensor=True
        )
        embedding_vectors = _sparse_tensor_to_vectors(embedding_tensors)

        return embedding_vectors


def _sparse_tensor_to_vectors(tensor: torch.Tensor) -> list[SparseVector]:
    tensor = tensor.coalesce()
    row_idx, col_idx = tensor.indices()
    values = tensor.values()

    per_row_indices: list[list[int]] = [[] for _ in range(tensor.shape[0])]
    per_row_values: list[list[float]] = [[] for _ in range(tensor.shape[0])]
    for row, col, val in zip(row_idx.tolist(), col_idx.tolist(), values.tolist()):
        per_row_indices[row].append(col)
        per_row_values[row].append(val)

    return [
        SparseVector(indices=idxs, values=vals)
        for idxs, vals in zip(per_row_indices, per_row_values)
    ]
