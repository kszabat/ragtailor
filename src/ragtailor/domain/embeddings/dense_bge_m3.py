from __future__ import annotations

from ragtailor.domain.embeddings.base import DenseEmbedder


class BgeM3DenseEmbedder(DenseEmbedder):
    dimension = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = False) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(
            sentences=texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense_vectors = output["dense_vecs"].tolist()

        return dense_vectors
