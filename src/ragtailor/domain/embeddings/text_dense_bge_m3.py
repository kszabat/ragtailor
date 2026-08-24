from __future__ import annotations

from ragtailor.domain.embeddings.base import (
    TextMultiVectorEmbedder,
    TextSingleVectorEmbedder,
)


class BgeM3SingleVectorEmbedder(TextSingleVectorEmbedder):
    dimension = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = False,
        device: str | None = None,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=device)

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts(texts=[query])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        output = self._model.encode(
            sentences=texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense_vectors = output["dense_vecs"].tolist()

        return dense_vectors


class BgeM3MultiVectorEmbedder(TextMultiVectorEmbedder):
    vector_dimension = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = False,
        device: str | None = None,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=device)

    def embed_query(self, query: str) -> list[list[float]]:
        return self.embed_texts(texts=[query])[0]

    def embed_texts(self, texts: list[str]) -> list[list[list[float]]]:
        output = self._model.encode(
            sentences=texts,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )

        multivectors = output["colbert_vecs"].tolist()

        return multivectors
