from __future__ import annotations

from qdrant_client import QdrantClient, models

from ragtailor.domain.embeddings.base import DenseEmbedder, SparseEmbedder
from ragtailor.domain.models import TextChunk, TextSearchMode


def index_text_chunks(
    client: QdrantClient,
    collection_name: str,
    mode: TextSearchMode,
    file_path: str,
    chunks: list[TextChunk],
    dense_embedder: DenseEmbedder | None = None,
    sparse_embedder: SparseEmbedder | None = None,
) -> None:
    if not chunks:
        return

    texts = [c.text for c in chunks]
    dense_vectors = (
        dense_embedder.embed_documents(texts)
        if mode in (TextSearchMode.DENSE, TextSearchMode.HYBRID)
        else None
    )
    sparse_vectors = (
        sparse_embedder.embed_documents(texts)
        if mode in (TextSearchMode.SPARSE, TextSearchMode.HYBRID)
        else None
    )

    points: list[models.PointStruct] = []
    for i, chunk in enumerate(chunks):
        vector: dict[str, object] = {}
        if dense_vectors is not None:
            vector["dense"] = dense_vectors[i]
        if sparse_vectors is not None:
            vector["sparse"] = models.SparseVector(
                indices=sparse_vectors[i].indices, values=sparse_vectors[i].values
            )

        point = models.PointStruct(
            id=chunk.id,
            vector=vector,
            payload={
                "file_id": chunk.file_id,
                "file_path": file_path,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "text": chunk.text,
            },
        )

        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
