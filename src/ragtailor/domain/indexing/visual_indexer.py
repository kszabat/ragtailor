from __future__ import annotations

from qdrant_client import QdrantClient, models

from ragtailor.domain.embeddings.base import VisualEmbedder
from ragtailor.domain.models import VisualPage


def index_visual_pages(
    client: QdrantClient,
    collection_name: str,
    file_path: str,
    pages: list[VisualPage],
    visual_embedder: VisualEmbedder,
) -> None:
    if not pages:
        return

    image_paths = [p.image_path for p in pages]
    page_vectors = visual_embedder.embed_pages(image_paths)

    points = [
        models.PointStruct(
            id=page.id,
            vector=page_vectors[i],
            payload={
                "file_id": page.file_id,
                "file_path": file_path,
                "page_number": page.page_number,
                "image_path": page.image_path,
            },
        )
        for i, page in enumerate(pages)
    ]

    client.upsert(collection_name=collection_name, points=points)
