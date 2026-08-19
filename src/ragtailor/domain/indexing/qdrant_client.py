from __future__ import annotations

from collections import defaultdict

from qdrant_client import QdrantClient, models

from ragtailor.config import Settings
from ragtailor.domain.models import CollectionConfig, TextSearchMode

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"

KNOWN_DENSE_DIMS = defaultdict(lambda: 1024, {"BAAI/bge-m3": 1024})
KNOWN_VISUAL_DIMS = defaultdict(lambda: 128, {"vidore/colqwen2.5-v0.2": 128})


def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=Settings.qdrant_url, api_key=Settings.qdrant_api_key)


def ensure_collections(
    client: QdrantClient,
    config: CollectionConfig,
    dense_vector_size: int | None = None,
    visual_vector_size: int | None = None,
):
    if config.text_enabled:
        assert config.qdrant_text_collection is not None, (
            "qdrant_text_collection must be set if text_enabled is True"
        )
        _ensure_text_collection(
            client,
            config.qdrant_text_collection,
            config.text_search_mode,
            dense_vector_size or KNOWN_DENSE_DIMS[config.dense_model],
        )

    if config.visual_enabled:
        assert config.qdrant_visual_collection is not None, (
            "qdrant_visual_collection must be set if visual_enabled is True"
        )
        _ensure_visual_collection(
            client,
            config.qdrant_visual_collection,
            visual_vector_size or KNOWN_VISUAL_DIMS[config.visual_model],
        )


def _ensure_text_collection(
    client: QdrantClient,
    name: str,
    mode: TextSearchMode | None,
    dense_size: int,
) -> None:
    if client.collection_exists(name):
        return

    vectors_config: dict[str, models.VectorParams] = {}
    sparse_vectors_config: dict[str, models.SparseVectorParams] = {}

    if mode in (TextSearchMode.DENSE, TextSearchMode.HYBRID):
        vectors_config[_DENSE_VECTOR_NAME] = models.VectorParams(
            size=dense_size, distance=models.Distance.COSINE
        )
    if mode in (TextSearchMode.SPARSE, TextSearchMode.HYBRID):
        sparse_vectors_config[_SPARSE_VECTOR_NAME] = models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))

    client.create_collection(
        collection_name=name,
        vectors_config=vectors_config or {},
        sparse_vectors_config=sparse_vectors_config or None,
    )


def _ensure_visual_collection(
    client: QdrantClient, name: str, patch_vector_size: int
) -> None:
    if client.collection_exists:
        return

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=patch_vector_size,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
        ),
    )


def delete_collections(client: QdrantClient, config: CollectionConfig) -> None:
    collection_names = tuple(
        config.qdrant_text_collection, config.qdrant_visual_collection
    )

    for name in collection_names:
        if name and client.collection_exists:
            client.delete_collection(name)


def delete_points_for_file(
    client: QdrantClient, collection_name: str, file_id: str
) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_id", match=models.MatchValue(value=file_id)
                    )
                ]
            )
        ),
    )
