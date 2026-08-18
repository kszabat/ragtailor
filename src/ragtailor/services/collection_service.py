from __future__ import annotations

from qdrant_client import QdrantClient

from ragtailor.domain.indexing.qdrant_client import (
    delete_collections,
    delete_points_for_file,
    ensure_collections,
)
from ragtailor.domain.metadata.store import MetadataStore
from ragtailor.domain.models import CollectionConfig, TextSearchMode


class CollectionService:
    def __init__(self, store: MetadataStore, qdrant: QdrantClient) -> None:
        self._store = store
        self._qdrant = qdrant

    def create_collection(
        self,
        name: str,
        source_folder: str,
        text_enabled: bool,
        visual_enabled: bool,
        text_search_mode: TextSearchMode | None = None,
        dense_model: str | None = None,
        sparse_model: str | None = None,
        visual_model: str | None = None,
        dense_vector_size: int | None = None,
        visual_vector_size: int | None = None,
    ) -> CollectionConfig:
        config = CollectionConfig.create(
            name=name,
            source_folder=source_folder,
            text_enabled=text_enabled,
            visual_enabled=visual_enabled,
            text_search_mode=text_search_mode,
            dense_model=dense_model,
            sparse_model=sparse_model,
            visual_model=visual_model,
        )

        ensure_collections(self._qdrant, config, dense_vector_size, visual_vector_size)

        try:
            self._store.create_collection(config)
        except Exception:
            delete_collections(self._qdrant, config)
            raise

        return config

    def get_collection(self, collection_id: str) -> CollectionConfig:
        return self._store.get_collection(collection_id)

    def list_collections(self) -> list[CollectionConfig]:
        return self._store.list_collections()

    def delete_collection(self, collection_id: str) -> None:
        config = self._store.get_collection(collection_id)

        if config is None:
            return

        delete_collections(self._qdrant, config)
        self._store.delete_collection(collection_id)

    def delete_file(self, collection_id: str, file_id: str) -> None:
        config = self._store.get_collection(collection_id)

        if config is None:
            return

        if config.text_enabled and config.qdrant_text_collection:
            delete_points_for_file(self._qdrant, config.qdrant_text_collection, file_id)

        if config.visual_enabled and config.qdrant_visual_collection:
            delete_points_for_file(
                self._qdrant, config.qdrant_visual_collection, file_id
            )

        self._store.delete_file(file_id)
