from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial

from pydantic import BaseModel, Field, model_validator


class TextSearchMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class FileStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    ERROR = "error"
    MISSING = "missing"


class SourceType(StrEnum):
    TEXT = "text"
    VISUAL = "visual"


class EmbeddingStyle(StrEnum):
    SINGLE_VECTOR = "single_vector"
    MULTI_VECTOR = "multi_vector"


def _generate_id() -> str:
    id = uuid.uuid4()
    return str(id)


class CollectionConfig(BaseModel):
    id: str = Field(default_factory=_generate_id)
    name: str
    source_folder: str

    text_enabled: bool = False
    text_search_mode: TextSearchMode | None = None
    dense_model: str | None = None
    dense_style: EmbeddingStyle | None = None
    sparse_model: str | None = None

    visual_enabled: bool = False
    visual_model: str | None = None
    visual_style: EmbeddingStyle | None = None

    qdrant_text_collection: str | None = None
    qdrant_visual_collection: str | None = None

    created_at: datetime = Field(default_factory=partial(datetime.now, UTC))
    updated_at: datetime = Field(default_factory=partial(datetime.now, UTC))

    @model_validator(mode="after")
    def _validate_mode_selection(self) -> CollectionConfig:

        if not self.text_enabled and not self.visual_enabled:
            raise ValueError(
                "At least one of text_enabled or visual_enabled must be True."
            )

        if self.text_enabled and self.text_search_mode is None:
            raise ValueError(
                "text_search_mode must be specified when text_enabled is True."
            )

        if (
            self.text_enabled
            and self.text_search_mode
            in (
                TextSearchMode.DENSE,
                TextSearchMode.HYBRID,
            )
            and self.dense_style is None
        ):
            raise ValueError(
                "dense_style must be specified when text_search_mode is DENSE or HYBRID."
            )

        if self.visual_enabled and self.visual_style is None:
            raise ValueError(
                "visual_style must be specified when visual_enabled is True."
            )

        return self

    @classmethod
    def create(
        cls,
        name: str,
        source_folder: str,
        text_enabled: bool,
        visual_enabled: bool,
        text_search_mode: TextSearchMode | None = None,
        dense_model: str | None = None,
        dense_style: EmbeddingStyle | None = None,
        sparse_model: str | None = None,
        visual_model: str | None = None,
        visual_style: EmbeddingStyle | None = None,
    ) -> CollectionConfig:
        collection_id = _generate_id()

        c = cls(
            id=collection_id,
            name=name,
            source_folder=source_folder,
            text_enabled=text_enabled,
            text_search_mode=text_search_mode,
            dense_model=dense_model,
            dense_style=dense_style,
            sparse_model=sparse_model,
            visual_enabled=visual_enabled,
            visual_model=visual_model,
            visual_style=visual_style,
            qdrant_text_collection=f"{collection_id}_text" if text_enabled else None,
            qdrant_visual_collection=(
                f"{collection_id}_visual" if visual_enabled else None
            ),
        )

        return c


class FileRecord(BaseModel):
    id: str = Field(default_factory=_generate_id)
    collection_id: str
    file_path: str
    file_hash: str
    file_size: int
    status: FileStatus = FileStatus.PENDING
    error_message: str | None = None
    num_text_chunks: int | None = None
    num_pages: int | None = None
    added_at: datetime = Field(default_factory=partial(datetime.now, UTC))
    last_indexed_at: datetime | None = None


class TextChunk(BaseModel):
    id: str = Field(default_factory=_generate_id)
    file_id: str
    chunk_index: int
    page_number: int | None = None
    text: str


class VisualPage(BaseModel):
    id: str = Field(default_factory=_generate_id)
    file_id: str
    page_number: int
    image_path: str


class SearchOptions(BaseModel):
    top_k_search: int = 100
    use_text: bool = True
    use_visual: bool = True
    fuse_results: bool = False
    rerank_enabled: bool = False
    top_k_rerank: int = 20


class SearchResult(BaseModel):
    id: str
    source_type: SourceType
    score: float
    file_id: str
    file_path: str
    page_number: int | None = None
    text: str | None = None
    image_path: str | None = None
    reranker_score: float | None = None
