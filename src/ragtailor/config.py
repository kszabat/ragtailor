from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_local_path: Path | None = Field(default=None)

    gemini_api_key: str | None = Field(default=None)

    data_dir: Path = Field(default=Path(__file__).resolve().parent.parent / "data")
    metadata_db_path: Path = Field(
        default=Path(__file__).resolve().parent.parent / "data" / "metadata.db"
    )

    default_dense_model: str = Field(default="BAAI/bge-m3")
    default_sparse_model: str = Field(default="naver/splade-v3")
    default_visual_model: str = Field(default="vidore/colpali-v1.3")
    default_reranker_model: str = Field(default="jinaai/jina-reranker-m0")

    # default_llm_model: str = Field(default="")

    @property
    def page_image_dir(self) -> Path:
        return self.data_dir / "page_images"


@lru_cache
def get_settings() -> Settings:
    return Settings()
