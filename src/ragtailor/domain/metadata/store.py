from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ragtailor.domain.models import (
    CollectionConfig,
    FileRecord,
    FileStatus,
    TextSearchMode,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id                       TEXT PRIMARY KEY,
    name                     TEXT UNIQUE NOT NULL,
    source_folder            TEXT NOT NULL,
    text_enabled             INTEGER NOT NULL DEFAULT 0,
    text_search_mode         TEXT,
    dense_model              TEXT,
    sparse_model             TEXT,
    chunk_size               INTEGER,
    chunk_overlap            INTEGER,
    visual_enabled           INTEGER NOT NULL DEFAULT 0,
    visual_model             TEXT,
    qdrant_text_collection   TEXT,
    qdrant_visual_collection TEXT,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id                TEXT PRIMARY KEY,
    collection_id     TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    file_path         TEXT NOT NULL,
    file_hash         TEXT NOT NULL,
    file_size         INTEGER NOT NULL,
    status            TEXT NOT NULL,
    error_message     TEXT,
    num_text_chunks   INTEGER,
    num_pages         INTEGER,
    added_at          TEXT NOT NULL,
    last_indexed_at   TEXT,
    UNIQUE(collection_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_files_collection ON files(collection_id);
"""


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_collection(self, config: CollectionConfig) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO collections (
                    id, name, source_folder, text_enabled, text_search_mode,
                    dense_model, sparse_model, chunk_size, chunk_overlap,
                    visual_enabled, visual_model,
                    qdrant_text_collection, qdrant_visual_collection,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                _collection_to_row(config),
            )

    def get_collection(self, collection_id: str) -> CollectionConfig | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM collections WHERE id = ?", (collection_id,)
            ).fetchone()
        return _row_to_collection(row) if row else None

    def list_collections(self) -> list[CollectionConfig]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM collections ORDER BY created_at"
            ).fetchall()
        return [_row_to_collection(row) for row in rows]

    def update_collection(self, config: CollectionConfig) -> None:
        config = config.model_copy(update={"updated_at": datetime.now(UTC)})
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE collections SET
                    name=?, source_folder=?, text_enabled=?, text_search_mode=?,
                    dense_model=?, sparse_model=?, chunk_size=?, chunk_overlap=?,
                    visual_enabled=?, visual_model=?,
                    qdrant_text_collection=?, qdrant_visual_collection=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    config.name,
                    config.source_folder,
                    int(config.text_enabled),
                    config.text_search_mode.value if config.text_search_mode else None,
                    config.dense_model,
                    config.sparse_model,
                    int(config.visual_enabled),
                    config.visual_model,
                    config.qdrant_text_collection,
                    config.qdrant_visual_collection,
                    config.updated_at.isoformat(),
                    config.id,
                ),
            )

    def delete_collection(self, collection_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))

    def upsert_file(self, file: FileRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (
                    id, collection_id, file_path, file_hash, file_size,
                    status, error_message, num_text_chunks, num_pages,
                    added_at, last_indexed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(collection_id, file_path) DO UPDATE SET
                    file_hash=excluded.file_hash,
                    file_size=excluded.file_size,
                    status=excluded.status,
                    error_message=excluded.error_message,
                    num_text_chunks=excluded.num_text_chunks,
                    num_pages=excluded.num_pages,
                    last_indexed_at=excluded.last_indexed_at
                """,
                _file_to_row(file),
            )

    def get_file_by_path(self, collection_id: str, file_path: str) -> FileRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE collection_id = ? AND file_path = ?",
                (collection_id, file_path),
            ).fetchone()
        return _row_to_file(row) if row else None

    def list_files(self, collection_id: str) -> list[FileRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE collection_id = ? ORDER BY added_at",
                (collection_id,),
            ).fetchall()
        return [_row_to_file(row) for row in rows]

    def update_file_status(
        self,
        file_id: str,
        status: FileStatus,
        error_message: str | None = None,
        num_text_chunks: int | None = None,
        num_pages: int | None = None,
    ) -> None:
        last_indexed_at = (
            datetime.now(UTC).isoformat() if status == FileStatus.INDEXED else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE files SET
                    status=?, error_message=?, num_text_chunks=?, num_pages=?,
                    last_indexed_at=COALESCE(?, last_indexed_at)
                WHERE id=?
                """,
                (
                    status.value,
                    error_message,
                    num_text_chunks,
                    num_pages,
                    last_indexed_at,
                    file_id,
                ),
            )

    def delete_file(self, file_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))


def _collection_to_row(c: CollectionConfig) -> tuple:
    return (
        c.id,
        c.name,
        c.source_folder,
        int(c.text_enabled),
        c.text_search_mode.value if c.text_search_mode else None,
        c.dense_model,
        c.sparse_model,
        int(c.visual_enabled),
        c.visual_model,
        c.qdrant_text_collection,
        c.qdrant_visual_collection,
        c.created_at.isoformat(),
        c.updated_at.isoformat(),
    )


def _row_to_collection(row: sqlite3.Row) -> CollectionConfig:
    return CollectionConfig(
        id=row["id"],
        name=row["name"],
        source_folder=row["source_folder"],
        text_enabled=bool(row["text_enabled"]),
        text_search_mode=(
            TextSearchMode(row["text_search_mode"]) if row["text_search_mode"] else None
        ),
        dense_model=row["dense_model"],
        sparse_model=row["sparse_model"],
        visual_enabled=bool(row["visual_enabled"]),
        visual_model=row["visual_model"],
        qdrant_text_collection=row["qdrant_text_collection"],
        qdrant_visual_collection=row["qdrant_visual_collection"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _file_to_row(f: FileRecord) -> tuple:
    return (
        f.id,
        f.collection_id,
        f.file_path,
        f.file_hash,
        f.file_size,
        f.status.value,
        f.error_message,
        f.num_text_chunks,
        f.num_pages,
        f.added_at.isoformat(),
        f.last_indexed_at.isoformat() if f.last_indexed_at else None,
    )


def _row_to_file(row: sqlite3.Row) -> FileRecord:
    return FileRecord(
        id=row["id"],
        collection_id=row["collection_id"],
        file_path=row["file_path"],
        file_hash=row["file_hash"],
        file_size=row["file_size"],
        status=FileStatus(row["status"]),
        error_message=row["error_message"],
        num_text_chunks=row["num_text_chunks"],
        num_pages=row["num_pages"],
        added_at=datetime.fromisoformat(row["added_at"]),
        last_indexed_at=(
            datetime.fromisoformat(row["last_indexed_at"])
            if row["last_indexed_at"]
            else None
        ),
    )
