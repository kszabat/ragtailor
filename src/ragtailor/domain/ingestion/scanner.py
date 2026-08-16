from __future__ import annotations

from hashlib import sha256
from dataclasses import dataclass, field
from pathlib import Path

from ragtailor.domain.metadata.store import MetadataStore
from ragtailor.domain.models import FileRecord, FileStatus

_HASH_CHUNK_SIZE = 1024 * 1024


def compute_hash(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


@dataclass
class ScanResult:
    new_files: list[FileRecord] = field(default_factory=list)
    changed_files: list[FileRecord] = field(default_factory=list)
    missing_files: list[FileRecord] = field(default_factory=list)
    reappeared_files: list[FileRecord] = field(default_factory=list)
    unchanged_count: int = 0

    @property
    def to_be_processed(self) -> list[FileRecord]:
        return self.new_files + self.changed_files


def scan_folder(collection_id: str, source_folder: Path, store: MetadataStore):
    disk_files = {
        path.relative_to(source_folder).as_posix(): path
        for path in sorted(source_folder.rglob("*"))
        if path.is_file() and path.suffix.lower() == ".pdf"
    }

    registered_files = {file.file_path: file for file in store.list_files()}

    result = ScanResult()

    for rel_path, abs_path in disk_files.items():
        file_hash = compute_hash(abs_path)
        file_size = abs_path.stat().st_size

        existing_file = registered_files.get(rel_path)

        if existing_file is None:
            record = FileRecord(
                collection_id=collection_id,
                file_path=rel_path,
                file_hash=file_hash,
                file_size=file_size,
                status=FileStatus.PENDING,
            )

            store.upsert_file(record)
            result.new_files.append(record)

            continue

        if existing_file.file_hash != file_hash:
            record = existing_file.model_copy(
                update={
                    "file_hash": file_hash,
                    "file_size": file_size,
                    "status": FileStatus.PENDING,
                }
            )

            store.upsert_file(record)
            result.changed_files.append(record)

            continue

        if existing_file.status == FileStatus.MISSING:
            restored_status = (
                FileStatus.INDEXED
                if existing_file.num_text_chunks
                else FileStatus.PENDING
            )

            record = existing_file.model_copy(update={"status": restored_status})
            store.upsert_file(record)
            result.reappeared_files.append(record)

            if restored_status == FileStatus.PENDING:
                result.new_files.append(record)

            continue

        result.unchanged_count += 1

    for rel_path, abs_path in registered_files.items():
        if rel_path in disk_files:
            continue

        if existing_file.status == FileStatus.MISSING:
            result.missing_files.append(existing_file)

            continue

        record = existing_file.model_copy(update={"status": FileStatus.MISSING})
        store.upsert_file(record)
        result.missing_files.append(record)

    return result
