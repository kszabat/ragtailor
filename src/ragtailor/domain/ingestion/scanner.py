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

    @property
    def to_be_processed(self) -> list[FileRecord]:
        return self.new_files + self.changed_files

    