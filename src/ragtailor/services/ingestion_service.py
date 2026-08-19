from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient

from ragtailor.domain.embeddings.base import (
    DenseEmbedder,
    SparseEmbedder,
    VisualEmbedder,
)
from ragtailor.domain.indexing.qdrant_client import delete_points_for_file
from ragtailor.domain.indexing.text_indexer import index_text_chunks
from ragtailor.domain.indexing.visual_indexer import index_visual_pages
from ragtailor.domain.ingestion.chunker import chunk_docling_json
from ragtailor.domain.ingestion.extractor import PdfExtractor
from ragtailor.domain.ingestion.scanner import ScanResult, scan_folder
from ragtailor.domain.metadata.store import MetadataStore
from ragtailor.domain.models import (
    CollectionConfig,
    FileRecord,
    FileStatus,
    TextChunk,
    VisualPage,
)


class IngestionService:
    def __init__(
        self,
        store: MetadataStore,
        qdrant: QdrantClient,
        extractor: PdfExtractor,
        docling_json_dir: Path,
        page_images_dir: Path,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        visual_embedder: VisualEmbedder | None = None,
        chunk_fn: Callable[[Path, str], list[TextChunk]] = chunk_docling_json,
    ) -> None:
        self._store = store
        self._qdrant = qdrant
        self._extractor = extractor
        self._docling_json_dir = docling_json_dir
        self._page_images_dir = page_images_dir
        self._dense_embedder = dense_embedder
        self._sparse_embedder = sparse_embedder
        self._visual_embedder = visual_embedder
        self._chunk_fn = chunk_fn

    def sync_and_ingest(self, config: CollectionConfig) -> ScanResult:
        scan_result = scan_folder(config.id, Path(config.source_folder), self._store)
        for file_record in scan_result.needs_processing:
            self.ingest_file(config, file_record)
        return scan_result

    def ingest_file(self, config: CollectionConfig, file_record: FileRecord) -> None:
        self._store.update_file_status(file_record.id, FileStatus.PROCESSING)
        try:
            pdf_path = str(Path(config.source_folder) / file_record.file_path)
            extraction = self._extractor.extract(
                pdf_path=pdf_path,
                file_id=file_record.id,
                json_output_dir=self._docling_json_dir,
                image_output_dir=self._page_images_dir
                if config.visual_enabled
                else None,
            )

            num_chunks: int | None = None
            num_pages: int | None = None

            if config.text_enabled:
                assert config.qdrant_text_collection
                delete_points_for_file(
                    self._qdrant, config.qdrant_text_collection, file_record.id
                )
                chunks = self._chunk_fn(extraction.json_path, file_record.id)
                index_text_chunks(
                    self._qdrant,
                    config.qdrant_text_collection,
                    config.text_search_mode,
                    file_record.file_path,
                    chunks,
                    self._dense_embedder,
                    self._sparse_embedder,
                )
                num_chunks = len(chunks)

            if config.visual_enabled:
                assert config.qdrant_visual_collection
                delete_points_for_file(
                    self._qdrant, config.qdrant_visual_collection, file_record.id
                )
                visual_pages = [
                    VisualPage(
                        file_id=file_record.id,
                        page_number=page_no,
                        image_path=str(path),
                    )
                    for page_no, path in extraction.page_image_paths.items()
                ]
                index_visual_pages(
                    self._qdrant,
                    config.qdrant_visual_collection,
                    file_record.file_path,
                    visual_pages,
                    self._visual_embedder,
                )
                num_pages = len(visual_pages)

            self._store.update_file_status(
                file_record.id,
                FileStatus.INDEXED,
                num_text_chunks=num_chunks,
                num_pages=num_pages,
            )
        except Exception as exc:
            self._store.update_file_status(
                file_record.id, FileStatus.ERROR, error_message=str(exc)
            )
