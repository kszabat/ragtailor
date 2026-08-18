from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol
from dataclasses import dataclass, field

from docling_core.types.doc import DoclingDocument


@dataclass
class ExtractionResult:
    json_path: Path
    page_image_paths: dict[int, Path] = field(default_factory=dict)


class PdfExtractor(Protocol):
    def extract(
        self,
        pdf_path: str,
        file_id: str,
        json_output_dir: Path,
        image_output_dir: Path | None = None,
    ) -> ExtractionResult: ...


class DoclingExtractor(PdfExtractor):
    def __init__(
        self,
        do_ocr: bool = False,
        do_table_structure: bool = True,
        generate_page_images: bool = False,
        images_scale: float = 2.0,
    ) -> None:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
        pipeline_options.images_scale = images_scale
        pipeline_options.generate_page_images = generate_page_images
        pipeline_options.generate_picture_images = False

        self._generate_page_images = generate_page_images
        self._extractor = DocumentConverter(
            allowed_formats=InputFormat.PDF,
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            },
        )

    def extract(
        self,
        pdf_path: str,
        file_id: str,
        json_output_dir: Path,
        image_output_dir: Path | None = None,
    ) -> ExtractionResult:
        if self._generate_page_images is True and image_output_dir is None:
            raise ValueError(
                "image_output_dir must be provided when generate_page_images is True"
            )

        result = self._extractor.convert(pdf_path)
        document = result.document

        json_output_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_output_dir / f"{file_id}.json"

        json_path.write_text(
            json.dumps(document.export_to_dict(), ensure_ascii=False, indent=2)
        )

        page_image_paths = (
            self._save_page_images(document, file_id, image_output_dir)
            if self._generate_page_images
            else {}
        )

        return ExtractionResult(json_path=json_path, page_image_paths=page_image_paths)

    @staticmethod
    def _save_page_images(
        document: DoclingDocument, file_id: str, image_output_dir: Path
    ) -> dict[int, Path]:
        output_dir = image_output_dir / file_id
        paths: dict[int, Path] = {}

        for page_no, page in document.pages.items():
            page_no = page.page_no

            if page.image is None:
                continue

            output_dir.mkdir(parents=True, exist_ok=True)

            image_path = output_dir / f"{page_no}.png"
            page.image.pil_image.save(image_path)
            paths[page_no] = image_path

        return paths
