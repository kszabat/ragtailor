from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, DoclingDocument
from docling_core.types.doc.labels import DocItemLabel
from pydantic import BaseModel


from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    json_path: Path
    page_image_paths: dict[int, Path] = field(default_factory=dict)


class PdfExtractor(Protocol):
    def extract(
        self, pdf_path: str, file_id: str, json_output_dir: Path, image_output_dir: Path
    ) -> ExtractionResult: ...


class DoclingExtractor(PdfExtractor):
    def __init__(
        self,
        do_ocr: bool = False,
        do_table_structure: bool = True,
        images_scale: float = 2.0,
    ) -> None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
        pipeline_options.images_scale = images_scale
        pipeline_options.generate_picture_images = False  # może zmienić
        pipeline_options.generate_page_images = True  # zmienić

        self._extractor = DocumentConverter(
            allowed_formats=InputFormat.PDF,
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            },
        )

    def extract(
        self, pdf_path: str, file_id: str, json_output_dir: Path, image_output_dir: Path
    ) -> ExtractionResult:
        result = self._extractor.convert(pdf_path)
        document = result.document

        json_output_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_output_dir / f"{file_id}.json"

        # document.save_as_json(
        #     filename=json_path,
        #     artifacts_dir=image_output_dir / file_id,
        #     image_mode=ImageRefMode.REFERENCED,
        # )

        json_path.write_text(
            json.dumps(document.export_to_dict(), ensure_ascii=False, indent=2)
        )

        page_image_paths = _save_page_images(document, file_id, image_output_dir)

        return ExtractionResult(json_path=json_path, page_image_paths=page_image_paths)


def _save_page_images(
    document: DoclingDocument, file_id: str, image_output_dir: Path
) -> dict[int, Path]:
    output_dir = image_output_dir / file_id
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[int, Path] = {}
    for page_no, page in document.pages.items():
        page_no = page.page_no

        if page.image is None:
            continue

        image_path = output_dir / f"{page_no}.png"
        page.image.pil_image.save(image_path)
        paths[page_no] = image_path

    return paths

