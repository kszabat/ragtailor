from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.labels import DocItemLabel
import json

from __future__ import annotations

from pydantic import BaseModel

from typing import Protocol

from pathlib import Path

class ExtractedPage(BaseModel):
    page_numer: int
    text: str

class ExtractedDocument(BaseModel):
    pages: list[ExtractedPage]

class PdfExtractor(Protocol):
    def extract(self, pdf_path: str, file_id: str, output_dir: Path) -> Path:
        ...

IMAGE_SCALE = 2.0

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_SCALE
pipeline_options.do_table_structure = True 
pipeline_options.do_ocr = 
pipeline_options.generate_page_images

class DoclingExtractor(PdfExtractor):
    def __init__(self) -> None:
        self._extractor = DocumentConverter()

    def extract(self, pdf_path:str, file_id:str, output_dir: Path) -> Path:
        result = self._extractor.convert(pdf_path)
        document = result.document
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{file_id}.json"
        document.save_as_json(json_path, image_mode=ImageRefMode.EMBEDDED)

        return json_path

