from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import BaseNode, MetadataMode

from ragtailor.domain.models import TextChunk


def chunk_docling_json(json_path: Path, file_id: str) -> list[TextChunk]:
    from docling.chunking import HierarchicalChunker
    from llama_index.node_parser.docling import DoclingNodeParser
    from llama_index.readers.docling import DoclingReader

    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    parser = DoclingNodeParser(chunker=HierarchicalChunker)

    documents = reader.load_data(file_path=json_path)
    nodes = parser.get_nodes_from_documents(documents=documents)

    return _nodes_to_chunks(nodes)


def _nodes_to_chunks(nodes: list[BaseNode], file_id: str) -> list[TextChunk]:
    text_chunks = [
        TextChunk(
            file_id=file_id,
            chunk_index=idx,
            page_number=_extract_page_number(node),
            text=node.get_content(metadata_mode=MetadataMode.NONE),
        )
        for idx, node in enumerate(nodes)
    ]

    return text_chunks


def _extract_page_number(node: BaseNode) -> int | None:
    doc_items = (node.metadata or {}).get("doc_items", [])

    for doc_item in doc_items:
        for prov in doc_item.get("prov", []):
            page_no = prov.get("page_no")
            if page_no is not None:
                return page_no

    return None
