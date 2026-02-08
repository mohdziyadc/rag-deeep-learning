

import json
import tempfile
from bs4 import BeautifulSoup
from typing import Any
from docling.document_converter import DocumentConverter
from core.parsers.base import BaseParser
from models.parsed import ParsedDocument, ParsedSection

AUDIO_EXTS = {"mp3", "wav", "vtt"}
HTML_EXPORT_EXTS = {"xls", "xlsx", "csv", "html", "htm"}
JSON_EXPORT_EXTS = {
    "pdf", "doc", "docx", "ppt", "pptx",
    "txt", "md", "markdown", "mdx",
    "jpg", "jpeg", "png", "gif", "tif", "tiff", "bmp", "webp",
    "json"
}

class DoclingParser(BaseParser):

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        
        title = metadata.get("title", name)
        file_type = name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            result = self.converter.convert(tmp.name)
        
      
        sections: list[ParsedSection] = []

        if file_type in HTML_EXPORT_EXTS or file_type in AUDIO_EXTS:
            html = result.document.export_to_html()
            soup = BeautifulSoup(html, 'html.parser')
            #Very crude level for now
            for table in soup.find_all("table"):
                sections.append(
                    ParsedSection(
                        text=str(table),
                        section_type='table',
                        content_format="html",
                        metadata={"docling": True}
                    )
                )

                table.decompose()

            text = soup.get_text("\n").strip()

            if text:
                section_type = "audio_transcript" if file_type in AUDIO_EXTS else "text"
                sections.append(
                    ParsedSection(
                        text=text,
                        section_type=section_type,
                        content_format="text",
                        metadata={"docling": True}
                    )
                )
            
        elif file_type in JSON_EXPORT_EXTS:
            json_text = json.dumps(result.document.export_to_dict())
            sections.append(
                ParsedSection(
                    text=json_text,
                    section_type="docling_json",
                    content_format="json",
                    metadata={"docling": True},
                )
            )
            
        
        raw_text = "\n".join([s.text for s in sections if s.content_format == 'text'])
        return ParsedDocument(
            doc_id=metadata['doc_id'],
            source_name=name,
            file_type=file_type,
            title=title,
            sections=sections,
            raw_text=raw_text,
            metadata=metadata
        )

        
