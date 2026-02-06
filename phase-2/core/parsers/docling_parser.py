

import tempfile
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter
from core.parsers.base import BaseParser
from models.parsed import ParsedDocument, ParsedSection

AUDIO_EXTS = {"mp3", "wav", "vtt"}

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
        
        html = result.document.export_to_html()
        soup = BeautifulSoup(html, 'html.parser')
        sections: list[ParsedSection] = []

        for table in soup.find_all("table"):
            sections.append(
                ParsedSection(
                    text=str(table),
                    section_type='table',
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
                    metadata={"docling": True}
                )
            )
        
        raw_text = "\n".join([s.text for s in sections if s.section_type == 'text'])

        return ParsedDocument(
            doc_id=metadata['doc_id'],
            source_name=name,
            file_type=file_type,
            title=title,
            sections=sections,
            raw_text=raw_text,
            metadata=metadata
        )

        
