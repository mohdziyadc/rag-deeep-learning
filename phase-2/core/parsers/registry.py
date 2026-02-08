
from core.parsers.docling_parser import DoclingParser
from core.parsers.base import BaseParser
from typing import Any
from core.parsers.email import EmailParser


class ParserRegistry:
    def __init__(self) -> None:
        docling_parser = DoclingParser()
        email_parser = EmailParser()
        #Video yet to implemented
        self._parsers: dict[str, BaseParser] = {
            "pdf": docling_parser,
            "docx": docling_parser,
            "doc": docling_parser,
            "pptx": docling_parser,
            "ppt": docling_parser,
            "xlsx": docling_parser,
            "xls": docling_parser,
            "csv": docling_parser,
            "html": docling_parser,
            "htm": docling_parser,
            "md": docling_parser,
            "markdown": docling_parser,
            "mdx": docling_parser,
            "json": docling_parser,
            "txt": docling_parser,
            "jpg": docling_parser,
            "jpeg": docling_parser,
            "png": docling_parser,
            "gif": docling_parser,
            "tif": docling_parser,
            "tiff": docling_parser,
            "bmp": docling_parser,
            "webp": docling_parser,
            "mp3": docling_parser,
            "wav": docling_parser,
            "vtt": docling_parser,
            "eml": email_parser,
            "msg": email_parser,
        }
    
    def get_parser(self, ext: str) -> BaseParser:
        ext = ext.lower().lstrip(".")
        if ext not in self._parsers:
            raise ValueError(f"Unsupported file type: {ext}")
        return self._parsers[ext]