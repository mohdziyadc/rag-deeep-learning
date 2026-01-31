# Phase 2: Parsing, Normalizing, Chunking, and Storage (RAGFlow-Faithful)

## Goal
Build a production-minded ingestion pipeline that mirrors how RAGFlow parses diverse files (PDF, DOCX, XLSX/CSV, PPTX, HTML/MD/JSON/TXT, images with OCR, audio/video transcription, and Google Docs/Sheets/Slides). You will:

- Convert many file types into a unified document model.
- Normalize and chunk content in a consistent, queryable way.
- Store into a **new Elasticsearch index** dedicated to parsed content.
- Keep the pipeline aligned with RAGFlow’s architecture and terminology while using open-source parsers whenever possible.

No toy snippets, no pseudo-code. This guide gives the exact code you should implement, with the reasoning behind every step and how it ties to RAGFlow.

---

## Table of Contents
1. The Big Picture
2. How RAGFlow Does It (Where To Look)
3. Phase 2 Architecture
4. Step-by-Step Implementation
5. File-Type Coverage Matrix and Parser Choices
6. Chunking Strategy (Production-Ready Libraries First)
7. Storage Design (Postgres + New Index)
8. FastAPI App (Ingestion API)
9. End-to-End Ingestion Flow
10. Testing Checklist

---

## 1. The Big Picture

RAGFlow treats parsing as the **entry point** of the RAG pipeline. Phase 1 handled retrieval; Phase 2 makes sure **anything** can be turned into clean, chunked, searchable text (plus metadata) before it reaches search.

Pipeline view:

```
┌───────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────┐
│  Files    │─▶│  Parser       │─▶│ Normalizer   │─▶│ Chunker      │─▶│ Storage  │
│ (any type)│  │ (type-aware)  │  │ (unified doc)│  │ (by strategy)│  │ (ES idx) │
└───────────┘   └──────────────┘   └──────────────┘   └─────────────┘   └──────────┘
```

Core idea: **parsing is a conversion problem**. Whatever the input (PDF, Google Doc, image, audio), the output must be **structured text + metadata**, then chunked, then indexed.

---

## 2. How RAGFlow Does It (Where To Look)

You asked for a RAGFlow-faithful approach. These are the relevant entry points and parsers:

- **Parser component** (pipeline config and file type mapping)
  - `rag/flow/parser/parser.py`
  - The `ParserParam` class defines supported file types, allowed output formats, and parsing methods.

- **Core file parsers**
  - `deepdoc/parser/` (PDF, DOCX, XLSX, PPTX, HTML, JSON, Markdown, TXT)
  - Example: `deepdoc/parser/pdf_parser.py`, `deepdoc/parser/docx_parser.py`, `deepdoc/parser/excel_parser.py`, `deepdoc/parser/ppt_parser.py`, `deepdoc/parser/html_parser.py`

- **OCR and vision** (RAGFlow uses an internal OCR stack)
  - `deepdoc/vision/ocr.py`
  - `deepdoc/vision/layout_recognizer.py`

- **Google Drive / Docs / Sheets / Slides**
  - `common/data_source/google_drive/doc_conversion.py`
  - `common/data_source/google_drive/section_extraction.py`
  - `common/data_source/google_drive/connector.py`

- **Storage abstractions**
  - `common/doc_store/doc_store_base.py`
  - `common/doc_store/es_conn_base.py`

- **File metadata and mapping**
  - `api/db/services/file_service.py`
  - `api/db/services/file2document_service.py`

This Phase 2 guide mirrors these concepts but keeps your implementation lightweight and production-ready using open-source libraries first.

---

## 3. Phase 2 Architecture

You will add a new ingestion module to your Phase 1 project. The new pieces are:

```
rag-deep-learning/
├── core/
│   ├── ingestion/
│   │   ├── ingest.py              # Orchestrates parse -> normalize -> chunk -> store
│   │   └── sources/
│   │       └── google_drive.py    # Optional: Google Docs/Sheets/Slides
│   ├── parsers/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── pdf.py
│   │   ├── docx.py
│   │   ├── pptx.py
│   │   ├── xlsx.py
│   │   ├── html.py
│   │   ├── markdown.py
│   │   ├── json_parser.py
│   │   ├── text.py
│   │   ├── image_ocr.py
│   │   ├── email.py
│   │   ├── audio.py
│   │   └── video.py
│   ├── normalization/
│   │   └── normalizer.py
│   ├── chunking/
│   │   └── chunker.py
│   └── storage/
│       ├── parsed_index.py        # New ES index + insert
│       └── metadata_store.py      # Postgres metadata CRUD
└── models/
    ├── parsed.py                  # Data models for parsed docs + chunks
    └── metadata.py                # Postgres models (File, Document)
```

This mirrors RAGFlow’s separation:
- **Parser** (type-aware logic)
- **Normalizer** (unified structure)
- **Chunker** (splitting rules)
- **DocStore** (index & write)

---

## 4. Step-by-Step Implementation

### Step 0: Dependencies (Production-Ready First)

We prioritize open-source libraries already used in production and in RAGFlow-like stacks.

Add these dependencies to your `pyproject.toml` (exact code):

```toml
[project]
dependencies = [
    # Existing Phase 1 deps...

    # File parsing
    "pypdf>=4.2.0",
    "pdfplumber>=0.11.0",
    "python-docx>=1.1.2",
    "python-pptx>=1.0.2",
    "openpyxl>=3.1.5",
    "pandas>=2.2.2",
    "beautifulsoup4>=4.12.3",
    "html5lib>=1.1",
    "markdown-it-py>=3.0.0",
    "chardet>=5.2.0",

    # API + DB
    "fastapi>=0.128.0",
    "uvicorn[standard]>=0.30.0",
    "sqlalchemy>=2.0.0",
    "asyncpg>=0.29.0",

    # OCR (open-source, scalable)
    "paddleocr>=2.7.3",
    "paddlepaddle>=2.6.1",

    # Chunking (production-ready splitter)
    "langchain-text-splitters>=0.3.0",

    # Google Docs/Sheets/Slides (final resort)
    "google-api-python-client>=2.125.0",
    "google-auth>=2.29.0",
    "google-auth-oauthlib>=1.2.0",
]
```

Why this matches RAGFlow:
- RAGFlow uses **deepdoc** for PDF (complex OCR/layout) and basic libraries for DOCX/XLSX/PPTX/HTML.
- You are using **open-source libraries** first. Google APIs are used only for Google-native formats (Docs/Sheets/Slides), which require API exports.

---

### Step 1: Define a Unified Parsed Document Model

RAGFlow parses into a JSON-like structure (see `deepdoc/parser/*`). You need a consistent model across all file types so chunking and storage are uniform.

Create `models/parsed.py`:

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


@dataclass
class ParsedSection:
    text: str
    section_type: str = "text"  # text, table, image, metadata
    page: Optional[int] = None
    title: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    doc_id: str
    source_name: str
    file_type: str
    title: str
    sections: list[ParsedSection]
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChunkedDocument:
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    source_name: str
    file_type: str
    section_type: str
    page: Optional[int]
    title: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)
```

Why this matters:
- RAGFlow outputs **structured JSON** per parser and then uses a separate chunker. You are doing the same: parse -> normalize -> chunk.

---

### Step 1.5: Add Postgres Metadata Models (RAGFlow-Style)

RAGFlow keeps **metadata in a relational DB** (MySQL/Postgres) and **content in ES**. We add a small metadata layer to track files, documents, and ingestion status.

Create `models/metadata.py`:

```python
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, JSON, ForeignKey


class Base(DeclarativeBase):
    pass


class File(Base):
    __tablename__ = "files"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)  # local, google_drive, etc.
    mime_type: Mapped[str] = mapped_column(String, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=True)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    file_id: Mapped[str] = mapped_column(String, ForeignKey("files.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, default="parsed")  # parsed, chunked, indexed
    title: Mapped[str] = mapped_column(String, nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

This mirrors RAGFlow’s `document` and `file` models in `api/db/db_models.py` and `file2document` mapping in `api/db/services/file2document_service.py`, but in a simplified, learnable form.

---

### Step 2: Parser Registry (RAGFlow-like dispatch)

RAGFlow uses a parser component that maps file types to parsers (`rag/flow/parser/parser.py`). You will implement a lightweight registry with explicit type mapping and fallback.

Create `core/parsers/base.py`:

```python
from abc import ABC, abstractmethod
from typing import Any
from models.parsed import ParsedDocument


class BaseParser(ABC):
    @abstractmethod
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        raise NotImplementedError
```

Create `core/parsers/registry.py`:

```python
from typing import Any
from core.parsers.base import BaseParser
from core.parsers.pdf import PdfParser
from core.parsers.docx import DocxParser
from core.parsers.pptx import PptxParser
from core.parsers.xlsx import XlsxParser
from core.parsers.html import HtmlParser
from core.parsers.markdown import MarkdownParser
from core.parsers.json_parser import JsonParser
from core.parsers.text import TextParser
from core.parsers.image_ocr import ImageOcrParser
from core.parsers.email import EmailParser
from core.parsers.audio import AudioParser
from core.parsers.video import VideoParser


class ParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[str, BaseParser] = {
            "pdf": PdfParser(),
            "docx": DocxParser(),
            "doc": DocxParser(),
            "pptx": PptxParser(),
            "ppt": PptxParser(),
            "xlsx": XlsxParser(),
            "xls": XlsxParser(),
            "csv": XlsxParser(),
            "html": HtmlParser(),
            "htm": HtmlParser(),
            "md": MarkdownParser(),
            "markdown": MarkdownParser(),
            "mdx": MarkdownParser(),
            "json": JsonParser(),
            "txt": TextParser(),
            "jpg": ImageOcrParser(),
            "jpeg": ImageOcrParser(),
            "png": ImageOcrParser(),
            "gif": ImageOcrParser(),
            "tif": ImageOcrParser(),
            "tiff": ImageOcrParser(),
            "eml": EmailParser(),
            "msg": EmailParser(),
            "mp3": AudioParser(),
            "wav": AudioParser(),
            "mp4": VideoParser(),
            "mkv": VideoParser(),
            "avi": VideoParser(),
        }

    def get_parser(self, ext: str) -> BaseParser:
        ext = ext.lower().lstrip(".")
        if ext not in self._parsers:
            raise ValueError(f"Unsupported file type: {ext}")
        return self._parsers[ext]
```

Why this matches RAGFlow:
- The registry mirrors `ParserParam.setups` and the dispatch in `rag/flow/parser/parser.py`.
- You can extend by adding new mappings just like RAGFlow supports more suffixes.

---

### Step 2.25: Postgres Metadata Store (CRUD)

Create `core/storage/metadata_store.py`:

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select
from models.metadata import Base, File, Document


class MetadataStore:
    def __init__(self, db_url: str) -> None:
        self.engine = create_async_engine(db_url, echo=False)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init_db(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_file(self, file: File) -> None:
        async with self.session_factory() as session:
            session.add(file)
            await session.commit()

    async def create_document(self, doc: Document) -> None:
        async with self.session_factory() as session:
            session.add(doc)
            await session.commit()

    async def update_document(self, doc_id: str, **kwargs) -> None:
        async with self.session_factory() as session:
            doc = await session.get(Document, doc_id)
            for k, v in kwargs.items():
                setattr(doc, k, v)
            await session.commit()

    async def get_document(self, doc_id: str) -> Document | None:
        async with self.session_factory() as session:
            return await session.get(Document, doc_id)

    async def list_documents(self) -> list[Document]:
        async with self.session_factory() as session:
            res = await session.execute(select(Document))
            return list(res.scalars().all())
```

This gives you the minimal relational layer you asked for: **files and documents metadata** tracked in Postgres, while **chunks live in ES**.

---

### Step 2.5: Embedded Image OCR Strategy (Fixes Image-in-Docs Problem)

You are correct: if you only OCR standalone image files, you miss **images embedded inside PDFs, DOCX, PPTX, XLSX, and HTML**. RAGFlow handles this via its DeepDoc pipeline (layout + OCR + table extraction). To stay faithful while keeping it production-ready, we add a **secondary OCR pass** for embedded images.

Design rule:
- Parse text first (fast path).
- Extract embedded images where supported.
- OCR those images and add them as `section_type="image"` blocks.

This gives you recall for image-heavy docs without fully re-implementing DeepDoc.

Update `core/parsers/image_ocr.py` to support multi-image OCR:

```python
from typing import Any
from io import BytesIO
from PIL import Image
from paddleocr import PaddleOCR
from models.parsed import ParsedDocument, ParsedSection


class ImageOcrParser:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def parse_image_bytes(self, image_bytes: bytes, metadata: dict[str, Any]) -> ParsedSection:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.parse_pil(img, metadata)

    def parse_pil(self, img: Image.Image, metadata: dict[str, Any]) -> ParsedSection:
        result = self.ocr.ocr(img, cls=True)
        lines = []
        for res in result:
            for line in res:
                text = line[1][0]
                if text.strip():
                    lines.append(text.strip())
        text = "\n".join(lines)
        return ParsedSection(
            text=text,
            section_type="image",
            page=metadata.get("page"),
            title=metadata.get("title"),
            metadata=metadata,
        )

    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        section = self.parse_image_bytes(data, metadata)
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="image",
            title=title,
            sections=[section],
            raw_text=section.text,
            metadata=metadata,
        )
```

This lets any file parser pass images to OCR without duplicating OCR logic.

---

### Step 3: Parsers (Open-Source First, Raw if Needed)

Below are **exact parsers** for each file type. These follow the same style RAGFlow uses in `deepdoc/parser/*` but are simplified, production-ready, and maintainable.

#### 3.1 PDF Parser (`core/parsers/pdf.py`)

Use `pdfplumber` + `pypdf` for plain text, then **optionally OCR page renders** for image-heavy PDFs.

```python
from typing import Any
import io
import pdfplumber
from pypdf import PdfReader
from models.parsed import ParsedDocument, ParsedSection
from core.parsers.image_ocr import ImageOcrParser


class PdfParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        sections: list[ParsedSection] = []
        ocr = ImageOcrParser()
        enable_page_ocr = metadata.get("enable_pdf_page_ocr", False)

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    sections.append(
                        ParsedSection(
                            text=text.strip(),
                            section_type="text",
                            page=i + 1,
                        )
                    )
                # OCR fallback: if page text is empty or you explicitly enable OCR
                if enable_page_ocr or not text.strip():
                    page_image = page.to_image(resolution=200).original
                    image_section = ocr.parse_pil(page_image, {"page": i + 1, "title": name})
                    if image_section.text.strip():
                        sections.append(image_section)

        raw_text = "\n".join([s.text for s in sections])
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="pdf",
            title=title,
            sections=sections,
            raw_text=raw_text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `deepdoc/parser/pdf_parser.py`
- `rag/flow/parser/parser.py` (parse_method = deepdoc/plain_text)

This is the **plain-text** path. For complex PDFs, RAGFlow uses `deepdoc` OCR/layout. You will integrate OCR as an optional fallback (see image/OCR parser) or later through a deepdoc-like pipeline if needed.

---

#### 3.2 DOCX Parser (`core/parsers/docx.py`)

```python
from typing import Any
from io import BytesIO
from docx import Document
from models.parsed import ParsedDocument, ParsedSection
from core.parsers.image_ocr import ImageOcrParser


class DocxParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        doc = Document(BytesIO(data))
        sections: list[ParsedSection] = []
        ocr = ImageOcrParser()

        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                sections.append(
                    ParsedSection(
                        text=text,
                        section_type="text",
                        title=p.style.name if p.style else None,
                    )
                )

        for table in doc.tables:
            rows = []
            for row in table.rows:
                rows.append([cell.text.strip() for cell in row.cells])
            if rows:
                table_text = "\n".join([", ".join(r) for r in rows])
                sections.append(ParsedSection(text=table_text, section_type="table"))

        # Embedded images OCR
        for rel in doc.part._rels.values():
            if "image" in rel.reltype:
                image_bytes = rel.target_part.blob
                image_section = ocr.parse_image_bytes(image_bytes, {"title": name})
                if image_section.text.strip():
                    sections.append(image_section)

        raw_text = "\n".join([s.text for s in sections])
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="docx",
            title=title,
            sections=sections,
            raw_text=raw_text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `deepdoc/parser/docx_parser.py`

---

#### 3.3 PPTX Parser (`core/parsers/pptx.py`)

```python
from typing import Any
from io import BytesIO
from pptx import Presentation
from models.parsed import ParsedDocument, ParsedSection
from core.parsers.image_ocr import ImageOcrParser


class PptxParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        prs = Presentation(BytesIO(data))
        sections: list[ParsedSection] = []
        ocr = ImageOcrParser()

        for i, slide in enumerate(prs.slides):
            texts: list[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    t = shape.text.strip()
                    if t:
                        texts.append(t)
            if texts:
                sections.append(
                    ParsedSection(
                        text="\n".join(texts),
                        section_type="text",
                        page=i + 1,
                    )
                )
            # Embedded images OCR per slide
            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    image_bytes = shape.image.blob
                    image_section = ocr.parse_image_bytes(image_bytes, {"page": i + 1, "title": name})
                    if image_section.text.strip():
                        sections.append(image_section)

        raw_text = "\n".join([s.text for s in sections])
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="pptx",
            title=title,
            sections=sections,
            raw_text=raw_text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `deepdoc/parser/ppt_parser.py`

---

#### 3.4 XLSX/CSV Parser (`core/parsers/xlsx.py`)

```python
from typing import Any
from io import BytesIO
import pandas as pd
from openpyxl import load_workbook
from models.parsed import ParsedDocument, ParsedSection
from core.parsers.image_ocr import ImageOcrParser


class XlsxParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        sections: list[ParsedSection] = []
        ocr = ImageOcrParser()

        try:
            wb = load_workbook(BytesIO(data), data_only=True)
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                rows = []
                for row in ws.iter_rows(values_only=True):
                    row_vals = [str(cell).strip() if cell is not None else "" for cell in row]
                    rows.append(", ".join(row_vals))
                if rows:
                    sections.append(
                        ParsedSection(
                            text="\n".join(rows),
                            section_type="table",
                            title=sheet,
                        )
                    )
                # Embedded images OCR
                for img in getattr(ws, "_images", []):
                    try:
                        image_bytes = img._data()
                        image_section = ocr.parse_image_bytes(image_bytes, {"title": f"{name}:{sheet}"})
                        if image_section.text.strip():
                            sections.append(image_section)
                    except Exception:
                        continue
        except Exception:
            df = pd.read_csv(BytesIO(data))
            rows = [", ".join(map(str, df.columns))]
            for _, row in df.iterrows():
                rows.append(", ".join(map(str, row.values)))
            sections.append(ParsedSection(text="\n".join(rows), section_type="table"))

        raw_text = "\n".join([s.text for s in sections])
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="xlsx",
            title=title,
            sections=sections,
            raw_text=raw_text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `deepdoc/parser/excel_parser.py`

---

#### 3.5 HTML Parser (`core/parsers/html.py`)

```python
from typing import Any
import base64
from bs4 import BeautifulSoup
from models.parsed import ParsedDocument, ParsedSection
from core.parsers.image_ocr import ImageOcrParser


class HtmlParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        html_text = data.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html_text, "html5lib")
        ocr = ImageOcrParser()

        for tag in soup(["script", "style"]):
            tag.extract()

        text = soup.get_text("\n")
        sections = [ParsedSection(text=text.strip(), section_type="text")]

        # Embedded base64 images (data URIs)
        for img in soup.find_all("img"):
            src = img.get("src") or ""
            if src.startswith("data:image/") and "base64," in src:
                try:
                    b64 = src.split("base64,", 1)[1]
                    image_bytes = base64.b64decode(b64)
                    image_section = ocr.parse_image_bytes(image_bytes, {"title": name})
                    if image_section.text.strip():
                        sections.append(image_section)
                except Exception:
                    continue
        raw_text = text.strip()
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="html",
            title=title,
            sections=sections,
            raw_text=raw_text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `deepdoc/parser/html_parser.py`

---

#### 3.6 Markdown Parser (`core/parsers/markdown.py`)

```python
from typing import Any
from markdown_it import MarkdownIt
from models.parsed import ParsedDocument, ParsedSection


class MarkdownParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        md_text = data.decode("utf-8", errors="ignore")
        md = MarkdownIt()
        tokens = md.parse(md_text)

        lines = []
        for t in tokens:
            if t.type == "inline" and t.content.strip():
                lines.append(t.content.strip())

        text = "\n".join(lines)
        sections = [ParsedSection(text=text, section_type="text")]
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="markdown",
            title=title,
            sections=sections,
            raw_text=text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `deepdoc/parser/markdown_parser.py`

---

#### 3.7 JSON Parser (`core/parsers/json_parser.py`)

```python
from typing import Any
import json
from models.parsed import ParsedDocument, ParsedSection


class JsonParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        obj = json.loads(data.decode("utf-8", errors="ignore"))
        text = json.dumps(obj, indent=2, ensure_ascii=False)
        sections = [ParsedSection(text=text, section_type="text")]
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="json",
            title=title,
            sections=sections,
            raw_text=text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `deepdoc/parser/json_parser.py`

---

#### 3.8 TXT Parser (`core/parsers/text.py`)

```python
from typing import Any
import chardet
from models.parsed import ParsedDocument, ParsedSection


class TextParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        detected = chardet.detect(data)
        encoding = detected.get("encoding") or "utf-8"
        text = data.decode(encoding, errors="ignore")
        sections = [ParsedSection(text=text, section_type="text")]
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="txt",
            title=title,
            sections=sections,
            raw_text=text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `deepdoc/parser/txt_parser.py`

---

#### 3.9 Image OCR Parser (`core/parsers/image_ocr.py`)

RAGFlow uses its own OCR stack (`deepdoc/vision/ocr.py`). You will use PaddleOCR as a production-ready open-source alternative. This follows your preference and stays faithful to RAGFlow’s role: extract text from images.

```python
from typing import Any
from io import BytesIO
from PIL import Image
from paddleocr import PaddleOCR
from models.parsed import ParsedDocument, ParsedSection


class ImageOcrParser:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en")

    def parse_image_bytes(self, image_bytes: bytes, metadata: dict[str, Any]) -> ParsedSection:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.parse_pil(img, metadata)

    def parse_pil(self, img: Image.Image, metadata: dict[str, Any]) -> ParsedSection:
        result = self.ocr.ocr(img, cls=True)
        lines = []
        for res in result:
            for line in res:
                text = line[1][0]
                if text.strip():
                    lines.append(text.strip())
        text = "\n".join(lines)
        return ParsedSection(
            text=text,
            section_type="image",
            page=metadata.get("page"),
            title=metadata.get("title"),
            metadata=metadata,
        )

    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        section = self.parse_image_bytes(data, metadata)
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="image",
            title=title,
            sections=[section],
            raw_text=section.text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `deepdoc/vision/ocr.py`
- `rag/flow/parser/parser.py` (image parse_method = ocr)

---

#### 3.10 Email Parser (`core/parsers/email.py`)

```python
from typing import Any
import email
from email import policy
from models.parsed import ParsedDocument, ParsedSection


class EmailParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        msg = email.message_from_bytes(data, policy=policy.default)

        subject = msg.get("subject", "")
        from_addr = msg.get("from", "")
        to_addr = msg.get("to", "")

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_content()
                    break
        else:
            body = msg.get_content()

        text = f"Subject: {subject}\nFrom: {from_addr}\nTo: {to_addr}\n\n{body}"
        sections = [ParsedSection(text=text, section_type="text")]
        title = subject or metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="email",
            title=title,
            sections=sections,
            raw_text=text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `rag/flow/parser/parser.py` (email parser type)

---

#### 3.11 Audio Parser (`core/parsers/audio.py`) and Video Parser (`core/parsers/video.py`)

RAGFlow uses external ASR/VLM models. For Phase 2, keep the interfaces but use a placeholder until you add ASR.

```python
from typing import Any
from models.parsed import ParsedDocument, ParsedSection


class AudioParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        text = "[AUDIO TRANSCRIPTION PENDING]"
        sections = [ParsedSection(text=text, section_type="text")]
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="audio",
            title=title,
            sections=sections,
            raw_text=text,
            metadata=metadata,
        )
```

```python
from typing import Any
from models.parsed import ParsedDocument, ParsedSection


class VideoParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        text = "[VIDEO TRANSCRIPTION PENDING]"
        sections = [ParsedSection(text=text, section_type="text")]
        title = metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="video",
            title=title,
            sections=sections,
            raw_text=text,
            metadata=metadata,
        )
```

RAGFlow reference:
- `rag/flow/parser/parser.py` (audio/video need LLM model)

---

### Step 4: Google Drive End-to-End (List -> Export -> Parse -> Store)

Google native files **must** be exported via Google APIs. RAGFlow does this in:
`common/data_source/google_drive/doc_conversion.py`.

Here is the **full end-to-end integration** you can follow:

Create `core/ingestion/sources/google_drive.py`:

```python
from typing import Any, Iterator
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from core.ingestion.ingest import IngestionService


GOOGLE_EXPORT_TARGETS: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
}


class GoogleDriveIngestor:
    def __init__(self, service_account_json: dict[str, Any], ingestion: IngestionService) -> None:
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        creds = Credentials.from_service_account_info(service_account_json, scopes=scopes)
        self.drive = build("drive", "v3", credentials=creds)
        self.ingestion = ingestion

    def list_files(self, folder_id: str) -> Iterator[dict[str, Any]]:
        page_token = None
        while True:
            resp = self.drive.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)",
                pageToken=page_token,
            ).execute()
            for f in resp.get("files", []):
                yield f
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    def _download_or_export(self, file: dict[str, Any]) -> tuple[bytes, str]:
        mime_type = file.get("mimeType", "")
        name = file.get("name", "unknown")

        if mime_type in GOOGLE_EXPORT_TARGETS:
            export_mime, ext = GOOGLE_EXPORT_TARGETS[mime_type]
            request = self.drive.files().export_media(fileId=file["id"], mimeType=export_mime)
            final_name = f"{name}{ext}"
        else:
            request = self.drive.files().get_media(fileId=file["id"])
            final_name = name

        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue(), final_name

    async def ingest_folder(self, folder_id: str) -> list[dict[str, Any]]:
        results = []
        for f in self.list_files(folder_id):
            data, final_name = self._download_or_export(f)
            metadata = {
                "source": "google_drive",
                "webViewLink": f.get("webViewLink"),
                "modifiedTime": f.get("modifiedTime"),
            }
            result = await self.ingestion.ingest(final_name, data, metadata)
            results.append({"name": final_name, "result": result})
        return results
```

How this aligns with RAGFlow:
- RAGFlow exports Docs/Sheets/Slides to Office formats and parses them normally.
- The export targets above match RAGFlow’s logic in `common/data_source/google_drive/doc_conversion.py`.
- You ingest the exported bytes through the same parser registry and storage pipeline.

RAGFlow reference:
- `common/data_source/google_drive/doc_conversion.py`
- `common/data_source/google_drive/section_extraction.py`

---

### Step 5: Normalization

Normalization is where you align all file types into one clean, structured representation. This matches RAGFlow’s internal JSON outputs.

Create `core/normalization/normalizer.py`:

```python
from models.parsed import ParsedDocument, ParsedSection


class DocumentNormalizer:
    def normalize(self, doc: ParsedDocument) -> ParsedDocument:
        normalized_sections: list[ParsedSection] = []
        for section in doc.sections:
            text = section.text.strip()
            if not text:
                continue
            normalized_sections.append(section)

        doc.sections = normalized_sections
        doc.raw_text = "\n".join([s.text for s in normalized_sections])
        return doc
```

Why this matters:
- RAGFlow output is structured but still normalized in later pipeline stages.
- Normalization is where you drop empty blocks, unify encodings, and keep metadata.

---

### Step 6: Chunking (Production-Ready)

Use `langchain-text-splitters` to avoid re-implementing chunk logic.

Create `core/chunking/chunker.py`:

```python
from typing import List
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.parsed import ParsedDocument, ChunkedDocument


class DocumentChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, doc: ParsedDocument) -> list[ChunkedDocument]:
        chunks: list[ChunkedDocument] = []
        chunk_index = 0

        for section in doc.sections:
            splits = self.splitter.split_text(section.text)
            for text in splits:
                chunks.append(
                    ChunkedDocument(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc.doc_id,
                        content=text,
                        chunk_index=chunk_index,
                        source_name=doc.source_name,
                        file_type=doc.file_type,
                        section_type=section.section_type,
                        page=section.page,
                        title=section.title or doc.title,
                        metadata=section.metadata | doc.metadata,
                    )
                )
                chunk_index += 1
        return chunks
```

This is equivalent to RAGFlow’s chunking stage, but you rely on a production-ready splitter rather than custom token logic.

---

### Step 7: Storage (New Index)

You want a **new index** dedicated to parsed chunks. The index structure should mirror Phase 1 but include parser metadata.

Create `core/storage/parsed_index.py`:

```python
from elasticsearch import AsyncElasticsearch


PARSED_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "whitespace_lowercase": {
                    "type": "custom",
                    "tokenizer": "whitespace",
                    "filter": ["lowercase"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "source_name": {"type": "keyword"},
            "file_type": {"type": "keyword"},
            "section_type": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "whitespace_lowercase"},
            "page": {"type": "integer"},
            "content": {"type": "text"},
            "content_ltks": {
                "type": "text",
                "analyzer": "whitespace_lowercase",
                "search_analyzer": "whitespace_lowercase"
            },
            "metadata": {"type": "object", "enabled": False}
        }
    }
}


class ParsedIndex:
    def __init__(self, es_host: str, index_name: str) -> None:
        self.client = AsyncElasticsearch(hosts=[es_host])
        self.index_name = index_name

    async def create_index(self) -> None:
        if not await self.client.indices.exists(index=self.index_name):
            await self.client.indices.create(index=self.index_name, body=PARSED_INDEX_MAPPING)

    async def bulk_insert(self, chunks: list[dict]) -> None:
        operations = []
        for c in chunks:
            operations.append({"index": {"_index": self.index_name, "_id": c["chunk_id"]}})
            operations.append(c)
        await self.client.bulk(operations=operations)
```

Why this matches RAGFlow:
- RAGFlow’s `common/doc_store/es_conn_base.py` uses a shared index schema, and stores content + metadata in ES.
- You are creating a dedicated index to keep parsed data separated from Phase 1.

---

### Step 8: Ingestion Orchestrator

Create `core/ingestion/ingest.py`:

```python
from typing import Any
import uuid
from core.parsers.registry import ParserRegistry
from core.normalization.normalizer import DocumentNormalizer
from core.chunking.chunker import DocumentChunker
from core.storage.parsed_index import ParsedIndex
from core.storage.metadata_store import MetadataStore
from models.metadata import File, Document
from models.parsed import ParsedDocument


class IngestionService:
    def __init__(self, es_host: str, index_name: str, db_url: str) -> None:
        self.registry = ParserRegistry()
        self.normalizer = DocumentNormalizer()
        self.chunker = DocumentChunker()
        self.index = ParsedIndex(es_host, index_name)
        self.meta = MetadataStore(db_url)

    async def ingest(self, name: str, data: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
        doc_id = str(uuid.uuid4())
        file_id = str(uuid.uuid4())
        metadata = dict(metadata or {})
        metadata["doc_id"] = doc_id
        metadata["file_id"] = file_id

        await self.meta.init_db()
        await self.meta.create_file(
            File(
                id=file_id,
                name=name,
                source=metadata.get("source", "local"),
                mime_type=metadata.get("mime_type"),
                size_bytes=len(data),
                metadata=metadata,
            )
        )
        ext = name.split(".")[-1].lower()
        parser = self.registry.get_parser(ext)
        parsed_doc = parser.parse(name, data, metadata)
        normalized = self.normalizer.normalize(parsed_doc)
        chunks = self.chunker.chunk(normalized)

        await self.index.create_index()
        await self.index.bulk_insert([c.__dict__ for c in chunks])

        await self.meta.create_document(
            Document(
                id=doc_id,
                file_id=file_id,
                status="indexed",
                title=parsed_doc.title,
                file_type=parsed_doc.file_type,
                chunk_count=len(chunks),
                metadata=metadata,
            )
        )
        return {
            "doc_id": doc_id,
            "chunks_created": len(chunks),
            "file_type": parsed_doc.file_type,
        }
```

This is your Phase 2 entry point. It mirrors the RAGFlow ingestion pipeline but in a simplified, explicit form.

---

## 5. File-Type Coverage Matrix and Parser Choices

| File Type | Parser Used | Library | RAGFlow Reference |
|-----------|-------------|---------|-------------------|
| PDF | Plain-text parser | pdfplumber + pypdf | `deepdoc/parser/pdf_parser.py` |
| DOCX | Docx parser | python-docx | `deepdoc/parser/docx_parser.py` |
| XLSX/XLS/CSV | Spreadsheet parser | openpyxl + pandas | `deepdoc/parser/excel_parser.py` |
| PPTX/PPT | Slides parser | python-pptx | `deepdoc/parser/ppt_parser.py` |
| HTML | HTML parser | BeautifulSoup | `deepdoc/parser/html_parser.py` |
| Markdown | Markdown parser | markdown-it-py | `deepdoc/parser/markdown_parser.py` |
| JSON | JSON parser | json | `deepdoc/parser/json_parser.py` |
| TXT | Text parser | chardet | `deepdoc/parser/txt_parser.py` |
| Images | OCR parser | PaddleOCR | `deepdoc/vision/ocr.py` |
| Embedded images (PDF/DOCX/PPTX/XLSX/HTML) | Secondary OCR pass | PaddleOCR | `deepdoc/vision/ocr.py` + `deepdoc/parser/pdf_parser.py` |
| Google Docs | Export -> DOCX | Google API | `common/data_source/google_drive/*` |
| Google Sheets | Export -> XLSX | Google API | `common/data_source/google_drive/*` |
| Google Slides | Export -> PPTX | Google API | `common/data_source/google_drive/*` |

If a file type has no stable library and no easy raw parser, then the fallback is:
- Export to a supported format (Google native types).
- OCR for images or scanned PDFs (and embedded images).

---

## 6. Chunking Strategy (Production-Ready Libraries First)

Why chunking matters:
- LLMs have context limits.
- Chunking affects recall, precision, and citation quality.

You are using `langchain-text-splitters` to avoid writing your own chunk logic, which is production-ready and proven.

If you want to align even closer to RAGFlow later:
- RAGFlow has multiple chunkers under `rag/app/*.py` (e.g., `rag/app/naive.py`).
- For PDFs, RAGFlow can chunk by page or layout sections when using DeepDoc.

---

## 7. Storage Design (Postgres + New Index)

You are using **Postgres for metadata** and a **new ES index for chunks**. That gives you:
- Clear separation from Phase 1 retrieval index.
- A relational source of truth for file/document state.
- A clean layer to test parsing and chunking quality without affecting search.

RAGFlow stores parsed chunks in a doc store abstraction and keeps file/document mapping in SQL:
- `common/doc_store/es_conn_base.py`
- `api/db/services/file2document_service.py`

You now mirror that split: Postgres = metadata, ES = chunk content.

---

## 8. FastAPI App (Ingestion API)

Yes, we build a FastAPI app in Phase 2. This gives you a real ingestion endpoint that:
- Accepts file uploads (local)
- Triggers the parsing + chunking + storage pipeline
- Writes metadata into Postgres and chunks into ES

Create `app/main.py`:

```python
from fastapi import FastAPI, UploadFile, File
from core.ingestion.ingest import IngestionService


app = FastAPI(title="Phase 2 Parser")

ingestion = IngestionService(
    es_host="http://localhost:9200",
    index_name="rag_parsed_chunks",
    db_url="postgresql+asyncpg://raguser:ragpass@localhost:5432/ragdb",
)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    data = await file.read()
    result = await ingestion.ingest(file.filename, data, {"source": "local", "mime_type": file.content_type})
    return result
```

This mirrors RAGFlow’s ingestion endpoint patterns but keeps the pipeline small and explicit.

---

## 9. End-to-End Ingestion Flow

Final ingestion flow:

1. Receive file (local upload or Google export).
2. Determine file extension.
3. Parse with correct parser.
4. OCR embedded images (if present).
5. Normalize sections.
6. Chunk with splitter.
7. Insert into new ES index.
8. Update Postgres metadata status.

This directly mirrors the RAGFlow ingestion pipeline: parser -> chunk -> store (+ metadata updates).

---

## 10. Testing Checklist

For each file type:

- Upload file
- Run parser
- Ensure sections extracted
- Ensure chunks inserted into new ES index
- Validate metadata fields (doc_id, file_type, page)

Suggested test set:
- PDF (both text-only and scanned)
- DOCX with tables
- XLSX with multiple sheets
- PPTX with bullet points
- HTML and Markdown
- JSON data file
- TXT large log
- Image with OCR text
- Google Doc/Sheet/Slide

---

## What You Now Understand (Phase 2 Outcomes)

After implementing this guide, you will know:

1. How RAGFlow routes file types to parsers.
2. How each file type is parsed into normalized text.
3. Why normalization matters before chunking.
4. How chunking works at scale using production-ready libraries.
5. How parsed chunks are stored in a separate ES index.

This completes Phase 2 and prepares you for Phase 3 (generation with citations).

