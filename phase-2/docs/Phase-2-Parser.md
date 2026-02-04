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

Core idea: **parsing is a conversion problem**. Whatever the input (PDF, Google Doc, image, audio), this pipeline should be followed:
**input → type‑preserving intermediate representation → schema‑aware chunking → indexable chunks + metadata. -> chunking -> indexing**

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
│   │   ├── docling_parser.py
│   │   ├── email.py
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

### Step 0: Dependencies (Docling-First)

We prioritize **Docling** as the unified parser so you do not need separate libraries for PDF/DOCX/PPTX/XLSX/HTML/Markdown/JSON/TXT/image OCR.

Add these dependencies to your `pyproject.toml` (exact code):

```toml
[project]
dependencies = [
    # Existing Phase 1 deps...

    # Unified parsing (Docling)
    "docling>=2.72.0",

    # Chunking helpers (table-aware splits)
    "beautifulsoup4>=4.14.3",

    # API + DB
    "fastapi>=0.128.0",
    "uvicorn>=0.40.0",
    "sqlalchemy[asyncio]>=2.0.46",
    "asyncpg>=0.31.0",
    "pydantic>=2.12.5",

    # Chunking (Docling HybridChunker + text splitter for HTML paths)
    "langchain-text-splitters>=1.1.0",
    "transformers",

    # Google Docs/Sheets/Slides (export only)
    "google-api-python-client>=2.188.0",
    "google-auth>=2.48.0",
    "google-auth-oauthlib>=1.2.4",
]
```

Why this matches RAGFlow:
- RAGFlow uses **deepdoc** for PDF (layout/OCR) and multiple format parsers. Docling provides the same role as a **single, unified parser**.
- Google APIs are still required only for Google-native formats (Docs/Sheets/Slides), which need export before parsing.

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
    section_type: str = "text"  # text, table, image, header, slide, metadata
    content_format: str = "text"  # text, html, json
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
    content_format: str
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

Create `core/parsers/docling_parser.py`:

```python
from typing import Any
import json
import tempfile
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter
from core.parsers.base import BaseParser
from models.parsed import ParsedDocument, ParsedSection


JSON_EXTS = {
    "ppt",
    "pptx",
    "pdf",
    "doc",
    "docx",
    "txt",
    "md",
    "markdown",
    "mdx",
    "jpg",
    "jpeg",
    "png",
    "gif",
    "tif",
    "tiff",
    "bmp",
    "webp",
}
HTML_EXTS = {"xls", "xlsx", "csv", "html", "htm"}
AUDIO_EXTS = {"mp3", "wav", "vtt"}


class DoclingParser(BaseParser):
    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        title = metadata.get("title", name)
        file_type = name.split(".")[-1].lower()

        # Docling's public API expects a file path or URL, so write bytes to a temp file.
        with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            result = self.converter.convert(tmp.name)

        sections: list[ParsedSection] = []
        raw_text = ""

        if file_type in JSON_EXTS:
            doc_dict = result.document.export_to_dict()
            json_text = json.dumps(doc_dict, indent=2, ensure_ascii=False)
            sections.append(
                ParsedSection(
                    text=json_text,
                    section_type="json",
                    content_format="json",
                    metadata={"docling": True},
                )
            )
            raw_text = json_text
        else:
            html = result.document.export_to_html()
            soup = BeautifulSoup(html, "html.parser")

            # Preserve tables as HTML blocks for table-aware chunking.
            for table in soup.find_all("table"):
                sections.append(
                    ParsedSection(
                        text=str(table),
                        section_type="table",
                        content_format="html",
                        metadata={"docling": True},
                    )
                )
                table.decompose()

            # Remaining content becomes text in reading order.
            text = soup.get_text("\n").strip()
            if text:
                section_type = "audio_transcript" if file_type in AUDIO_EXTS else "text"
                sections.append(
                    ParsedSection(
                        text=text,
                        section_type=section_type,
                        content_format="text",
                        metadata={"docling": True},
                    )
                )

            raw_text = "\n".join([s.text for s in sections if s.content_format == "text"])
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type=file_type,
            title=title,
            sections=sections,
            raw_text=raw_text,
            metadata=metadata,
        )
```

Create `core/parsers/registry.py`:

```python
from typing import Any
from core.parsers.base import BaseParser
from core.parsers.docling_parser import DoclingParser
from core.parsers.email import EmailParser
from core.parsers.video import VideoParser


class ParserRegistry:
    def __init__(self) -> None:
        docling = DoclingParser()
        self._parsers: dict[str, BaseParser] = {
            "pdf": docling,
            "docx": docling,
            "doc": docling,
            "pptx": docling,
            "ppt": docling,
            "xlsx": docling,
            "xls": docling,
            "csv": docling,
            "html": docling,
            "htm": docling,
            "md": docling,
            "markdown": docling,
            "mdx": docling,
            "json": docling,
            "txt": docling,
            "jpg": docling,
            "jpeg": docling,
            "png": docling,
            "gif": docling,
            "tif": docling,
            "tiff": docling,
            "bmp": docling,
            "webp": docling,
            "mp3": docling,
            "wav": docling,
            "vtt": docling,
            "eml": EmailParser(),
            "msg": EmailParser(),
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

### Step 2.5: Embedded Image OCR Strategy (Handled by Docling)

Docling already performs OCR and layout analysis for scanned PDFs and images, and it extracts embedded image content when possible. That means you do **not** need a separate OCR parser or a secondary OCR pass for embedded images. Keep your pipeline simple:

Design rule:
- Let Docling handle OCR + layout extraction.
- Convert Docling output into your `ParsedDocument` schema in one place (the adapter).

This keeps the pipeline faithful to RAGFlow’s “deep parsing” idea while avoiding custom OCR plumbing.

---

### Step 2.75: Intermediate Representations by File Type (Preserve Context)

We do **not** force everything into plain text early. Each parser outputs an **intermediate representation** that preserves structure and relationships, then chunking is applied in a type-aware way.

| File Type | Intermediate Representation | Why It Preserves Context |
|-----------|-----------------------------|---------------------------|
| PPT/PPTX, PDF, DOC/DOCX, TXT/MD, Images | DoclingDocument lossless JSON export | Preserves structure and metadata for rich documents |
| Sheets/Excel, HTML, CSV | DoclingDocument exported to HTML, then split into table + text sections | Tables remain intact for table-aware chunking |
| Audio (MP3/WAV/VTT) | DoclingDocument transcript | Keeps ASR output in a unified format |
| Email | Header JSON + body text + attachment metadata | Preserves fields + context |
| Video | Placeholder transcript (until you add VLM/ASR) | Keeps pipeline shape consistent |

This mirrors RAGFlow’s “output_format per parser” design, but Docling becomes the single source of truth for most formats. Docling supports both HTML and lossless JSON export. [Docling docs](https://docling-project.github.io/docling/)

---

### Step 3: Parsers (Docling-First)

Docling replaces the per-format parsers with a single adapter. Keep small custom parsers for email and video.

#### 3.1 Docling Parser (`core/parsers/docling_parser.py`)

```python
from typing import Any
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
        soup = BeautifulSoup(html, "html.parser")
        sections: list[ParsedSection] = []

        # Preserve tables as HTML blocks for table-aware chunking.
        for table in soup.find_all("table"):
            sections.append(
                ParsedSection(
                    text=str(table),
                    section_type="table",
                    content_format="html",
                    metadata={"docling": True},
                )
            )
            table.decompose()

        # Remaining content becomes text in reading order.
        text = soup.get_text("\n").strip()
        if text:
            section_type = "audio_transcript" if file_type in AUDIO_EXTS else "text"
                    sections.append(
                        ParsedSection(
                    text=text,
                    section_type=section_type,
                            content_format="text",
                    metadata={"docling": True},
                )
            )

        raw_text = "\n".join([s.text for s in sections if s.content_format == "text"])
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type=file_type,
            title=title,
            sections=sections,
            raw_text=raw_text,
            metadata=metadata,
        )
```

Notes:
- This adapter uses lossless JSON export for PPT/PPTX, PDF, DOC/DOCX, TXT/MD, and images to preserve structure. [Docling docs](https://docling-project.github.io/docling/)
- It uses HTML export for Sheets/Excel, HTML, and CSV so tables become explicit HTML sections for the chunker. [Docling docs](https://docling-project.github.io/docling/)
- Audio inputs (MP3/WAV/VTT) go through the same converter and yield a transcript section. [Docling docs](https://docling-project.github.io/docling/)

#### 3.2 Email Parser (`core/parsers/email.py`)

```python
from typing import Any
import email
import json
from email import policy
from models.parsed import ParsedDocument, ParsedSection


class EmailParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        msg = email.message_from_bytes(data, policy=policy.default)

        subject = msg.get("subject", "")
        from_addr = msg.get("from", "")
        to_addr = msg.get("to", "")
        cc_addr = msg.get("cc", "")
        date = msg.get("date", "")

        body = ""
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_content()
                if part.get_filename():
                    attachments.append(part.get_filename())
        else:
            body = msg.get_content()

        header_json = {
            "subject": subject,
            "from": from_addr,
            "to": to_addr,
            "cc": cc_addr,
            "date": date,
            "attachments": attachments,
        }
        sections = [
            ParsedSection(text=json.dumps(header_json, indent=2), section_type="header", content_format="json"),
            ParsedSection(text=body or "", section_type="body", content_format="text"),
        ]
        title = subject or metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata["doc_id"],
            source_name=name,
            file_type="email",
            title=title,
            sections=sections,
            raw_text=body or "",
            metadata=metadata,
        )
```

#### 3.3 Video Parser (`core/parsers/video.py`)

```python
from typing import Any
from models.parsed import ParsedDocument, ParsedSection


class VideoParser:
    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        text = "[VIDEO TRANSCRIPTION PENDING]"
        sections = [ParsedSection(text=text, section_type="text", content_format="text")]
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
The key rule: **normalize without flattening**. Preserve section boundaries, content format (text/html/json), and structural metadata so chunking can remain type-aware.

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

### Step 6: Chunking (Schema-Aware)

We use `langchain-text-splitters` for plain text, but **tables and JSON are chunked differently** to preserve relationships (table rows stay together, JSON isn’t split arbitrarily).

Create `core/chunking/chunker.py`:

```python
from typing import List
import uuid
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.parsed import ParsedDocument, ChunkedDocument


class DocumentChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def _split_html_table(self, html: str, chunk_rows: int = 200) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return [html]
        header = rows[0]
        body_rows = rows[1:] if len(rows) > 1 else []
        chunks = []
        for i in range(0, len(body_rows), chunk_rows):
            tbl = BeautifulSoup("<table></table>", "html.parser")
            table = tbl.table
            table.append(header)
            for r in body_rows[i : i + chunk_rows]:
                table.append(r)
            chunks.append(str(table))
        return chunks or [html]

    def chunk(self, doc: ParsedDocument) -> list[ChunkedDocument]:
        chunks: list[ChunkedDocument] = []
        chunk_index = 0

        for section in doc.sections:
            if section.content_format == "html" and section.section_type == "table":
                splits = self._split_html_table(section.text, chunk_rows=doc.metadata.get("sheet_chunk_rows", 200))
            elif section.content_format == "json":
                splits = [section.text]
            else:
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
                        content_format=section.content_format,
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
            "content_format": {"type": "keyword"},
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

| File Type | Parser Used | Intermediate Representation | Library | RAGFlow Reference |
|-----------|-------------|-----------------------------|---------|-------------------|
| PPT/PPTX, PDF, DOC/DOCX, TXT/MD, Images | Docling parser | Docling lossless JSON export | Docling | `deepdoc/parser/*` |
| Sheets/Excel, HTML, CSV | Docling parser | Docling HTML export split into table + text sections | Docling | `deepdoc/parser/*` |
| Audio (MP3/WAV/VTT) | Docling parser | Docling transcript section | Docling | `rag/flow/parser/parser.py` (audio) |
| Email | Email parser | Header JSON + body text | stdlib `email` | `rag/flow/parser/parser.py` (email) |
| Video | Video parser (placeholder) | Placeholder transcript | custom | `rag/flow/parser/parser.py` (video) |
| Google Docs | Export -> DOCX -> Docling | DoclingDocument | Google API + Docling | `common/data_source/google_drive/*` |
| Google Sheets | Export -> XLSX -> Docling | DoclingDocument | Google API + Docling | `common/data_source/google_drive/*` |
| Google Slides | Export -> PPTX -> Docling | DoclingDocument | Google API + Docling | `common/data_source/google_drive/*` |

If a file type has no stable library and no easy raw parser, then the fallback is:
- Export to a supported format (Google native types).
- Use Docling OCR for images or scanned PDFs.

---

## 6. Chunking Strategy (Docling HybridChunker + HTML Splitter)

Why chunking matters:
- LLMs have context limits.
- Chunking affects recall, precision, and citation quality.

Use **Docling’s HybridChunker** for JSON paths (PPT/PPTX, PDF, DOC/DOCX, TXT/MD, Images) because it chunks the **DoclingDocument** with structure‑aware serialization. For HTML paths (Sheets/Excel, HTML, CSV), keep table‑aware HTML splitting and `RecursiveCharacterTextSplitter` for remaining text. See Docling’s advanced chunking example for the HybridChunker setup. [Docling advanced chunking](https://docling-project.github.io/docling/examples/advanced_chunking_and_serialization/)

Create `core/chunking/chunker.py`:

```python
from typing import List
import tempfile
import uuid
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from models.parsed import ParsedDocument, ChunkedDocument


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer: BaseTokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
)


class DocumentChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self.hybrid_chunker = HybridChunker(tokenizer=tokenizer)

    def _split_html_table(self, html: str, chunk_rows: int = 200) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")
        if not rows:
            return [html]
        header = rows[0]
        body_rows = rows[1:] if len(rows) > 1 else []
        chunks = []
        for i in range(0, len(body_rows), chunk_rows):
            tbl = BeautifulSoup("<table></table>", "html.parser")
            table = tbl.table
            table.append(header)
            for r in body_rows[i : i + chunk_rows]:
                table.append(r)
            chunks.append(str(table))
        return chunks or [html]

    def _chunk_docling_json(self, json_text: str) -> list[str]:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True, mode="w") as tmp:
            tmp.write(json_text)
            tmp.flush()
            doc = DoclingDocument.load_from_json(tmp.name)
        chunks = []
        for chunk in self.hybrid_chunker.chunk(dl_doc=doc):
            chunks.append(self.hybrid_chunker.contextualize(chunk=chunk))
        return chunks

    def chunk(self, doc: ParsedDocument) -> list[ChunkedDocument]:
        chunks: list[ChunkedDocument] = []
        chunk_index = 0

        for section in doc.sections:
            if section.content_format == "json":
                splits = self._chunk_docling_json(section.text)
                content_format = "text"
            elif section.content_format == "html" and section.section_type == "table":
                splits = self._split_html_table(section.text, chunk_rows=doc.metadata.get("sheet_chunk_rows", 200))
                content_format = "html"
            else:
                splits = self.splitter.split_text(section.text)
                content_format = section.content_format

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
                        content_format=content_format,
                        page=section.page,
                        title=section.title or doc.title,
                        metadata=section.metadata | doc.metadata,
                    )
                )
                chunk_index += 1
        return chunks
```

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
4. Produce a type-specific intermediate representation.
5. OCR embedded images (if present).
6. Normalize sections (without flattening structure).
7. Chunk with schema-aware splitter.
8. Insert into new ES index.
9. Update Postgres metadata status.

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

