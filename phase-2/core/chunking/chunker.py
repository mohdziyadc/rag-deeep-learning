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

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer: BaseTokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL)
)

class DocumentChunker:

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
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
            for row in body_rows[i: i + chunk_rows]:
                table.append(row)
            chunks.append(str(table))
        
        return chunks or [html]
    

    def _chunk_docling_json(self, json_text: str) -> list[str]:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True, mode="w") as tmp:
            tmp.write(json_text)
            tmp.flush()
            doc = DoclingDocument.load_from_json(tmp.name)
        
        chunks = []
        
        for chunk in self.hybrid_chunker.chunk(dl_doc=doc):
            contextualized_chunk = self.hybrid_chunker.contextualize(chunk=chunk)
            chunks.append(contextualized_chunk)
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
            
            for item in splits:
                chunks.append(
                    ChunkedDocument(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc.doc_id,
                        content=item,
                        chunk_index=chunk_index,
                        source_name=doc.source_name,
                        file_type=doc.file_type,
                        section_type=section.section_type,
                        content_format=content_format,
                        page=section.page,
                        title=section.title or doc.title,
                        metadata=section.metadata or doc.metadata
                    )
                )
                chunk_index += 1
        return chunks