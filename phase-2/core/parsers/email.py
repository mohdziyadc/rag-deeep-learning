from typing import Any
import email
import json
from email import policy
from models.parsed import ParsedDocument, ParsedSection

from core.parsers.base import BaseParser


class EmailParser(BaseParser):

    def parse(self, name: str, data: bytes, metadata: dict[str, Any]) -> ParsedDocument:
        
        msg = email.message_from_bytes(data, policy=policy.default)

        subject = msg.get("subject", "")
        from_addr = msg.get("from", "")
        to_addr = msg.get("to", "")
        cc_addr = msg.get("cc", "")
        date = msg.get("date", "")

        body = ""
        body_html = ""
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_maintype() == "multipart":
                    continue

                dispositon = part.get_content_disposition()
                filename = part.get_filename()

                if filename:
                    attachments.append(filename)

                if dispositon == 'attachment':
                    continue
                
                ctype = part.get_content_type()
                if ctype == "text/plain" and not body:
                    body = part.get_content()
                elif ctype == "text/html" and not body_html:
                    body_html = part.get_content()        
        else:
            body = msg.get_content()

        header = {
            "subject": subject,
            "from": from_addr, 
            "to": to_addr,
            "cc": cc_addr,
            "date": date,
            "attachments": attachments
        }

        sections = [
            ParsedSection(
                text=json.dumps(header, indent=2),
                section_type="header",
                content_format="json"
            ),
            ParsedSection(
                text=body,
                section_type="body",
                content_format="text"
            ),
            ParsedSection(
                text=body_html,
                section_type="body_html",
                content_format="html"
            )
        ]

        title = subject or metadata.get("title", name)
        return ParsedDocument(
            doc_id=metadata['doc_id'],
            source_name=name,
            file_type="email",
            title=title,
            sections=sections,
            raw_text=body or body_html,
            metadata=metadata
        )
                    

