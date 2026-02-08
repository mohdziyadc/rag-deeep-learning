from models.parsed import ParsedDocument, ParsedSection

class DocumentNormalizer:
    def normalize(self, doc: ParsedDocument) -> ParsedDocument:
        normalized_sections = []

        for section in doc.sections:
            text = section.text.strip()
            if not text:
                continue
            normalized_sections.append(section)
        doc.sections = normalized_sections
        doc.raw_text = "\n".join([s.text for s in normalized_sections])
        return doc