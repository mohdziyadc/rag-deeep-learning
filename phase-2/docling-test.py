

import argparse
import json
import sys

from bs4 import BeautifulSoup

from docling.document_converter import DocumentConverter

from models.parsed import ParsedDocument, ParsedSection


def scrape(url: str) -> str:
    converter = DocumentConverter()
    result = converter.convert(url)
    return result.document.export_to_html()


def convert_html_to_doc():

    with open('scrape-output.html', 'r', encoding='utf-8') as html_file:
        raw_html = html_file.read()
    
    soup = BeautifulSoup(raw_html, 'html.parser')

    sections: list[ParsedSection] = []

    for table in soup.find_all('table'):
        sections.append(
            ParsedSection(
                text=str(table),
                section_type="table",
                metadata={"docling": True}
            )
        )
        table.decompose()
    
    text = soup.get_text("\n").strip()
    
    if text:
        sections.append(
            ParsedSection(
                text=text,
                section_type="text",
                metadata={"docling": True}
            )
        )
    
    raw_text = "\n".join([s.text for s in sections if s.section_type == 'text'])

    res = ParsedDocument(
        doc_id="1",
        source_name="jcole",
        file_type='html',
        title='JCOLE FALL-OFF',
        sections=sections,
        raw_text=raw_text,
        metadata={"docling": True}
    )

    with open('parsed-doc.json', 'w', encoding='utf-8') as parsed:
        json.dump(res.model_dump(mode='json'), parsed, indent=4)

        print(f"Parsed Document success")
    


def main() -> int:
    # parser = argparse.ArgumentParser(
    #     description="Scrape a URL with Docling and output HTML."
    # )
    # parser.add_argument("url", help="URL to scrape (HTML, PDF, etc.)")
    # args = parser.parse_args()

    # try:
    #     html = scrape(args.url)
    # except Exception as exc:  # noqa: BLE001
    #     print(f"Docling scrape failed: {exc}", file=sys.stderr)
    #     return 1

    # output_path = "scrape-output.html"
    # with open(output_path, "w", encoding="utf-8") as html_file:
    #     html_file.write(html)
    # print(f"HTML successfully dumped to {output_path}")
    # return 0

    convert_html_to_doc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())