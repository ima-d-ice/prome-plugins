"""Generate deterministic synthetic DOCX/PPTX fixtures for parser benchmarks.

Run:  python benchmarks/parsers/office/generate_fixtures.py
Fixtures are committed to the repo; rerun only when the corpus changes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from pptx import Presentation
from pptx.util import Inches as PptxInches
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"


def _png_bytes(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_simple_text() -> None:
    doc = Document()
    doc.core_properties.title = "Quarterly Operations Brief"
    doc.core_properties.author = "A. Example"
    doc.core_properties.subject = "Operations"
    doc.add_paragraph("The quarter closed on a steady note.")
    doc.add_paragraph("Revenue grew eighteen percent year over year.")
    doc.add_paragraph("Hiring focused on infrastructure and support.")
    doc.save(FIXTURES / "simple_text.docx")


def make_headings() -> None:
    doc = Document()
    doc.add_heading("Prome Platform Overview", level=1)
    doc.add_paragraph("Intro paragraph under the top heading.")
    doc.add_heading("Architecture", level=2)
    doc.add_paragraph("Architecture overview paragraph.")
    doc.add_heading("Storage Layer", level=3)
    doc.add_paragraph("Details about the storage layer.")
    doc.add_heading("Indexing", level=4)
    doc.add_paragraph("Details about indexing.")
    doc.add_heading("Operations", level=2)
    doc.add_paragraph("Operations overview paragraph.")
    doc.save(FIXTURES / "headings.docx")


def make_nested_lists() -> None:
    doc = Document()
    doc.add_paragraph("Grocery list:")
    doc.add_paragraph("Apples", style="List Bullet")
    doc.add_paragraph("Bread", style="List Bullet")
    doc.add_paragraph("Almond flour", style="List Bullet 2")
    doc.add_paragraph("Rye flour", style="List Bullet 2")
    doc.add_paragraph("Eggs", style="List Bullet")
    doc.add_paragraph("Build steps:")
    doc.add_paragraph("Install dependencies", style="List Number")
    doc.add_paragraph("Run the test suite", style="List Number")
    doc.add_paragraph("Unit tests", style="List Number 2")
    doc.add_paragraph("Integration tests", style="List Number 2")
    doc.add_paragraph("Ship the release", style="List Number")
    doc.save(FIXTURES / "nested_lists.docx")


def make_tables() -> None:
    doc = Document()
    doc.add_heading("Invoice Summary", level=1)
    table = doc.add_table(rows=3, cols=3)
    table.style = "Table Grid"
    for i, cell_text in enumerate(["Item", "Quantity", "Total"]):
        table.rows[0].cells[i].text = cell_text
    for i, row in enumerate([["Widget Pro", "12", "239.88"], ["Gadget Mini", "7", "69.93"]], start=1):
        for j, cell_text in enumerate(row):
            table.rows[i].cells[j].text = cell_text
    doc.add_paragraph("Second table follows.")
    table2 = doc.add_table(rows=2, cols=2)
    table2.style = "Table Grid"
    table2.rows[0].cells[0].text = "Plan"
    table2.rows[0].cells[1].text = "Seats"
    table2.rows[1].cells[0].text = "Enterprise"
    table2.rows[1].cells[1].text = "250"
    doc.save(FIXTURES / "tables.docx")


def make_images_captions() -> None:
    doc = Document()
    doc.add_heading("Figure Set", level=1)
    doc.add_picture(io.BytesIO(_png_bytes(64, 64, (200, 30, 30))), width=Inches(1.2))
    doc.add_paragraph("Figure 1: Quarterly growth chart.")
    doc.add_paragraph("Closing remarks follow the figure.")
    doc.save(FIXTURES / "images_captions.docx")


def make_headers_footers() -> None:
    doc = Document()
    section = doc.sections[0]
    section.header.paragraphs[0].text = "Prome Confidential Header"
    section.footer.paragraphs[0].text = "Page footer with legal notice"
    doc.add_heading("Headered Document", level=1)
    doc.add_paragraph("Body content under the header.")
    doc.save(FIXTURES / "headers_footers.docx")


def make_sections() -> None:
    doc = Document()
    doc.add_heading("First Section", level=1)
    doc.add_paragraph("Content of the first section.")
    doc.add_section(WD_SECTION.NEW_PAGE)
    doc.add_heading("Second Section", level=1)
    doc.add_paragraph("Content of the second section.")
    doc.save(FIXTURES / "sections.docx")


def make_unicode() -> None:
    doc = Document()
    doc.add_heading("Unicode Showcase", level=1)
    doc.add_paragraph("Café au lait — naïve résumé.")
    doc.add_paragraph("日本語のテキストと中文文本。")
    doc.add_paragraph("Ελληνικά και Русский текст.")
    doc.save(FIXTURES / "unicode.docx")


def make_empty() -> None:
    doc = Document()
    doc.save(FIXTURES / "empty.docx")


def make_malformed() -> None:
    (FIXTURES / "malformed.docx").write_bytes(b"This is not a real docx file at all.")


def make_hyperlinks() -> None:
    doc = Document()
    doc.add_heading("Reference List", level=1)
    paragraph = doc.add_paragraph("See the ")
    part = paragraph.part
    r_id = part.relate_to(
        "https://example.com/prome-reference",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    from docx.oxml.shared import OxmlElement
    from docx.oxml.ns import qn

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "Prome reference guide"
    run.append(text)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
    doc.add_paragraph("End of references.")
    doc.save(FIXTURES / "hyperlinks.docx")


def _blank_presentation() -> Presentation:
    return Presentation()


def make_multi_slide() -> None:
    prs = _blank_presentation()
    prs.core_properties.title = "Prome Launch Review"
    prs.core_properties.author = "B. Example"
    layout = prs.slide_layouts[1]  # title + content
    for idx, (title, body) in enumerate(
        [("Launch Overview", "We shipped the platform."),
         ("Adoption Metrics", "Adoption doubled quarter over quarter."),
         ("Next Steps", "Expand the parser coverage.")],
        start=1,
    ):
        slide = prs.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    prs.save(FIXTURES / "multi_slide.pptx")


def make_text_boxes() -> None:
    prs = _blank_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    box = slide.shapes.add_textbox(PptxInches(1), PptxInches(1), PptxInches(6), PptxInches(4))
    tf = box.text_frame
    tf.text = "Top level point"
    p2 = tf.add_paragraph()
    p2.text = "Nested bullet point"
    p2.level = 1
    p3 = tf.add_paragraph()
    p3.text = "Second top level point"
    p3.level = 0
    prs.save(FIXTURES / "text_boxes.pptx")


def make_table_slide() -> None:
    prs = _blank_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    gfx_shape = slide.shapes.add_table(3, 3, PptxInches(0.5), PptxInches(1), PptxInches(6), PptxInches(2))
    table = gfx_shape.table
    values = [["Region", "Rep", "Quota"], ["North", "Diaz", "120000"], ["South", "Nguyen", "95000"]]
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            table.cell(i, j).text = value
    prs.save(FIXTURES / "table.pptx")


def make_images_notes() -> None:
    prs = _blank_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_picture(io.BytesIO(_png_bytes(80, 80, (30, 30, 200))), PptxInches(1), PptxInches(1))
    slide.notes_slide.notes_text_frame.text = "Remind the audience about the roadmap."
    prs.save(FIXTURES / "images_notes.pptx")


def make_shapes() -> None:
    prs = _blank_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    shape = slide.shapes.add_shape(1, PptxInches(1), PptxInches(1), PptxInches(3), PptxInches(1.5))  # rectangle
    shape.text_frame.text = "Shape callout text"
    prs.save(FIXTURES / "shapes.pptx")


def make_unicode_pptx() -> None:
    prs = _blank_presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(PptxInches(1), PptxInches(1), PptxInches(6), PptxInches(3))
    box.text_frame.text = "Café — 日本語 — Ελληνικά"
    prs.save(FIXTURES / "unicode.pptx")


def make_empty_pptx() -> None:
    prs = _blank_presentation()
    prs.save(FIXTURES / "empty.pptx")


def make_malformed_pptx() -> None:
    (FIXTURES / "malformed.pptx").write_bytes(b"definitely not a pptx file")


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    make_simple_text()
    make_headings()
    make_nested_lists()
    make_tables()
    make_images_captions()
    make_headers_footers()
    make_sections()
    make_unicode()
    make_empty()
    make_malformed()
    make_hyperlinks()
    make_multi_slide()
    make_text_boxes()
    make_table_slide()
    make_images_notes()
    make_shapes()
    make_unicode_pptx()
    make_empty_pptx()
    make_malformed_pptx()
    print(f"Wrote fixtures to {FIXTURES}")


if __name__ == "__main__":
    main()
