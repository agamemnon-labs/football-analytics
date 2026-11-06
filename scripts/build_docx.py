"""Write docs/TECHNICAL_REPORT.md as an editable Word file, Agamemnon.docx.

    pip install -r requirements-docs.txt
    python scripts/build_docx.py [--out Agamemnon.docx]

Handles the Markdown the report actually uses: front matter, level-1 headings, paragraphs,
numbered lists, pipe tables, fenced code, images with their alt text as a caption, and inline
`code`, *italic* and **bold**. Anything fancier should be added here when the report needs it.
"""

import argparse
import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
SOURCE = os.path.join(DOCS, "TECHNICAL_REPORT.md")
BODY_FONT, CODE_FONT = "Calibri", "Consolas"
GREY = RGBColor(0x44, 0x44, 0x44)
INLINE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)")


def front_matter(text):
    meta = {}
    if text.startswith("---"):
        head, text = text[3:].split("\n---", 1)
        for line in head.strip().splitlines():
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
    return meta, text


def blocks(text):
    """Split the body into blocks, keeping fenced code together even across blank lines."""
    out, current, fenced = [], [], False
    for line in text.splitlines():
        if line.startswith("```"):
            current.append(line)
            if fenced:
                out.append(current)
                current = []
            fenced = not fenced
            continue
        if not line.strip() and not fenced:
            if current:
                out.append(current)
                current = []
            continue
        current.append(line)
    if current:
        out.append(current)
    return out


def add_inline(paragraph, text, size=None, colour=None):
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("`"):
            run = paragraph.add_run(piece[1:-1])
            run.font.name = CODE_FONT
            run.font.size = Pt(9)
        elif piece.startswith("**"):
            run = paragraph.add_run(piece[2:-2])
            run.bold = True
        elif piece.startswith("*"):
            run = paragraph.add_run(piece[1:-1])
            run.italic = True
        else:
            run = paragraph.add_run(piece)
        if size and not piece.startswith("`"):
            run.font.size = size
        if colour is not None:
            run.font.color.rgb = colour


def shade(cell, fill):
    props = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    props.append(shading)


def add_table(doc, lines):
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            add_inline(cell.paragraphs[0], text, size=Pt(8.5))
            if i == 0:
                for run in cell.paragraphs[0].runs:
                    run.bold = True
                shade(cell, "EDEDEA")
    doc.add_paragraph()


def add_list(doc, lines):
    items = []
    for line in lines:
        m = re.match(r"\d+\.\s+(.*)", line)
        if m:
            items.append(m.group(1))
        else:
            items[-1] += " " + line.strip()
    for item in items:
        add_inline(doc.add_paragraph(style="List Number"), item)


def add_figure(doc, alt, src):
    path = os.path.join(DOCS, src)
    width = Cm(12) if "demo" in src else Cm(16)
    doc.add_picture(path, width=width)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption = doc.add_paragraph(style="Caption")
    add_inline(caption, alt, size=Pt(8.5), colour=GREY)


def add_code(doc, lines):
    for line in lines[1:-1]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(0.5)
        run = p.add_run(line)
        run.font.name = CODE_FONT
        run.font.size = Pt(9)
    doc.add_paragraph()


def setup(doc):
    section = doc.sections[0]
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Cm(2.0))
    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(4)
    normal.paragraph_format.line_spacing = 1.1
    for name, size in (("Title", 20), ("Heading 1", 13)):
        style = doc.styles[name]
        style.font.name = BODY_FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0x1B, 0x1B, 0x1B)
        style.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 0)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "Agamemnon.docx"))
    args = parser.parse_args()
    with open(SOURCE, encoding="utf-8") as handle:
        meta, body = front_matter(handle.read())

    doc = Document()
    setup(doc)
    doc.core_properties.title = meta.get("title", "")
    doc.core_properties.author = "Team Agamemnon"
    doc.add_paragraph(meta.get("title", ""), style="Title")
    for key, size in (("subtitle", Pt(11.5)), ("author", Pt(9.5)), ("date", Pt(9.5))):
        p = doc.add_paragraph()
        add_inline(p, meta.get(key, ""), size=size, colour=GREY)

    for block in blocks(body):
        first = block[0]
        if first.startswith("# "):
            doc.add_heading(first[2:].strip(), level=1)
        elif first.startswith("```"):
            add_code(doc, block)
        elif first.startswith("|"):
            add_table(doc, block)
        elif re.match(r"\d+\.\s", first):
            add_list(doc, block)
        elif re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", " ".join(block).strip()):
            alt, src = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", " ".join(block).strip()).groups()
            add_figure(doc, alt, src)
        else:
            add_inline(doc.add_paragraph(), " ".join(line.strip() for line in block))

    doc.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
