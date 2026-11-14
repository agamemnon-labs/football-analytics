"""Build the technical document: docs/TECHNICAL_REPORT.md to Agamemnon.docx and Agamemnon.pdf.

    pip install -r requirements-docs.txt
    python scripts/build_report.py [--docx Agamemnon.docx] [--pdf Agamemnon.pdf] [--no-pdf]

The Word file is written with python-docx. The PDF is exported from that same file by Microsoft
Word (Windows) or LibreOffice (anywhere `soffice` is on the path), so the two always match.

The Markdown it understands is what the report uses: front matter, `#` and `##` headings,
paragraphs, numbered lists, pipe tables, fenced code, images, and inline `code`, *italic* and
**bold**. Three conventions carry the layout:

  **Table 1.** text     a paragraph that opens like this is the caption of the table below it
  **Figure 1.** text    the same for the image above it
  <!-- table: widths=30,70 align=l,c -->   column widths in percent and alignment, placed on the
                                           line before a table
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
SOURCE = os.path.join(DOCS, "TECHNICAL_REPORT.md")

BODY, HEAD, MONO = "Cambria", "Calibri", "Consolas"
INK = RGBColor(0x1F, 0x29, 0x33)
GREY = RGBColor(0x5E, 0x63, 0x6E)
ACCENT = "B08A4A"
RULE, HAIRLINE, SHADE = "1F2933", "C9CDD2", "F3F4F6"
MARGIN = Cm(2.3)
TEXT_WIDTH = Cm(21.0) - 2 * MARGIN
NBSP = " "

INLINE = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
DIRECTIVE = re.compile(r"<!--\s*table:\s*(.*?)\s*-->")
ALIGN = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}

# Word wants the children of a properties element in schema order. These are the orders of the
# elements this script adds, with the neighbours python-docx may already have written.
ORDER = {
    "pPr": ("pStyle", "keepNext", "keepLines", "pageBreakBefore", "widowControl", "numPr", "pBdr",
            "shd", "tabs", "spacing", "ind", "contextualSpacing", "jc", "outlineLvl", "rPr"),
    "tblPr": ("tblStyle", "tblW", "jc", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar",
              "tblLook"),
    "tcPr": ("tcW", "gridSpan", "vMerge", "tcBorders", "shd", "tcMar", "vAlign"),
}


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


def tidy(text):
    """Keep a number with its unit, and a reference word with its number, on one line."""
    text = re.sub(r"(?<=\d) (?=(?:s|m|h|M|fps|GB|m/s)\b)", NBSP, text)
    text = text.replace("A-League", "A‑League")
    return re.sub(r"\b(Table|Figure|Section|Law) (?=\d)", r"\1" + NBSP, text)


def fonts(target, name):
    """Set a font on a style or run for every script, so Word does not fall back."""
    target.font.name = name
    props = target.element.get_or_add_rPr()
    rfonts = props.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        props.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)


def add_inline(paragraph, text, size=None, colour=None, bold=None):
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("`"):
            run = paragraph.add_run(piece[1:-1])
            fonts(run, MONO)
            run.font.size = Pt((size.pt if size else 10.5) - 0.75)
        elif piece.startswith("**"):
            run = paragraph.add_run(tidy(piece[2:-2]))
            run.bold = True
        elif piece.startswith("*"):
            run = paragraph.add_run(tidy(piece[1:-1]))
            run.italic = True
        else:
            run = paragraph.add_run(tidy(piece))
        if size and not piece.startswith("`"):
            run.font.size = size
        if colour is not None:
            run.font.color.rgb = colour
        if bold:
            run.bold = True
    return paragraph


def place(parent, element):
    """Insert a child where the schema wants it, so Word opens the file without repairs."""
    order = ORDER[parent.tag.split("}")[1]]
    name = element.tag.split("}")[1]
    later = ["w:" + tag for tag in order[order.index(name) + 1:]]
    parent.insert_element_before(element, *later)


def border(parent, tag, edges):
    """Attach a border element (pBdr, tblBorders or tcBorders) with the given edges."""
    element = OxmlElement(tag)
    for edge, (size, colour) in edges.items():
        line = OxmlElement("w:" + edge)
        line.set(qn("w:val"), "single" if size else "nil")
        line.set(qn("w:sz"), str(size))
        line.set(qn("w:space"), "0")
        line.set(qn("w:color"), colour)
        element.append(line)
    place(parent, element)
    return element


def shade(props, fill):
    shading = OxmlElement("w:shd")
    shading.set(qn("w:val"), "clear")
    shading.set(qn("w:color"), "auto")
    shading.set(qn("w:fill"), fill)
    place(props, shading)


def flag(parent, tag):
    parent.append(OxmlElement(tag))


def add_table(doc, lines, spec):
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    columns = len(rows[0])
    widths = [float(w) for w in spec.get("widths", ",".join(["1"] * columns)).split(",")]
    aligns = spec.get("align", ",".join(["l"] * columns)).split(",")
    if len(widths) != columns or len(aligns) != columns:
        raise SystemExit("table directive does not match {} columns: {}".format(columns, rows[0]))
    widths = [int(TEXT_WIDTH * w / sum(widths)) for w in widths]

    table = doc.add_table(rows=len(rows), cols=columns)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    props = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    place(props, layout)
    border(props, "w:tblBorders", {"top": (10, RULE), "bottom": (10, RULE),
                                   "insideH": (4, HAIRLINE)})
    margins = OxmlElement("w:tblCellMar")
    for side, value in (("top", 55), ("left", 90), ("bottom", 55), ("right", 90)):
        node = OxmlElement("w:" + side)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    place(props, margins)

    for i, row in enumerate(rows):
        row_props = table.rows[i]._tr.get_or_add_trPr()
        flag(row_props, "w:cantSplit")
        if i == 0:
            flag(row_props, "w:tblHeader")
        for j, text in enumerate(row):
            cell = table.cell(i, j)
            cell.width = widths[j]
            paragraph = cell.paragraphs[0]
            paragraph.style = doc.styles["Table Text"]
            paragraph.alignment = ALIGN[aligns[j]]
            add_inline(paragraph, text, size=Pt(9.5), bold=(i == 0))
            if i == 0:
                border(cell._tc.get_or_add_tcPr(), "w:tcBorders", {"bottom": (8, RULE)})
            if i < len(rows) - 1:
                paragraph.paragraph_format.keep_with_next = True
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.line_spacing = Pt(6)


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


def add_figure(doc, src):
    width = Cm(8.6) if "demo" in src else TEXT_WIDTH
    doc.add_picture(os.path.join(DOCS, src), width=width)
    paragraph = doc.paragraphs[-1]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(8)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.keep_with_next = True


def add_caption(doc, text, above):
    """A table caption sits above its table, a figure caption below its image."""
    paragraph = add_inline(doc.add_paragraph(style="Caption"), text, size=Pt(9))
    if above:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        paragraph.paragraph_format.space_before = Pt(8)
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.paragraph_format.keep_with_next = True
    else:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(10)


def add_code(doc, lines):
    paragraph = doc.add_paragraph(style="Code")
    for i, line in enumerate(lines[1:-1]):
        run = paragraph.add_run(line)
        if i < len(lines) - 3:
            run.add_break(WD_BREAK.LINE)


def page_field(paragraph):
    """Insert a PAGE field, which Word fills in with the page number."""
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        run = paragraph.add_run()
        if kind:
            char = OxmlElement("w:fldChar")
            char.set(qn("w:fldCharType"), kind)
            run._r.append(char)
        else:
            instruction = OxmlElement("w:instrText")
            instruction.set(qn("xml:space"), "preserve")
            instruction.text = text
            run._r.append(instruction)
        run.font.size = Pt(9)
        run.font.color.rgb = GREY
        fonts(run, HEAD)


def setup(doc, meta):
    section = doc.sections[0]
    section.start_type = WD_SECTION.NEW_PAGE
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, MARGIN)
    section.header_distance = section.footer_distance = Cm(1.1)
    section.different_first_page_header_footer = True

    styles = doc.styles
    normal = styles["Normal"]
    fonts(normal, BODY)
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.widow_control = True

    for name, size, before, after in (("Heading 1", 14, 14, 6), ("Heading 2", 11.5, 9, 4)):
        style = styles[name]
        fonts(style, HEAD)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.italic = False
        style.font.color.rgb = INK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    caption = styles["Caption"]
    fonts(caption, HEAD)
    caption.font.size = Pt(9)
    caption.font.bold = False
    caption.font.italic = False
    caption.font.color.rgb = GREY
    caption.paragraph_format.line_spacing = 1.1

    numbered = styles["List Number"]
    numbered.paragraph_format.space_after = Pt(4)

    table_text = styles.add_style("Table Text", 1)
    table_text.base_style = normal
    fonts(table_text, HEAD)
    table_text.font.size = Pt(9.5)
    table_text.paragraph_format.space_after = Pt(0)
    table_text.paragraph_format.line_spacing = 1.08
    table_text.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    code = styles.add_style("Code", 1)
    code.base_style = normal
    fonts(code, MONO)
    code.font.size = Pt(9)
    code.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    code.paragraph_format.line_spacing = 1.2
    code.paragraph_format.left_indent = code.paragraph_format.right_indent = Cm(0.25)
    code.paragraph_format.space_before = Pt(4)
    code.paragraph_format.space_after = Pt(10)
    code.paragraph_format.keep_together = True
    code_props = code.element.get_or_add_pPr()
    box = border(code_props, "w:pBdr",
                 {edge: (4, HAIRLINE) for edge in ("top", "left", "bottom", "right")})
    for line in box:
        line.set(qn("w:space"), "6")
    shade(code_props, SHADE)

    reference = styles.add_style("Reference", 1)
    reference.base_style = normal
    reference.font.size = Pt(9.5)
    reference.paragraph_format.left_indent = Cm(0.7)
    reference.paragraph_format.first_line_indent = Cm(-0.7)
    reference.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    reference.paragraph_format.space_after = Pt(4)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = header.add_run("Team Agamemnon  ·  " + meta.get("subtitle", "").split("·")[0].strip())
    fonts(run, HEAD)
    run.font.size = Pt(8.5)
    run.font.color.rgb = GREY
    border(header._p.get_or_add_pPr(), "w:pBdr", {"bottom": (4, HAIRLINE)})
    for footer in (section.footer, section.first_page_footer):
        footer.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        page_field(footer.paragraphs[0])


def title_block(doc, meta):
    for key, font, size, colour, bold, after in (
            ("title", HEAD, 20, INK, True, 4), ("subtitle", HEAD, 12, GREY, False, 12),
            ("author", HEAD, 11.5, INK, False, 2), ("affiliation", HEAD, 10.5, GREY, False, 0),
            ("date", HEAD, 10.5, GREY, False, 8)):
        if not meta.get(key):
            continue
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(after)
        paragraph.paragraph_format.line_spacing = 1.05
        run = paragraph.add_run(meta[key])
        fonts(run, font)
        run.font.size = Pt(size)
        run.font.color.rgb = colour
        run.bold = bold
    rule = doc.add_paragraph()
    rule.paragraph_format.space_after = Pt(4)
    rule.paragraph_format.line_spacing = Pt(4)
    border(rule._p.get_or_add_pPr(), "w:pBdr", {"bottom": (12, ACCENT)})


def write_docx(path):
    with open(SOURCE, encoding="utf-8") as handle:
        meta, body = front_matter(handle.read())
    doc = Document()
    setup(doc, meta)
    doc.core_properties.title = meta.get("title", "")
    doc.core_properties.subject = meta.get("subtitle", "")
    doc.core_properties.author = "Team Agamemnon"
    title_block(doc, meta)

    for block in blocks(body):
        first = block[0]
        text = " ".join(line.strip() for line in block)
        spec = {}
        directive = DIRECTIVE.fullmatch(first.strip())
        if directive:
            spec = dict(part.split("=", 1) for part in directive.group(1).split())
            block, first = block[1:], block[1]
        if first.startswith("## "):
            doc.add_heading(tidy(first[3:].strip()), level=2)
        elif first.startswith("# "):
            doc.add_heading(tidy(first[2:].strip()), level=1)
        elif first.startswith("```"):
            add_code(doc, block)
        elif first.startswith("|"):
            add_table(doc, block, spec)
        elif re.match(r"\d+\.\s", first):
            add_list(doc, block)
        elif IMAGE.fullmatch(text):
            add_figure(doc, IMAGE.fullmatch(text).group(2))
        elif text.startswith("**Table"):
            add_caption(doc, text, above=True)
        elif text.startswith("**Figure"):
            add_caption(doc, text, above=False)
        elif re.match(r"\[\d+\] ", text):
            add_inline(doc.add_paragraph(style="Reference"), text, size=Pt(9.5))
        else:
            add_inline(doc.add_paragraph(), text)
    doc.save(path)
    print("wrote", path)


WORD_EXPORT = """
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {{
    $doc = $word.Documents.Open('{src}', $false, $true)
    $doc.ExportAsFixedFormat('{out}', 17, $false, 0, 0, 0, 0, 0, $true, $true, 1)
    $doc.Close($false)
}} finally {{
    $word.Quit()
}}
"""


def write_pdf(docx, pdf):
    """Export the Word file to PDF with Word itself, or with LibreOffice where there is no Word."""
    docx, pdf = os.path.abspath(docx), os.path.abspath(pdf)
    if os.path.isfile(pdf):
        os.remove(pdf)
    soffice = shutil.which("soffice")
    if sys.platform == "win32" and shutil.which("powershell"):
        script = WORD_EXPORT.format(src=docx.replace("'", "''"), out=pdf.replace("'", "''"))
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       check=False, capture_output=True, timeout=300)
    if not os.path.isfile(pdf) and soffice:
        folder = os.path.dirname(pdf)
        subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", folder, docx],
                       check=True, capture_output=True, timeout=300)
        made = os.path.join(folder, os.path.splitext(os.path.basename(docx))[0] + ".pdf")
        if made != pdf:
            os.replace(made, pdf)
    if not os.path.isfile(pdf):
        sys.exit("could not export the PDF: this needs Microsoft Word or LibreOffice (soffice)")
    print("wrote", pdf)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docx", default=os.path.join(ROOT, "Agamemnon.docx"))
    parser.add_argument("--pdf", default=os.path.join(ROOT, "Agamemnon.pdf"))
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()
    write_docx(args.docx)
    if not args.no_pdf:
        write_pdf(args.docx, args.pdf)


if __name__ == "__main__":
    main()
