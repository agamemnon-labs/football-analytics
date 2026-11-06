"""Render docs/TECHNICAL_REPORT.md to Agamemnon.pdf.

    pip install -r requirements-docs.txt
    python scripts/build_report.py [--out Agamemnon.pdf] [--browser PATH]

Markdown is converted to HTML with python-markdown and printed to PDF by a headless Chromium
browser (Edge or Chrome), so no LaTeX installation is needed.
"""

import argparse
import html
import os
import shutil
import subprocess
import sys
import tempfile

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "docs", "TECHNICAL_REPORT.md")

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt;
       line-height: 1.42; color: #1b1b1b; }
header { border-bottom: 1.5px solid #1b1b1b; padding-bottom: 6pt; margin-bottom: 10pt; }
header h1 { font-size: 17pt; margin: 0 0 3pt 0; }
header .sub { font-size: 10.5pt; color: #444; margin: 0 0 4pt 0; }
header .meta { font-size: 8.8pt; color: #555; margin: 0; }
h1 { font-size: 12.5pt; margin: 13pt 0 4pt 0; page-break-after: avoid; }
p, li { margin: 3pt 0; }
ol, ul { padding-left: 16pt; margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 7pt 0; font-size: 8.6pt;
        page-break-inside: avoid; }
th, td { border-bottom: 0.6px solid #bbb; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { border-bottom: 1px solid #333; }
code { font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 8.6pt; }
pre { background: #f3f3f3; padding: 5pt 7pt; font-size: 8.4pt; page-break-inside: avoid; }
"""


def front_matter(text):
    """Split a YAML-style block of key: "value" lines from the body."""
    meta = {}
    if text.startswith("---"):
        head, body = text[3:].split("\n---", 1)
        for line in head.strip().splitlines():
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
        text = body
    return meta, text


def find_browser(explicit):
    for candidate in ([explicit] if explicit else []) + BROWSERS:
        path = candidate if os.path.isfile(candidate) else shutil.which(candidate)
        if path:
            return path
    raise SystemExit("no Chromium-based browser found; pass --browser PATH")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "Agamemnon.pdf"))
    parser.add_argument("--browser")
    args = parser.parse_args()

    with open(SOURCE, encoding="utf-8") as handle:
        meta, body = front_matter(handle.read())
    content = markdown.markdown(body, extensions=["tables", "fenced_code"])
    page = """<!doctype html><html><head><meta charset="utf-8"><title>{t}</title><style>{css}</style>
</head><body><header><h1>{t}</h1><p class="sub">{s}</p><p class="meta">{a}<br>{d}</p></header>
{c}</body></html>""".format(t=html.escape(meta.get("title", "")), s=html.escape(meta.get("subtitle", "")),
                          a=html.escape(meta.get("author", "")), d=html.escape(meta.get("date", "")),
                          css=CSS, c=content)

    browser = find_browser(args.browser)
    out = os.path.abspath(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        page_path = os.path.join(tmp, "report.html")
        with open(page_path, "w", encoding="utf-8") as handle:
            handle.write(page)
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        "--user-data-dir=" + os.path.join(tmp, "profile"),
                        "--print-to-pdf=" + out, "file:///" + page_path.replace("\\", "/")],
                       check=True, capture_output=True, timeout=120)
    if not os.path.isfile(out):
        sys.exit("the browser did not write {}".format(out))
    print("wrote", out)


if __name__ == "__main__":
    main()
