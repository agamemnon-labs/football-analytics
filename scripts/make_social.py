"""The 1280x640 preview card used for link previews (repository social image and the demo site).

The strip at the bottom is the real output on the demo match: truth on top, our calls below,
read from site/demo.json. The card is laid out in HTML and screenshotted with a headless
Chromium-based browser (Edge or Chrome), so the brand fonts render properly.

    pip install -r requirements-docs.txt
    python scripts/export_demo.py
    python scripts/make_social.py [--browser PATH]
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile

from build_report import find_browser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")
DEMO = os.path.join(ROOT, "site", "demo.json")
LEVEL = "mid"
COLOURS = {0: "#3987E5", 1: "#D95926", 2: "#5F5E5A"}
STRIP_W = 1136

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500&display=block" rel="stylesheet">
<style>
html, body {{ margin: 0; width: 1280px; height: 640px; overflow: hidden; background: #0A0B0D; color: #F1EDE4; }}
body {{ font-family: Inter, sans-serif; position: relative; }}
.top {{ position: absolute; left: 72px; right: 72px; top: 56px; display: flex; align-items: center; justify-content: space-between; }}
.brand {{ display: flex; align-items: center; gap: 16px; }}
.brand span {{ font-family: Archivo; font-stretch: 125%; font-weight: 800; font-size: 20px; letter-spacing: 0.16em; }}
.meta {{ font-family: 'IBM Plex Mono'; font-size: 16px; color: #8C919B; }}
h1 {{ position: absolute; left: 72px; top: 132px; margin: 0; font-family: Archivo; font-stretch: 125%; font-weight: 800;
      font-size: 96px; line-height: 0.96; letter-spacing: -0.015em; }}
.side {{ position: absolute; left: 760px; right: 72px; top: 150px; }}
.side p {{ margin: 0 0 28px; font-size: 22px; line-height: 1.45; color: #B4B0A6; }}
.figs {{ display: flex; gap: 40px; }}
.figs b {{ display: block; font-family: Archivo; font-stretch: 125%; font-weight: 800; font-size: 40px; line-height: 1; }}
.figs span {{ display: block; margin-top: 8px; font-size: 15px; color: #8C919B; }}
.strip {{ position: absolute; left: 72px; top: 420px; }}
.strip .lab {{ font-family: 'IBM Plex Mono'; font-size: 14px; fill: #8C919B; }}
.cap {{ position: absolute; left: 72px; right: 72px; bottom: 48px; display: flex; justify-content: space-between;
        font-family: 'IBM Plex Mono'; font-size: 15px; color: #8C919B; }}
.cap .url {{ color: #B08A4A; }}
</style></head><body>
<div class="top">
  <div class="brand">
    <svg viewBox="56 36 288 328" width="36" height="41"><polygon fill="#F1EDE4" points="180,36 195,36 195,133.8 108,364 56,364"/><polygon fill="#F1EDE4" points="220,36 205,36 205,133.8 292,364 344,364"/><polygon fill="#B08A4A" points="166.4,236 233.6,236 248.7,276 151.3,276"/></svg>
    <span>AGAMEMNON</span>
  </div>
  <div class="meta">More than a Hack 2027 · Phase 1</div>
</div>
<h1>Who has<br>the ball?</h1>
<div class="side">
  <p>Ball possession called 25&nbsp;times a second from automatic tracking.</p>
  <div class="figs">
    <div><b>0.681</b><span>spellF1, 27 unseen matches</span></div>
    <div><b>12 s</b><span>per match, laptop CPU</span></div>
  </div>
</div>
<svg class="strip" width="{w}" height="80" shape-rendering="crispEdges">{strip}</svg>
<div class="cap"><span>truth vs our calls, 10 minutes of a match the model never saw</span><span class="url">agamemnon-labs.github.io/football-analytics</span></div>
</body></html>
"""


def expand(rle):
    out = []
    for code, length in zip(rle[::2], rle[1::2]):
        out += [code] * length
    return out


def rows(truth, pred):
    """Two rows of rectangles, one per run, labelled on the left."""
    x0, w = 64, STRIP_W - 64
    parts = []
    for y, name, codes in ((4, "truth", truth), (44, "model", pred)):
        parts.append('<text class="lab" x="0" y="{}">{}</text>'.format(y + 21, name))
        n, start = len(codes), 0
        for i in range(1, n + 1):
            if i == n or codes[i] != codes[start]:
                a = x0 + w * start / n
                b = x0 + w * i / n
                parts.append('<rect x="{:.2f}" y="{}" width="{:.2f}" height="30" fill="{}"/>'.format(
                    a, y, max(1.0, b - a), COLOURS.get(codes[start], "#2a2d33")))
                start = i
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--browser", help="path to Edge or Chrome")
    args = parser.parse_args()
    browser = find_browser(args.browser)

    with open(DEMO, encoding="utf-8") as handle:
        demo = json.load(handle)
    html = PAGE.format(w=STRIP_W, strip=rows(expand(demo["truth"]), expand(demo["levels"][LEVEL]["pred"])))

    out = os.path.join(ASSETS, "social.png")
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "card.html")
        with open(page, "w", encoding="utf-8") as handle:
            handle.write(html)
        shot = os.path.join(tmp, "card.png")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        "--window-size=1280,640", "--virtual-time-budget=8000", "--screenshot=" + shot,
                        "file:///" + page.replace("\\", "/")], check=True, capture_output=True)
        shutil.copy(shot, out)
    site = os.path.join(ROOT, "site", "assets")
    if os.path.isdir(site):
        shutil.copy(out, os.path.join(site, "social.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
