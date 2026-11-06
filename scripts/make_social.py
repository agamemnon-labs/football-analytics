"""The 1280x640 preview card used for link previews (repository social image and the demo site).

    python scripts/make_broadcast.py --still docs/assets/broadcast-still.png
    python scripts/make_social.py
"""

import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")
BG, CREAM, GOLD, MUTED = "#0b0c0d", "#f2efe6", "#b08a4e", "#9a968c"


def mark(ax, x, y, h):
    """The team mark, redrawn from its 512 px artwork so it stays sharp at any size."""
    k = h / 572.0
    shapes = (([(37, 535), (117, 535), (248, 185), (248, 37), (225, 37)], CREAM),
              ([(264, 37), (287, 37), (475, 535), (395, 535), (264, 185)], CREAM),
              ([(205, 341), (307, 341), (330, 402), (182, 402)], GOLD))
    for pts, colour in shapes:
        ax.add_patch(Polygon([(x + px * k, y + (572 - py) * k) for px, py in pts], closed=True,
                             facecolor=colour, lw=0))


def main():
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 640)
    ax.axis("off")

    still = mpimg.imread(os.path.join(ASSETS, "broadcast-still.png"))
    frame = FancyBboxPatch((560, 108), 680, 382, boxstyle="round,pad=0,rounding_size=14",
                           fill=False, ec="#2a2e33", lw=2, zorder=2)
    ax.add_patch(frame)
    img = ax.imshow(still, extent=(560, 1240, 108, 490), zorder=1)
    img.set_clip_path(frame)

    mark(ax, 60, 538, 54)
    ax.text(118, 565, "AGAMEMNON", fontsize=17, color=CREAM, weight="bold", va="center")
    ax.text(60, 448, "Who has", fontsize=58, color=CREAM, weight="bold", va="center")
    ax.text(60, 372, "the ball?", fontsize=58, color=CREAM, weight="bold", va="center")
    ax.text(60, 290, "Frame-level ball possession from\nautomatic tracking, 25 calls a second.",
            fontsize=16, color=MUTED, va="center", linespacing=1.5)
    ax.text(60, 200, "0.681", fontsize=30, color=CREAM, weight="bold", va="center")
    ax.text(60, 162, "spellF1, 27 unseen matches", fontsize=12, color=MUTED, va="center")
    ax.text(330, 200, "12 s", fontsize=30, color=CREAM, weight="bold", va="center")
    ax.text(330, 162, "per match, laptop CPU", fontsize=12, color=MUTED, va="center")
    ax.text(60, 70, "More than a Hack 2027  ·  Phase 1  ·  github.com/agamemnon-labs/football-analytics",
            fontsize=11.5, color=GOLD, va="center")

    out = os.path.join(ASSETS, "social.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    site = os.path.join(ROOT, "site", "assets")
    if os.path.isdir(site):
        shutil.copy(out, os.path.join(site, "social.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
