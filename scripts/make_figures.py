"""Figures for the README and the technical document, drawn from one external match.

Everything here comes from the benchmark (IDSSE match J03WMX, tracking degraded at the "mid"
level, out-of-fold network outputs from cv_w96), never from the competition match.

    python scripts/make_figures.py            # writes docs/assets/*.png and demo.gif
"""

import argparse
import os
from dataclasses import replace
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.patches import Arc, Circle, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agamemnon import priors  # noqa: E402
from agamemnon.external import bench  # noqa: E402
from agamemnon.external.degrade import degrade  # noqa: E402
from agamemnon.schema import A, D, H  # noqa: E402

OUT = os.path.join(ROOT, "docs", "assets")
GAME, LEVEL = ("idsse", "J03WMX"), "mid"
CREDIT = "IDSSE / DFL data, CC BY 4.0"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e2dd",
                  home="#2a78d6", away="#eb6834", dead="#a3a19b"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34332f",
                 home="#3987e5", away="#d95926", dead="#5f5e5a"),
}
NAMES = {H: "Home", A: "Away", D: "Dead ball"}


def load_example():
    os.environ.setdefault("AGAMEMNON_OOF", "cv_w96")
    cell = bench.Cell(GAME[0], GAME[1], LEVEL)
    pred = cell.decode(priors, True, cell.oof_logp(True))
    match = bench.load_game(*GAME)
    # Same steps as bench.prepare, so the positions drawn are the ones the model was given.
    tagged = replace(match, tags=bench.synthetic_tags(match, match.name))
    seen = degrade(tagged, LEVEL)
    return match, seen, cell.gt, pred


def runs(codes):
    """(start, stop, code) for each run of equal values."""
    edges = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(codes)]))
    return [(int(a), int(b), int(codes[a])) for a, b in zip(starts, stops)]


def band(ax, codes, y, h, t0, fps, th):
    for a, b, c in runs(codes):
        if c in NAMES:
            colour = th["home"] if c == H else th["away"] if c == A else th["dead"]
            ax.add_patch(Rectangle(((t0 + a) / fps, y), (b - a) / fps, h, lw=0, color=colour))


def style_axes(ax, th):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["grid"])
    ax.tick_params(colors=th["ink2"], labelsize=9, length=0)


def legend_row(fig, th, y):
    x = 0.08
    for code in (H, A, D):
        colour = th["home"] if code == H else th["away"] if code == A else th["dead"]
        fig.patches.append(Rectangle((x, y), 0.014, 0.022, transform=fig.transFigure,
                                     color=colour, lw=0))
        fig.text(x + 0.02, y + 0.011, NAMES[code], va="center", fontsize=9, color=th["ink2"])
        x += 0.14


def timeline(match, seen, gt, pred, theme):
    """Ten minutes from kick-off: truth, our labels, and when the ball was tracked."""
    th = THEMES[theme]
    fps = match.fps
    start = match.periods[0][0]
    stop = start + int(600 * fps)
    sl = slice(start, stop)
    fig, ax = plt.subplots(figsize=(9, 2.5), dpi=150)
    fig.patch.set_facecolor(th["surface"])
    style_axes(ax, th)
    band(ax, gt[sl], 2.15, 0.7, 0, fps, th)
    band(ax, pred[sl], 1.2, 0.7, 0, fps, th)
    tracked = ~np.isnan(seen.ball[sl, 0])
    for a, b, c in runs(tracked.astype(np.int8)):
        if c:
            ax.add_patch(Rectangle((a / fps, 0.55), (b - a) / fps, 0.3, lw=0, color=th["ink2"]))
    ax.set_xlim(0, 600)
    ax.set_ylim(0.4, 3.0)
    ax.set_yticks([2.5, 1.55, 0.7])
    ax.set_yticklabels(["ground truth", "our labels", "ball tracked"], color=th["ink"])
    ax.set_xticks(range(0, 601, 60))
    ax.set_xticklabels(["{}:00".format(m) for m in range(11)])
    ax.set_xlabel("minutes after kick-off", color=th["ink2"], fontsize=9)
    legend_row(fig, th, 0.9)
    fig.text(0.99, 0.91, CREDIT, ha="right", va="center", fontsize=7, color=th["ink2"])
    fig.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.2)
    path = os.path.join(OUT, "timeline-{}.png".format(theme))
    fig.savefig(path, facecolor=th["surface"])
    plt.close(fig)
    return path


# Numbers from docs/BENCHMARK.md (nested estimates on the left, locked decoder on the right).
BENCH = [("logistic emissions", 0.501), ("CNN, 64 channels", 0.611),
         ("CNN, 96 channels (submitted)", 0.681)]
ABLATION = [("no Smart Tagging", 0.086), ("switch penalty halved", 0.007),
            ("switch penalty doubled", 0.003), ("no hand-over snapping", 0.000)]


def benchmark(theme):
    """Model versions on the external benchmark, and what each part is worth."""
    th = THEMES[theme]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 2.6), dpi=150,
                                      gridspec_kw={"width_ratios": [1, 1], "wspace": 0.95})
    fig.patch.set_facecolor(th["surface"])
    for ax, rows, title, fmt, xmax in (
            (left, BENCH, "spellF1, nested cross-validation", "{:.3f}", 0.8),
            (right, ABLATION, "spellF1 lost by one change", "{:.3f}", 0.1)):
        style_axes(ax, th)
        y = np.arange(len(rows))[::-1]
        values = [v for _, v in rows]
        colours = [th["home"] if i == (len(rows) - 1 if ax is left else 0) else th["dead"]
                   for i in range(len(rows))]
        ax.barh(y, values, height=0.55, color=colours, lw=0)
        for yi, v in zip(y, values):
            ax.text(v + xmax * 0.015, yi, fmt.format(v), va="center", fontsize=9, color=th["ink"])
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _ in rows], color=th["ink"], fontsize=9)
        ax.set_xlim(0, xmax)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(title, loc="left", fontsize=10, color=th["ink"], pad=8)
    fig.subplots_adjust(left=0.25, right=0.96, top=0.82, bottom=0.08)
    path = os.path.join(OUT, "benchmark-{}.png".format(theme))
    fig.savefig(path, facecolor=th["surface"])
    plt.close(fig)
    return path


def draw_pitch(ax, length, width, line):
    kw = dict(fill=False, lw=1.0, color=line)
    ax.add_patch(Rectangle((0, 0), length, width, **kw))
    ax.plot([length / 2] * 2, [0, width], color=line, lw=1.0)
    ax.add_patch(Circle((length / 2, width / 2), 9.15, **kw))
    for x0, sgn in ((0, 1), (length, -1)):
        ax.add_patch(Rectangle((x0 if sgn > 0 else x0 - 16.5, width / 2 - 20.16), 16.5, 40.32, **kw))
        ax.add_patch(Rectangle((x0 if sgn > 0 else x0 - 5.5, width / 2 - 9.16), 5.5, 18.32, **kw))
        spot = x0 + sgn * 11
        ax.add_patch(Arc((spot, width / 2), 18.3, 18.3, theta1=-53 if sgn > 0 else 127,
                         theta2=53 if sgn > 0 else 233, lw=1.0, color=line))
    ax.set_xlim(-2, length + 2)
    ax.set_ylim(-2, width + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def pick_window(gt, periods, fps, seconds):
    """The first 30 s window of the first half that holds a stoppage and two hand-overs."""
    n = int(seconds * fps)
    start, stop = periods[0]
    for a in range(start + int(60 * fps), stop - n, int(5 * fps)):
        r = runs(gt[a:a + n])
        teams = [c for _, _, c in r if c in (H, A)]
        switches = sum(1 for p, q in zip(teams, teams[1:]) if p != q)
        if any(c == D for _, _, c in r) and switches >= 2:
            return a, a + n
    return start, start + n


def demo_gif(match, seen, gt, pred, seconds=30, step=3):
    """Animated example: tracking as the model sees it, with truth and our labels below."""
    th = dict(THEMES["dark"], pitch="#22382a", line="#9fb3a5")
    fps = match.fps
    a, b = pick_window(gt, match.periods, fps, seconds)
    length, width = match.pitch
    frames = list(range(a, b, step))
    team = seen.team  # degradation can add slots (fragmented tracks)
    fig = plt.figure(figsize=(7.2, 5.6), dpi=100)
    fig.patch.set_facecolor(th["surface"])
    ax = fig.add_axes([0.03, 0.27, 0.94, 0.66])
    ax.set_facecolor(th["pitch"])
    ax.add_patch(Rectangle((-2, -2), length + 4, width + 4, color=th["pitch"], lw=0, zorder=0))
    draw_pitch(ax, length, width, th["line"])
    home_dots = ax.scatter([], [], s=70, color=th["home"], edgecolors=th["surface"], lw=1.2, zorder=3)
    away_dots = ax.scatter([], [], s=70, color=th["away"], edgecolors=th["surface"], lw=1.2, zorder=3)
    trail, = ax.plot([], [], color="#ffffff", lw=1.2, alpha=0.5, zorder=4)
    ball = ax.scatter([], [], s=30, color="#ffffff", edgecolors="#000000", lw=0.8, zorder=5)
    missing = ax.text(length / 2, width + 0.5, "", ha="center", va="bottom", fontsize=9,
                      color=th["ink2"], zorder=6)

    tl = fig.add_axes([0.16, 0.08, 0.8, 0.13])
    style_axes(tl, th)
    tl.spines["bottom"].set_visible(False)
    band(tl, gt[a:b], 1.1, 0.75, 0, fps, th)
    band(tl, pred[a:b], 0.1, 0.75, 0, fps, th)
    tl.set_xlim(0, (b - a) / fps)
    tl.set_ylim(0, 2)
    tl.set_xticks([])
    tl.set_yticks([1.47, 0.47])
    tl.set_yticklabels(["ground truth", "our labels"], color=th["ink"], fontsize=9)
    cursor = tl.axvline(0, color=th["ink"], lw=1.5)

    title = fig.text(0.03, 0.965, "", fontsize=10.5, color=th["ink"], va="center")
    clock = "first half, {}:{:02d} to {}:{:02d}".format(
        *divmod(int((a - match.periods[0][0]) / fps), 60), *divmod(int((b - match.periods[0][0]) / fps), 60))
    fig.text(0.97, 0.972, clock, fontsize=8, color=th["ink2"], va="center", ha="right")
    fig.text(0.97, 0.945, CREDIT, fontsize=7, color=th["ink2"], va="center", ha="right")
    legend_row(fig, th, 0.015)
    fig.text(0.97, 0.026, "tracking degraded: about 35% of ball positions removed", fontsize=7.5,
             color=th["ink2"], va="center", ha="right")

    def frame(i):
        f = frames[i]
        p = seen.players[f]
        for dots, side in ((home_dots, H), (away_dots, A)):
            xy = p[team == side]
            dots.set_offsets(xy[~np.isnan(xy[:, 0])])
        recent = seen.ball[max(a, f - int(fps)):f + 1]
        recent = recent[~np.isnan(recent[:, 0])]
        trail.set_data(recent[:, 0], recent[:, 1])
        if np.isnan(seen.ball[f, 0]):
            ball.set_offsets(np.empty((0, 2)))
            missing.set_text("ball not tracked in this frame")
        else:
            ball.set_offsets(seen.ball[f][None])
            missing.set_text("")
        cursor.set_xdata([(f - a) / fps] * 2)
        title.set_text("truth: {:<9}   ours: {}".format(NAMES.get(int(gt[f]), "-"),
                                                       NAMES.get(int(pred[f]), "-")))
        return home_dots, away_dots, trail, ball, missing, cursor, title

    anim = animation.FuncAnimation(fig, frame, frames=len(frames), blit=False)
    path = os.path.join(OUT, "demo.gif")
    anim.save(path, writer=animation.PillowWriter(fps=round(fps / step)),
              savefig_kwargs={"facecolor": th["surface"]})
    # A still for the PDF and DOCX, which cannot play the animation: the first frame after the
    # midpoint of the window in which the ball is visible.
    k = next((i for i in range(len(frames) // 2, len(frames))
              if not np.isnan(seen.ball[frames[i], 0])), len(frames) // 2)
    frame(k)
    fig.savefig(os.path.join(OUT, "demo-still.png"), facecolor=th["surface"], dpi=150)
    plt.close(fig)
    return path, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    match, seen, gt, pred = load_example()
    for theme in THEMES:
        print("wrote", timeline(match, seen, gt, pred, theme))
        print("wrote", benchmark(theme))
    if not args.no_gif:
        path, (a, b) = demo_gif(match, seen, gt, pred)
        print("wrote", path, "frames {}..{}".format(a, b))


if __name__ == "__main__":
    main()
