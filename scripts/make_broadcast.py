"""The README's hero animation: the external example match seen through a TV-style camera.

Same match, window and labels as make_figures.py, but drawn from a camera high on the near side
that pans with the ball, with a possession graphic in the corner. Players are positions from
tracking, so they are drawn as pins, not people.

    python scripts/make_broadcast.py              # docs/assets/broadcast.gif
    python scripts/make_broadcast.py --mp4 out.mp4 --step 1   # full frame rate, for sharing
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import CREDIT, NAMES, OUT, THEMES, band, load_example, pick_window  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from agamemnon.schema import A, D, H  # noqa: E402

W_PX, H_PX = 960, 540
FOCAL = 1000.0
CAM_BACK, CAM_HEIGHT = 35.0, 22.0     # metres behind the near touchline, metres up
LOOK_Y = 20.0                         # aim point; this framing keeps both touchlines in shot
PAN_TAU_S = 1.2                       # camera smoothing; real operators lag the ball too
GRASS = ("#2f6b3a", "#347442")
SURROUND = "#24502d"
LINE = "#e9efe9"
TH = THEMES["dark"]


class Camera:
    def __init__(self, length):
        self.length = length

    def set(self, look_x):
        eye = np.array([look_x, -CAM_BACK, CAM_HEIGHT])
        target = np.array([look_x, LOOK_Y, 0.0])
        f = target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.eye, self.f, self.r, self.u = eye, f, r, u

    def project(self, pts):
        """(..., 3) world points in metres to (..., 2) screen pixels and depth."""
        d = np.asarray(pts, float) - self.eye
        z = d @ self.f
        return np.stack([FOCAL * (d @ self.r) / z, FOCAL * (d @ self.u) / z], -1), z


def pan_track(seen, a, b, fps, length):
    """Where the camera looks along x: the ball when seen, else the players' centroid, smoothed
    forwards and backwards so the pan neither lags nor jumps."""
    target = np.empty(b - a)
    for i, f in enumerate(range(a, b)):
        bx = seen.ball[f, 0]
        if np.isnan(bx):
            xs = seen.players[f, :, 0]
            bx = np.nanmean(xs) if np.isfinite(xs).any() else length / 2
        target[i] = bx
    k = 1.0 - np.exp(-1.0 / (PAN_TAU_S * fps))
    out = target.copy()
    for rng in (range(1, len(out)), range(len(out) - 2, -1, -1)):
        for i in rng:
            j = i - 1 if rng.step == 1 else i + 1
            out[i] = out[j] + k * (out[i] - out[j])
    return np.clip(out, 28.0, length - 28.0)


def pitch_geometry(length, width):
    """Grass stripes as polygons and markings as 3-D polylines, all in world metres."""
    stripes = []
    n = 18
    for i in range(n):
        x0, x1 = length * i / n, length * (i + 1) / n
        stripes.append(([(x0, 0, 0), (x1, 0, 0), (x1, width, 0), (x0, width, 0)], GRASS[i % 2]))
    lines = []

    def seg(*pts):
        lines.append([(x, y, 0.0) for x, y in pts])

    seg((0, 0), (length, 0), (length, width), (0, width), (0, 0))
    seg((length / 2, 0), (length / 2, width))
    t = np.linspace(0, 2 * np.pi, 72)
    lines.append([(length / 2 + 9.15 * np.cos(a), width / 2 + 9.15 * np.sin(a), 0) for a in t])
    for x0, s in ((0.0, 1), (length, -1)):
        seg((x0, width / 2 - 20.16), (x0 + s * 16.5, width / 2 - 20.16),
            (x0 + s * 16.5, width / 2 + 20.16), (x0, width / 2 + 20.16))
        seg((x0, width / 2 - 9.16), (x0 + s * 5.5, width / 2 - 9.16),
            (x0 + s * 5.5, width / 2 + 9.16), (x0, width / 2 + 9.16))
        arc = np.linspace(-0.927, 0.927, 24)
        cx = x0 + s * 11.0
        lines.append([(cx + s * 9.15 * np.cos(a), width / 2 + 9.15 * np.sin(a), 0) for a in arc
                      if s * (cx + s * 9.15 * np.cos(a) - x0) > 16.5])
        g0, g1 = width / 2 - 3.66, width / 2 + 3.66
        lines.append([(x0, g0, 0), (x0, g0, 2.44), (x0, g1, 2.44), (x0, g1, 0)])
        lines.append([(x0, g0, 2.44), (x0 - s * 1.5, g0, 2.0), (x0 - s * 1.5, g1, 2.0),
                      (x0, g1, 2.44)])
    surround = [(-12, -8, 0), (length + 12, -8, 0), (length + 12, width + 14, 0), (-12, width + 14, 0)]
    # Pitch-side boards behind the far touchline: 1 m tall panels in alternating colours.
    boards = []
    edge = 6.0
    for i, x0 in enumerate(np.arange(-6.0, length + 6.0, edge)):
        quad = [(x0, width + 4, 0), (x0 + edge - 0.2, width + 4, 0),
                (x0 + edge - 0.2, width + 4, 1.0), (x0, width + 4, 1.0)]
        boards.append((quad, ("#14181c", TH["home"], "#14181c", TH["away"])[i % 4]))
    return stripes, [l for l in lines if len(l) > 1], surround, boards


def rounded(ax, x, y, w, h, colour, alpha=0.88):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=7", lw=0,
                         facecolor=colour, alpha=alpha, transform=ax.transData, zorder=20)
    ax.add_patch(box)
    return box


def render(step, mp4=None, seconds=30, still=None):
    match, seen, gt, pred = load_example()
    fps = match.fps
    a, b = pick_window(gt, match.periods, fps, seconds)
    length, width = match.pitch
    cam = Camera(length)
    look = pan_track(seen, a, b, fps, length)
    stripes, lines, surround, boards = pitch_geometry(length, width)
    team = seen.team
    colour = {H: TH["home"], A: TH["away"], D: TH["dead"]}

    fig = plt.figure(figsize=(W_PX / 100, H_PX / 100), dpi=100)
    fig.patch.set_facecolor(SURROUND)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-W_PX / 2, W_PX / 2)
    ax.set_ylim(-H_PX / 2, H_PX / 2)
    ax.axis("off")
    ax.set_facecolor(SURROUND)

    surround_poly = PolyCollection([], facecolors=SURROUND, edgecolors="none", zorder=0)
    grass = PolyCollection([], facecolors=[c for _, c in stripes], edgecolors="none", zorder=1)
    marks = LineCollection([], colors=LINE, linewidths=1.6, alpha=0.9, zorder=2)
    board_polys = PolyCollection([], facecolors=[c for _, c in boards], edgecolors="none",
                                 alpha=0.95, zorder=1.5)
    shadows = ax.scatter([], [], s=[], c="#000000", alpha=0.28, marker="o", lw=0, zorder=3)
    bodies = LineCollection([], linewidths=[], capstyle="round", zorder=4)
    heads = ax.scatter([], [], s=[], facecolors="#ffffff", edgecolors="#10151a", lw=0.8, zorder=5)
    ball_shadow = ax.scatter([], [], s=[], c="#000000", alpha=0.35, lw=0, zorder=6)
    ball = ax.scatter([], [], s=[], c="#ffffff", edgecolors="#111111", lw=0.9, zorder=7)
    ghost = ax.scatter([], [], s=[], facecolors="none", edgecolors="#ffffff", lw=1.2, zorder=7)
    ghost_text = ax.text(0, 0, "", fontsize=8, color="#ffffff", ha="center", va="bottom", zorder=8)
    for artist in (surround_poly, grass, board_polys, marks, bodies):
        ax.add_collection(artist)

    # Corner graphic, styled after a broadcast possession bug.
    rounded(ax, -466, 168, 236, 88, "#0f1215")
    ax.text(-454, 238, "POSSESSION CALL", fontsize=7.5, color="#9aa3a8", weight="bold", zorder=21)
    chip = rounded(ax, -456, 192, 120, 34, TH["home"], alpha=1.0)
    chip_text = ax.text(-396, 209, "", fontsize=13, color="#ffffff", weight="bold", ha="center",
                        va="center", zorder=22)
    truth_text = ax.text(-328, 209, "", fontsize=8.5, color="#d7dcdf", va="center", zorder=22)
    ax.text(-454, 176, "the model's call, live", fontsize=7, color="#7d878c", zorder=21)

    rounded(ax, 236, 214, 230, 42, "#0f1215")
    ax.text(246, 244, "SHARE OF THIS CLIP", fontsize=7, color="#9aa3a8", weight="bold", zorder=21)
    share_home = Rectangle((246, 222), 0, 12, color=TH["home"], zorder=22, lw=0)
    share_away = Rectangle((246, 222), 0, 12, color=TH["away"], zorder=22, lw=0)
    ax.add_patch(share_home)
    ax.add_patch(share_away)
    share_text = ax.text(456, 244, "", fontsize=7.5, color="#d7dcdf", ha="right", zorder=22)

    rounded(ax, -466, -262, 932, 66, "#0f1215")
    tl = fig.add_axes([0.135, 0.045, 0.83, 0.075])
    tl.set_facecolor("#0f1215")
    tl.axis("off")
    band(tl, gt[a:b], 1.08, 0.82, 0, fps, TH)
    band(tl, pred[a:b], 0.08, 0.82, 0, fps, TH)
    tl.set_xlim(0, (b - a) / fps)
    tl.set_ylim(0, 2)
    cursor = tl.axvline(0, color="#ffffff", lw=1.5)
    fig.text(0.125, 0.105, "truth", fontsize=8, color="#d7dcdf", ha="right", va="center")
    fig.text(0.125, 0.065, "model", fontsize=8, color="#d7dcdf", ha="right", va="center")
    fig.text(0.985, 0.19, CREDIT + "  |  35% of ball positions removed", fontsize=6.5,
             color="#c9d3cb", ha="right", va="center")

    frames = list(range(a, b, step))
    last_seen = {"xy": None, "f": None}

    def frame(i):
        f = frames[i]
        cam.set(look[f - a])
        surround_poly.set_verts([cam.project(surround)[0]])
        grass.set_verts([cam.project(p)[0] for p, _ in stripes])
        marks.set_segments([cam.project(l)[0] for l in lines])
        board_polys.set_verts([cam.project(q)[0] for q, _ in boards])

        p = seen.players[f]
        ok = np.isfinite(p[:, 0])
        feet = np.column_stack([p[ok], np.zeros(ok.sum())])
        sides = team[ok]
        foot_px, depth = cam.project(feet)
        scale = FOCAL / depth
        # Drawn upright on screen, like a billboard; true perspective would tilt the pins outward
        # towards the edges, which reads as players leaning.
        up = np.column_stack([np.zeros_like(scale), scale])
        top_px = foot_px + up * 1.45
        head_px = foot_px + up * 1.75
        order = np.argsort(-depth)
        bodies.set_segments([[foot_px[k], top_px[k]] for k in order])
        bodies.set_linewidths(scale[order] * 0.42)
        bodies.set_colors([colour[int(s)] for s in sides[order]])
        heads.set_offsets(head_px[order])
        heads.set_sizes((scale[order] * 0.5) ** 2)
        heads.set_facecolors([colour[int(s)] for s in sides[order]])
        shadows.set_offsets(foot_px + np.column_stack([scale * 0.05, np.zeros_like(scale)]))
        shadows.set_sizes((scale * 0.42) ** 2)

        bxy = seen.ball[f]
        if np.isfinite(bxy[0]):
            last_seen.update(xy=bxy.copy(), f=f)
            ground, z = cam.project([bxy[0], bxy[1], 0.0])
            lift, _ = cam.project([bxy[0], bxy[1], 0.3])
            ball.set_offsets([lift])
            ball.set_sizes([(FOCAL / z * 0.30) ** 2])  # drawn about 1.3x life size so it stays visible
            ball_shadow.set_offsets([ground])
            ball_shadow.set_sizes([(FOCAL / z * 0.30) ** 2])
            ghost.set_offsets(np.empty((0, 2)))
            ghost_text.set_text("")
        else:
            ball.set_offsets(np.empty((0, 2)))
            ball_shadow.set_offsets(np.empty((0, 2)))
            if last_seen["xy"] is not None:
                g, z = cam.project([*last_seen["xy"], 0.15])
                ghost.set_offsets([g])
                ghost.set_sizes([(FOCAL / z * 0.5) ** 2])
                ghost_text.set_position((g[0], g[1] + FOCAL / z * 0.45))
                ghost_text.set_text("ball not tracked")

        call, truth = int(pred[f]), int(gt[f])
        chip.set_facecolor(colour.get(call, TH["dead"]))
        chip_text.set_text({H: "HOME", A: "AWAY", D: "DEAD"}.get(call, "-"))
        truth_text.set_text("truth: {}  {}".format(NAMES.get(truth, "-").lower(),
                                                   "✓" if call == truth else "✗"))
        done = pred[a:f + 1]
        nh, na = int((done == H).sum()), int((done == A).sum())
        total = max(nh + na, 1)
        share_home.set_width(210 * nh / total)
        share_away.set_x(246 + 210 * nh / total)
        share_away.set_width(210 * na / total)
        share_text.set_text("{:.0f}% / {:.0f}%".format(100 * nh / total, 100 * na / total))
        cursor.set_xdata([(f - a) / fps] * 2)
        return ()

    if still:
        frame(len(frames) // 2)
        fig.savefig(still, facecolor=SURROUND)
        plt.close(fig)
        return still
    anim = animation.FuncAnimation(fig, frame, frames=len(frames), blit=False)
    rate = fps / step
    if mp4:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        anim.save(mp4, dpi=200, writer=animation.FFMpegWriter(fps=rate, bitrate=9000,
                                                     extra_args=["-pix_fmt", "yuv420p"]))
        out = mp4
    else:
        out = os.path.join(OUT, "broadcast.gif")
        # 80 dpi keeps the GIF near 4 MB; the panning camera changes every pixel, so GIF
        # compression has little to work with.
        anim.save(out, dpi=80, writer=animation.PillowWriter(fps=round(rate)))
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=int, default=3, help="draw every n-th frame")
    parser.add_argument("--mp4", help="write an MP4 here instead of the GIF (needs imageio-ffmpeg)")
    parser.add_argument("--still", help="write one PNG from the middle of the clip and stop")
    args = parser.parse_args()
    print("wrote", render(args.step, args.mp4, still=args.still))


if __name__ == "__main__":
    main()
