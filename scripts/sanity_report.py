"""Aggregate sanity statistics of the submission, next to the same statistics on the benchmark.

Prints counts and quantiles only. It exists to catch bugs (a D run per frame, possessions of half a
second), not to tune anything: a difference between the two columns is a reason to look for a bug
on the external games, never to move a parameter towards the competition match.

    python scripts/sanity_report.py
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from agamemnon import pipeline, priors  # noqa: E402
from agamemnon.external import bench  # noqa: E402
from agamemnon.schema import A, D, GOALS, H, SET_PIECES, X  # noqa: E402
import tune  # noqa: E402

LEVEL = "high"


def runs(codes, periods):
    out = []
    for p0, p1 in periods:
        seg = codes[p0:p1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        out += [(p0 + int(a), p0 + int(b), int(seg[a])) for a, b in zip(bounds[:-1], bounds[1:])]
    return out


def stats(codes, periods, fps):
    r = runs(codes, periods)
    poss = np.array([b - a for a, b, k in r if k in (H, A)]) / fps
    dead = np.array([b - a for a, b, k in r if k == D]) / fps
    graded = np.concatenate([codes[a:b] for a, b in periods])
    minutes = len(graded) / fps / 60
    return {
        "share H/A/D/X": "/".join("{:.0%}".format(np.mean(graded == k)) for k in (H, A, D, X)),
        "possessions per 90 min": len(poss) / minutes * 90,
        "possession p10/median/p90 s": "{:.1f} / {:.1f} / {:.1f}".format(
            *np.percentile(poss, [10, 50, 90])) if len(poss) else "-",
        "possessions < 3 s": "{:.0%}".format(np.mean(poss < 3)) if len(poss) else "-",
        "D runs per 90 min": len(dead) / minutes * 90,
        "D run median s": "{:.1f}".format(np.median(dead)) if len(dead) else "-",
    }


def tag_overlap(codes, match):
    """How many D runs contain or touch a SET PIECES / GOALS tag, and how many tags have one."""
    dead_runs = [(a, b) for a, b, k in runs(codes, match.periods) if k == D]
    tags = [t for t in match.tags if t.code in (SET_PIECES, GOALS)]
    slack = int(5 * match.fps)
    tagged = sum(any(a - slack <= t.stop and t.start <= b + slack for t in tags)
                 for a, b in dead_runs)
    by_kind = {}
    for t in tags:
        kind = t.kind or t.code
        hit = any(a - slack <= t.stop and t.start <= b + slack for a, b in dead_runs)
        n, h = by_kind.get(kind, (0, 0))
        by_kind[kind] = (n + 1, h + hit)
    return len(dead_runs), tagged, len(tags), by_kind


def bench_stats():
    prm = SimpleNamespace(**tune.base_params())
    rows = {}
    for d, m in bench.all_games():
        cell = bench.Cell(d, m, LEVEL)
        pred = cell.decode(prm)
        for key, value in stats(pred, cell.periods, cell.fps).items():
            rows.setdefault(key, []).append(value)
        for key, value in stats(cell.gt, cell.periods, cell.fps).items():
            rows.setdefault("truth " + key, []).append(value)
    return {k: (np.mean(v) if isinstance(v[0], float) else v[0] + " (first game)")
            for k, v in rows.items()}


def main():
    codes, match = pipeline.run()
    ours = stats(codes, match.periods, match.fps)
    ext = bench_stats()
    print("{:<30} {:>22} {:>26} {:>22}".format("statistic", "competition", "benchmark prediction",
                                               "benchmark truth"))
    for key, value in ours.items():
        fmt = lambda v: "{:.1f}".format(v) if isinstance(v, float) else str(v)
        print("{:<30} {:>22} {:>26} {:>22}".format(key, fmt(value), fmt(ext[key]),
                                                   fmt(ext["truth " + key])))
    n_dead, tagged, n_tags, by_kind = tag_overlap(codes, match)
    print("\nD runs {}  of which near a SET PIECES/GOALS tag {}  ({} such tags)".format(
        n_dead, tagged, n_tags))
    for kind, (n, hit) in sorted(by_kind.items()):
        print("  {:<24} tags {:>3}  with a D run nearby {:>3}".format(kind, n, hit))
    flags = []
    if n_dead > 3 * max(n_tags, 1):
        flags.append("far more D runs than restart tags")
    if ours["possessions < 3 s"] != "-" and float(ours["possessions < 3 s"].rstrip("%")) > 40:
        flags.append("many very short possessions")
    print("\nflags: {}".format(", ".join(flags) if flags else "none"))


if __name__ == "__main__":
    main()
