"""How much the network depends on the shape of the imitation tags it was trained with.

The benchmark has no real Smart Tagging, so its tags are generated (bench.synthetic_tags). This
script swaps the tag inputs of every game for tags drawn by a second, independent generator in two
styles and rescores the out-of-fold networks:

  fixed windows       set pieces as 35 s windows, goal intervals that run to after the kick-off,
                      phases as 10 to 30 s windows (the structure of the real file)
  possession-aligned  short windows cut at the labelled possession edges (the structure the first
                      version of the benchmark used)

A network that only works with the style it was trained on would fall between the rows.

    python scripts/tag_styles.py --oof cv_v2
"""

import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from agamemnon import deadball, decoder, priors, tcn  # noqa: E402
from agamemnon.external import CACHE_DIR, bench  # noqa: E402
from agamemnon.schema import A, D, GOALS, H, SET_PIECES, Tag  # noqa: E402
import train_tcn  # noqa: E402

pm = bench.pm
STYLES = {"fixed windows": True, "possession-aligned": False}
ROWS = ["tags as trained", "no tags"] + list(STYLES)


def other(team):
    return A if team == H else H


def restart_kind(match, start, first_of_period, rng):
    """Guess the restart that opens a possession from where the ball is."""
    length, width = match.pitch
    x, y = match.ball[start]
    if first_of_period:
        return "Kick Off (Start)"
    if np.isfinite(x) and np.hypot(x - length / 2, y - width / 2) < 3.0:
        return "Kick Off (After Goal)"
    if np.isfinite(y) and min(y, width - y) < 2.0:
        return "Throw-in"
    return "Free Kick"


def draw_tags(match, fixed):
    """Imitation tags for one game, in the fixed-window or the possession-aligned style."""
    rng = np.random.default_rng(zlib.crc32(("styles|" + match.name).encode()))
    fps = match.fps
    tags = []
    for p_start, p_stop in match.periods:
        seg = match.labels[p_start:p_stop]
        spells = [(a + p_start, b + p_start, k) for a, b, k in pm.segments(seg, [(0, len(seg))])]
        for i, (a, b, team) in enumerate(spells):
            if team == D:
                continue
            if i == 0 or spells[i - 1][2] == D:
                kind = restart_kind(match, a, i <= 1, rng)
                if kind == "Kick Off (After Goal)" and i >= 2 and rng.random() < 0.9:
                    goal = spells[i - 2][1]
                    if fixed:
                        tags.append(Tag(GOALS, goal - int(rng.uniform(10, 60) * fps),
                                        a + int(rng.uniform(30, 120) * fps), other(team), None))
                    else:
                        tags.append(Tag(GOALS, goal - int(rng.uniform(1, 3) * fps),
                                        goal + int(fps), other(team), None))
                if rng.random() < (0.9 if kind.startswith("Kick Off") else 0.45):
                    tagged = team
                    if kind in ("Throw-in", "Kick Off (After Goal)") or rng.random() < 0.08:
                        tagged = None
                    elif rng.random() < 0.07:
                        tagged = other(team)
                    if fixed:
                        lo = a - int(rng.uniform(3.0, 5.0) * fps)
                        hi = lo + int((5 if kind.startswith("Kick Off") else 35) * fps)
                    else:
                        lo = a - int(rng.uniform(0.5, 4.0) * fps)
                        hi = a + int(rng.uniform(0.5, 4.0) * fps)
                    tags.append(Tag(SET_PIECES, lo, hi, tagged, kind))
            if b - a >= 3 * fps and rng.random() < 0.55:
                lo = a + int(rng.uniform(-2, 3) * fps)
                hi = (lo + int(rng.uniform(10, 30) * fps) if fixed
                      else b + int(rng.uniform(-3, 2) * fps))
                if hi > lo:
                    if rng.random() < 0.6:
                        code, tagged = bench.ATTACK[rng.integers(len(bench.ATTACK))], team
                    else:
                        code, tagged = bench.DEFEND[rng.integers(len(bench.DEFEND))], other(team)
                    if rng.random() < 0.1:
                        tagged = other(tagged)
                    tags.append(Tag(code, lo, hi, tagged, None))
        for _ in range(rng.poisson((p_stop - p_start) / (fps * 180))):
            lo = int(rng.integers(p_start, p_stop))
            tags.append(Tag(bench.ATTACK[rng.integers(len(bench.ATTACK))], lo,
                            lo + int(rng.uniform(10, 30) * fps), H if rng.random() < 0.5 else A, None))
    return sorted(tags, key=lambda t: t.start)


def _worker(args):
    dataset, mid, fold, oof, levels = args
    os.environ["AGAMEMNON_OOF"] = oof
    match = bench.load_game(dataset, mid)
    net = tcn.load(os.path.join(CACHE_DIR, oof, "fold{}.json".format(fold)))
    drawn = {name: deadball.tag_features(replace(match, tags=draw_tags(match, fixed)), priors)
             for name, fixed in STYLES.items()}
    i_tag = decoder.TEAM_COLUMNS.index("tag")
    i_setpiece = deadball.DEAD_COLUMNS.index("setpiece")
    i_goal = deadball.DEAD_COLUMNS.index("post_goal")
    out = []
    for level in levels:
        cell = bench.Cell(dataset, mid, level)
        for name, tags in (("tags as trained", True), ("no tags", False)):
            pred = cell.decode(priors, tags, cell.oof_logp(tags))
            out.append((dataset, name, bench.score(cell.gt, pred)["spell_f1"]))
        for name, (team, setpiece, goal) in drawn.items():
            Xt, Xd = (x.copy() for x in cell.designs(True))
            Xt[:, i_tag], Xd[:, i_setpiece], Xd[:, i_goal] = team, setpiece, goal
            logp = tcn.log_probs(net, tcn.inputs(Xt, Xd), cell.periods)
            pred = decoder.decode_design(Xt, Xd, cell.touch_home, cell.touch_away, cell.periods,
                                         cell.kickoffs, cell.fps, priors, logp)
            out.append((dataset, name, bench.score(cell.gt, pred)["spell_f1"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oof", default="cv_v2")
    parser.add_argument("--levels", nargs="+", default=["mid", "high"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    games = bench.all_games()
    fold = {g: f for f, held in enumerate(train_tcn.folds_of(games, 5)) for g in held}
    jobs = [(d, m, fold[(d, m)], args.oof, args.levels) for d, m in games]
    scores = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for part in pool.map(_worker, jobs):
            for dataset, name, value in part:
                scores.setdefault((name, dataset), []).append(value)
    print("| tag inputs | IDSSE spellF1 | SkillCorner spellF1 | balanced spellF1 |")
    print("|---|---|---|---|")
    for name in ROWS:
        idsse, skill = np.mean(scores[(name, "idsse")]), np.mean(scores[(name, "skillcorner")])
        print("| {} | {:.3f} | {:.3f} | {:.3f} |".format(name, idsse, skill, (idsse + skill) / 2))


if __name__ == "__main__":
    main()
