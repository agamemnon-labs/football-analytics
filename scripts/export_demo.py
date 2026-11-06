"""Data for the browser demo in site/: ten minutes of one external match, at every degradation level.

Players come from the clean tracking (sampled at 5 fps; the page interpolates), the ball and our
labels from each degraded version, the truth from the dataset. Nothing from the competition match.

    python scripts/export_demo.py          # writes site/demo.json
"""

import json
import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agamemnon import priors  # noqa: E402
from agamemnon.external import bench  # noqa: E402
from agamemnon.external.degrade import LEVELS, degrade  # noqa: E402

GAME = ("idsse", "J03WMX")
MINUTES = 10
PLAYER_STEP, BALL_STEP = 5, 2          # 25 fps source: players at 5 fps, ball at 12.5 fps
OUT = os.path.join(ROOT, "site", "demo.json")


def runs(codes):
    """Run-length encoding as [code, length, code, length, ...]."""
    out = []
    edges = np.flatnonzero(np.diff(codes)) + 1
    for a, b in zip(np.concatenate(([0], edges)), np.concatenate((edges, [len(codes)]))):
        out += [int(codes[a]), int(b - a)]
    return out


def dm(values):
    """Positions in decimetres; -1 marks a missing value."""
    v = np.round(np.asarray(values, float) * 10)
    return np.where(np.isfinite(v), v, -1).astype(int).ravel().tolist()


def main():
    os.environ.setdefault("AGAMEMNON_OOF", "cv_w96")
    match = bench.load_game(*GAME)
    tagged = replace(match, tags=bench.synthetic_tags(match, match.name))
    fps = match.fps
    a = match.periods[0][0]
    b = a + int(MINUTES * 60 * fps)
    home = np.flatnonzero(match.team == 0)
    away = np.flatnonzero(match.team == 1)
    players = match.players[a:b:PLAYER_STEP][:, np.concatenate([home, away])]

    levels = {}
    for name in bench.LEVELS:
        cell = bench.Cell(GAME[0], GAME[1], name)
        pred = cell.decode(priors, True, cell.oof_logp(True))
        full = bench.score(cell.gt, pred)
        seen = degrade(tagged, name)
        levels[name] = {
            "missing": LEVELS[name],
            "ball": dm(seen.ball[a:b:BALL_STEP]),
            "pred": runs(pred[a:b]),
            "spell_f1": round(float(full["spell_f1"]), 3),
            "macro_f1": round(float(full["macro_f1"]), 3),
        }
        print(name, "spellF1 {spell_f1} macroF1 {macro_f1}".format(**levels[name]))

    gt = bench.ground_truth(match)
    data = {
        "match": "Bundesliga 2022/23, IDSSE match {}".format(GAME[1]),
        "credit": "IDSSE / DFL data, CC BY 4.0 (Bassek et al., Scientific Data 2025). "
                  "Ball removed and positions perturbed by us to imitate automatic tracking.",
        "fps": fps, "pitch": list(match.pitch), "minutes": MINUTES,
        "player_step": PLAYER_STEP, "ball_step": BALL_STEP,
        "n_home": int(len(home)), "n_away": int(len(away)),
        "players": dm(players), "truth": runs(gt[a:b]), "levels": levels,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"))
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")


if __name__ == "__main__":
    main()
