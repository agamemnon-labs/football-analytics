"""Render docs/PRIORS.md from src/agamemnon/priors.py, so the table and the code cannot drift.

    python scripts/priors_table.py            # rewrite docs/PRIORS.md
    python scripts/priors_table.py --check    # exit 1 if docs/PRIORS.md is out of date

tests/test_priors.py runs the check.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agamemnon import priors  # noqa: E402

OUT = os.path.join(ROOT, "docs", "PRIORS.md")
TUNE = "scripts/tune.py"
TRAIN = "scripts/train_tcn.py"

# name: (source, rationale, produced by)
ROWS = {
    "BALL_MAX_SPEED": ("A", "fastest shots are around 35 m/s; faster jumps are false detections", "-"),
    "PLAYER_MAX_SPEED": ("A", "sprint speed tops out near 10-11 m/s; faster is a slot swap", "-"),
    "BALL_INTERP_MAX_S": ("A", "a ball unseen for longer may have been played on; do not invent a path", "-"),
    "BALL_SMOOTH_FRAMES": ("A", "removes frame-to-frame jitter before differentiating", "-"),
    "PLAYER_SMOOTH_FRAMES": ("A", "same for players", "-"),
    "VEL_HALF_WINDOW": ("A", "central difference over 0.16 s", "-"),
    "BALL_STILL_SPEED": ("A", "a placed or held ball moves slower than this", "-"),
    "TOUCH_HALF_WINDOW": ("A", "velocity before and after a touch over 0.12 s", "-"),
    "TOUCH_DV": ("A", "velocity change of a pass, trap or deflection", "-"),
    "TOUCH_RADIUS": ("A", "reach of a player playing the ball", "-"),
    "OOB_MARGIN": ("A, C", "ball wholly over the line (Law 9) plus tracking jitter", "-"),
    "FAR_OOB": ("A", "unambiguously off the pitch", "-"),
    "TEAM_DIST_SCALE": ("A", "distance advantage per unit of team evidence", "-"),
    "TEAM_DIST_CLIP": ("A", "beyond this the nearer side is clear", "-"),
    "CLOSE_RADIUS": ("A", "within playing distance", "-"),
    "NEAR_RADIUS": ("A", "players who can contest the ball within about a second", "-"),
    "TOUCH_MEMORY_S": ("C", "a pass takes one to two seconds to arrive", "-"),
    "SPREAD_SCALE": ("A", "team spread difference per unit", "-"),
    "SETPIECE_PRE_S": ("C", "tagged set pieces mark a window around the restart (starter docs)", "-"),
    "POST_GOAL_S": ("C", "celebration and kick-off take up to about a minute", "-"),
    "TEAM_COEF": ("E", "logistic team model, fitted on all external games", TUNE + " --final"),
    "DEAD_COEF": ("E", "logistic dead-ball model, fitted on all external games", TUNE + " --final"),
    "TEAM_TEMPER": ("E", "logistic emissions: per-frame log-odds are correlated", TUNE),
    "DEAD_TEMPER": ("E", "same, dead-ball model", TUNE),
    "DEAD_PRIOR": ("E", "intercept correction of the dead-ball model", TUNE),
    "SWITCH_COST": ("E", "Viterbi penalty for H <-> A", TUNE),
    "END_COST": ("E", "Viterbi penalty for alive -> D", TUNE),
    "START_COST": ("E", "Viterbi penalty for D -> alive", TUNE),
    "SNAP_WINDOW_S": ("E", "search window for moving a hand-over to the taker's touch", TUNE),
    "EMISSION": ("E", "which emission model the decoder uses", TUNE + " --emission network"),
    "NN_TEMPER": ("E", "network emissions: scale of per-frame log-probabilities", TUNE),
    "NN_DEAD_BIAS": ("E", "network emissions: offset of the dead-ball log-probability", TUNE),
}

HEADER = """# Priors

Every constant the pipeline uses. Sources: **A** physics or geometry, **C** a rule, definition or
general football knowledge, **E** fitted or tuned on the external games only (docs/EXTERNAL_DATA.md).
Nothing here was measured on, fitted to or tuned against the competition match or its ground truth.

This file is generated from `src/agamemnon/priors.py` by `scripts/priors_table.py`, and
`tests/test_priors.py` fails if the two disagree. The network weights are in
`src/agamemnon/models/tcn.json`, produced by `{train} --final`.

| name | value | source | rationale | produced by |
|---|---|---|---|---|
""".format(train=TRAIN)


def fmt(value):
    if isinstance(value, tuple):
        return "(" + ", ".join("{:g}".format(v) for v in value) + ")"
    if isinstance(value, float):
        return "{:g}".format(value)
    return str(value)


def render():
    names = [k for k in dir(priors) if k.isupper()]
    missing = sorted(set(names) - set(ROWS))
    if missing:
        raise SystemExit("no rationale for: {}".format(", ".join(missing)))
    lines = [HEADER]
    for name in ROWS:
        if name not in names:
            continue
        source, why, by = ROWS[name]
        lines.append("| `{}` | `{}` | {} | {} | {} |\n".format(
            name, fmt(getattr(priors, name)), source, why, by))
    lines.append(AMENDMENTS)
    return "".join(lines)


AMENDMENTS = """
## Amendments

None yet. After the parameters are locked, any change is listed here with its date and a reason
that is not a score.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        current = open(OUT, encoding="utf-8").read() if os.path.isfile(OUT) else ""
        sys.exit(0 if current == text else 1)
    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print("wrote", OUT)
