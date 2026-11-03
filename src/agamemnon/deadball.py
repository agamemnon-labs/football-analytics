"""Dead-ball evidence: out of bounds, collective stoppage, set pieces, goals.

The ball is dead from the moment it leaves the pitch or play is stopped until the restart is
taken. The tracking shows that as a ball last seen beyond a line, a ball held still, players
walking rather than running, and, around kick-offs, both teams back in their own halves. Smart
Tagging adds SET PIECES and GOALS intervals, which are evidence of a stoppage nearby but are
themselves automatic and can be wrong or missing, so they enter as features with fitted weights
rather than as rules. The decoder turns this evidence into D runs; see decoder.py.
"""

import numpy as np

from .schema import A, GOALS, H, SET_PIECES

# Smart Tagging codes whose Team is the side in possession, and those whose Team is defending.
ATTACK_CODES = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"}
DEFEND_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}

DEAD_COLUMNS = ("bias", "seen", "log_age", "last_oob", "far_oob", "speed_1s", "speed_3s",
                "log_speed_3s", "q75_3s", "ball_still", "ball_speed", "edge", "n_players",
                "own_half", "centre", "setpiece", "post_goal")


def tag_features(match, prm):
    """(team prior in [-1, 1], set-piece window flag, post-goal flag) per frame."""
    n = match.n_frames
    fps = match.fps
    team = np.zeros(n, dtype=np.float32)
    setpiece = np.zeros(n, dtype=np.float32)
    post_goal = np.zeros(n, dtype=np.float32)
    for tag in match.tags:
        lo, hi = max(0, tag.start), min(n, tag.stop)
        if hi <= lo and tag.code != GOALS:
            continue
        sign = 0.0
        if tag.team in (H, A):
            sign = 1.0 if tag.team == H else -1.0
        if tag.code in ATTACK_CODES or tag.code == SET_PIECES:
            team[lo:hi] += sign
        elif tag.code in DEFEND_CODES:
            team[lo:hi] -= sign
        if tag.code == SET_PIECES:
            pre = max(0, lo - int(prm.SETPIECE_PRE_S * fps))
            setpiece[pre:hi] = 1.0
        if tag.code == GOALS:
            post_goal[hi:min(n, hi + int(prm.POST_GOAL_S * fps))] = 1.0
    return np.clip(team, -1, 1), setpiece, post_goal


def dead_design(F, match, prm, tags=None):
    n = match.n_frames
    seen = F["seen"]
    speed1 = np.nan_to_num(F["med_speed_1s"], nan=2.0)
    speed3 = np.nan_to_num(F["med_speed_3s"], nan=2.0)
    q75 = np.nan_to_num(F["q75_speed_3s"], nan=3.0)
    still = np.nan_to_num(F["ball_still"], nan=0.0)
    bspeed = np.clip(np.nan_to_num(F["ball_speed"], nan=0.0), 0, 30) / 10.0
    edge = np.clip(np.nan_to_num(F["edge_dist"], nan=20.0), -5, 20) / 10.0
    centre = np.where(F["have"] > 0, np.clip(F["centre_dist"] / 10.0, 0, 3), 3.0)
    if tags is None:
        setpiece = post_goal = np.zeros(n, np.float32)
    else:
        setpiece, post_goal = tags[1], tags[2]
    return np.column_stack([
        np.ones(n), seen, np.log1p(np.minimum(F["age_s"], 600.0)), F["last_oob"], F["far_oob"],
        speed1, speed3, np.log(speed3 + 0.2), q75, still, bspeed, edge, F["n_players"] / 22.0,
        np.nan_to_num(F["own_half"], nan=0.5), centre, setpiece, post_goal,
    ]).astype(np.float32)
