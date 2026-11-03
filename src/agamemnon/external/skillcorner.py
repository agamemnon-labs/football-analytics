"""SkillCorner Open Data (A-League 2024/25, broadcast tracking at 10 fps, MIT) to `Match`.

Labels follow docs/POSSESSION_RULES.md: a frame is alive when it lies inside a SkillCorner phase of
play (phases exist only while the ball is in play) and takes the frame's possession group; frames
inside a phase whose group is empty keep the last team that had the ball, which is the
competition's rule for loose balls. Everything outside a phase is dead.

Positions are resampled to 25 fps so every benchmark game shares the competition frame rate and
the decoder's durations mean the same thing everywhere.
"""

import json
import os

import numpy as np
import pandas as pd

from ..schema import A, D, H, X, Match
from . import EXTERNAL_DIR, cache

SOURCE_FPS = 10.0
TARGET_FPS = 25.0


def match_ids(root=None):
    root = root or os.path.join(EXTERNAL_DIR, "skillcorner")
    with open(os.path.join(root, "matches.json"), encoding="utf-8") as handle:
        return sorted(str(m["id"]) for m in json.load(handle))


def _read_tracking(path, slot_of, n_slots):
    frames, periods, groups, balls, players = [], [], [], [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            frames.append(row["frame"])
            periods.append(row["period"] or 0)
            groups.append({"home team": H, "away team": A}.get(row["possession"]["group"], X))
            ball = row["ball_data"]
            # Extrapolated ball positions are SkillCorner's guesses, not observations; the
            # competition tracking has no such fill, so they are dropped.
            if ball.get("is_detected") and ball["x"] is not None:
                balls.append((ball["x"], ball["y"]))
            else:
                balls.append((np.nan, np.nan))
            slot_xy = np.full((n_slots, 2), np.nan, dtype=np.float32)
            for p in row["player_data"] or ():
                slot = slot_of.get(p["player_id"])
                if slot is not None and p["x"] is not None:
                    slot_xy[slot] = (p["x"], p["y"])
            players.append(slot_xy)
    return (np.asarray(frames), np.asarray(periods), np.asarray(groups, dtype=np.int8),
            np.asarray(balls, dtype=np.float32), np.stack(players))


def _labels_10fps(frames, groups, phases):
    """H/A/D per source frame, from phases of play plus the per-frame possession group."""
    alive = np.zeros(len(frames), dtype=bool)
    phase_team = np.full(len(frames), X, dtype=np.int8)
    index = {f: i for i, f in enumerate(frames)}
    for start, stop, team in phases:
        lo, hi = index.get(start), index.get(stop)
        if lo is None or hi is None:
            continue
        alive[lo:hi + 1] = True
        phase_team[lo:hi + 1] = team
    labels = np.full(len(frames), D, dtype=np.int8)
    last = X
    for i in range(len(frames)):
        if not alive[i]:
            last = X
            continue
        if groups[i] in (H, A):
            last = groups[i]
        elif last == X:
            last = phase_team[i]
        labels[i] = last if last in (H, A) else phase_team[i]
    return labels


def _resample(values, t_src, t_dst):
    """Linear interpolation over time; NaN wherever either neighbour is NaN or outside the span."""
    idx = np.searchsorted(t_src, t_dst, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 2)
    w = (t_dst - t_src[idx]) / (t_src[idx + 1] - t_src[idx])
    shape = (-1,) + (1,) * (values.ndim - 1)
    w = w.reshape(shape).astype(np.float32)
    out = values[idx] * (1 - w) + values[idx + 1] * w
    outside = (t_dst < t_src[0]) | (t_dst > t_src[-1])
    out[outside] = np.nan
    return out


def convert(match_id, root=None):
    root = root or os.path.join(EXTERNAL_DIR, "skillcorner")
    folder = os.path.join(root, match_id)
    with open(os.path.join(folder, "{}_match.json".format(match_id)), encoding="utf-8") as handle:
        info = json.load(handle)
    home_id = info["home_team"]["id"]
    length, width = float(info["pitch_length"]), float(info["pitch_width"])
    roster = sorted(info["players"], key=lambda p: (p["team_id"] != home_id, p["id"]))
    slot_of = {p["id"]: i for i, p in enumerate(roster)}
    team = np.array([H if p["team_id"] == home_id else A for p in roster], dtype=np.int8)

    frames, periods, groups, ball, players = _read_tracking(
        os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(match_id)),
        slot_of, len(roster))
    phases = pd.read_csv(os.path.join(folder, "{}_phases_of_play.csv".format(match_id)))
    phase_rows = [(int(a), int(b), H if t == home_id else A)
                  for a, b, t in zip(phases.frame_start, phases.frame_end,
                                     phases.team_in_possession_id)]
    labels_src = _labels_10fps(frames, groups, phase_rows)

    shift = np.array([length / 2, width / 2], dtype=np.float32)
    ball = ball + shift
    players = players + shift

    out_ball, out_players, out_labels, out_periods, kickoffs = [], [], [], [], []
    offset = 0
    for period in (1, 2):
        sel = np.flatnonzero(periods == period)
        if not len(sel):
            continue
        t_src = frames[sel] / SOURCE_FPS
        n_dst = int(np.floor((t_src[-1] - t_src[0]) * TARGET_FPS)) + 1
        t_dst = t_src[0] + np.arange(n_dst) / TARGET_FPS
        out_ball.append(_resample(ball[sel], t_src, t_dst))
        out_players.append(_resample(players[sel], t_src, t_dst))
        nearest = np.searchsorted(t_src, t_dst + 1e-9, side="right") - 1
        lab = labels_src[sel][np.clip(nearest, 0, len(sel) - 1)]
        out_labels.append(lab)
        out_periods.append((offset, offset + n_dst))
        live = np.flatnonzero(lab != D)
        kickoffs.append(offset + int(live[0]) if len(live) else offset)
        offset += n_dst

    match = Match(
        name="skillcorner-{}".format(match_id), fps=TARGET_FPS, pitch=(length, width),
        ball=np.concatenate(out_ball).astype(np.float32),
        players=np.concatenate(out_players).astype(np.float32),
        team=team, periods=out_periods, kickoffs=kickoffs,
        labels=np.concatenate(out_labels).astype(np.int8),
    )
    return match.validate()


def load(match_id, root=None, refresh=False):
    name = "skillcorner-{}".format(match_id)
    match = None if refresh else cache.load(name)
    if match is None:
        match = convert(match_id, root)
        cache.save(match)
    return match
